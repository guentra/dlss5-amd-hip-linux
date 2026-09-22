// Frozen pre-optimization kernel as a numerical differential oracle.
#include "dlss5_runtime.hpp"
#include "dlss5_launch.hpp"
#include "dlss5_linalg.hpp"
#include <cstdio>
#include <vector>
using namespace dlss5;
__global__ void reference_linear(const float* input, const u8* weights, float* output,
                             uint m, uint n, uint k, const float* residual,
                             const float* scales, int mode, int ordered, bool matrix_residual = false) {
    using namespace linalg;
    const uint first = blockIdx.x * 16, col = blockIdx.y * 16;
    __shared__ u8 tile[512];
    C32 acc = C32::Splat(0.f);
    if (residual && mode != 4)
        for (uint i = 0; i < acc.Length(); ++i) {
            uint2 rc = acc.GetCoordinate(i);
            if (first + rc.x < m)
                acc.Set(i, matrix_residual
                    ? residual[(first + rc.x) * n + col + rc.y] * scales[col + rc.y]
                    : H(residual[(first + rc.x) * n + col + rc.y] * scales[col + rc.y]));
        }
    C32 partition = C32::Splat(0.f);
    for (uint g = 0; g < k / 32; ++g) {
        for (uint i = threadIdx.x; i < 512; i += blockDim.x) {
            const uint row = first + i / 32;
            tile[i] = row < m ? e4m3_byte(input[row * k + g * 32 + i % 32]) : 0;
        }
        __syncthreads();
        A8 a = A8::Load(tile, 0, 32);
        B8 b = B8::LoadRowT(weights, ((col / 16) * (k / 32) + g) * 512, 16);
        if (ordered == 2) {
            partition.MultiplyAccumulate(a,b);
            if ((g+1) % (k/128) == 0) {
                for(uint i=0;i<acc.Length();++i) acc.Set(i,acc.Get(i)+partition.Get(i));
                partition=C32::Splat(0.f);
            }
        } else if (ordered == 1) {
            C32 part = Multiply(a, b);
            for (uint i = 0; i < acc.Length(); ++i) acc.Set(i, H(acc.Get(i) + part.Get(i)));
        } else acc.MultiplyAccumulate(a, b);
        __syncthreads();
    }
    for (uint i = 0; i < acc.Length(); ++i) {
        uint2 rc = acc.GetCoordinate(i);
        float v = acc.Get(i);
        if (mode == 4 && residual && first + rc.x < m)
            v += residual[(first + rc.x) * n + col + rc.y] * scales[col + rc.y];
        if (mode == 2) v = from_e4m3(e4m3_byte(ActivatePoly(v)));
        else if (mode == 1) v = from_e4m3(e4m3_byte(H(v)));
        else if (mode == 5) v = from_e4m3(e4m3_byte(v));
        else if (mode == 0 || mode == 4) v = H(v);
        if (first + rc.x < m) output[(first + rc.x) * n + col + rc.y] = v;
    }
}

int main(int argc,char**argv){try{
 hip_check(hipSetDevice(argc>1?std::atoi(argv[1]):0),"device");
 unsigned cases=0;float speedup=0;
 for(uint m:{17u,32768u})for(uint n:{16u,32u,48u,96u,256u})for(uint k:{32u,128u})
 for(int mode=0;mode<6;++mode)for(int ordered=0;ordered<3;++ordered){
 if(ordered==2&&k%128)continue;
 std::vector<float> in(size_t(m)*k),w(size_t(n)*k),res(size_t(m)*n),sc(n);
 for(size_t i=0;i<in.size();++i)in[i]=(int(i*13%67)-33)*.0078125f;
 for(size_t i=0;i<w.size();++i)w[i]=(int(i*7%31)-15)*.015625f;
 for(size_t i=0;i<res.size();++i)res[i]=(int(i%43)-21)*.0023f;
 for(uint i=0;i<n;++i)sc[i]=.1f+i*.002f;
 std::vector<u8> pw(w.size());pack_tiled_e4m3(pw.data(),w.data(),n,k);
 DeviceBuf di(in.size()*4),dw(pw.size()),dr(res.size()*4),ds(n*4),da(res.size()*4+64),db(res.size()*4+64);
 di.upload(in.data(),in.size()*4);dw.upload(pw.data(),pw.size());dr.upload(res.data(),res.size()*4);ds.upload(sc.data(),n*4);
 hip_check(hipMemset(da.p,0xa5,da.bytes),"guard");hip_check(hipMemset(db.p,0xa5,db.bytes),"guard");
 const float* rp=cases%2?dr.as<float>():nullptr;bool matrix=cases%3==0;
 auto ref=[&]{reference_linear<<<dim3((m+15)/16,n/16),32>>>(di.as<float>(),dw.as<u8>(),da.as<float>(),m,n,k,rp,ds.as<float>(),mode,ordered,matrix);};
 auto opt=[&]{launch_linear_f32(di.as<float>(),dw.as<u8>(),db.as<float>(),m,n,k,rp,ds.as<float>(),mode,ordered,matrix);};
 ref();opt();hip_check(hipDeviceSynchronize(),"run");std::vector<u8>a(da.bytes),b(db.bytes);da.download(a.data(),a.size());db.download(b.data(),b.size());
 if(a!=b){printf("shape=%u,%u,%u mode=%d ordered=%d\n",m,n,k,mode,ordered);throw std::runtime_error("linear bytes/guard mismatch");}
 if(m==32768&&n==96&&k==32&&mode==3&&ordered==0){
 hipEvent_t start,end;hip_check(hipEventCreate(&start),"event");hip_check(hipEventCreate(&end),"event");
 auto time=[&](auto fn){hip_check(hipEventRecord(start),"start");for(int i=0;i<20;++i)fn();hip_check(hipEventRecord(end),"end");hip_check(hipEventSynchronize(end),"sync");float ms;hip_check(hipEventElapsedTime(&ms,start,end),"ms");return ms/20;};
 float old=time(ref),now=time(opt);speedup=old/now;printf("linear reference_ms=%.6f candidate_ms=%.6f speedup=%.3f\n",old,now,speedup);hip_check(hipEventDestroy(start),"destroy");hip_check(hipEventDestroy(end),"destroy");}
 ++cases;
 }
 printf("PASS linear %u cases exact bytes and guards\n",cases);
 if(argc>2&&speedup<1.3f)throw std::runtime_error("linear performance: need >=1.3x baseline");
 return 0;
 }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
