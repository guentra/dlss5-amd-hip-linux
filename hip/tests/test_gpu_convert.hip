// Bit-exactness test: the GPU pixel conversion kernels (kernels/gpu_convert.hip) must
// reproduce the hook's CPU pipeline byte-for-byte:
//   in : NativePixelsToRgbaF32 -> NativeResizeChannels(->1920x1080, 4ch)
//   out: NativeResizeChannels(1920x1080 -> w x h, 3ch) -> NativeRgbF32ToPixels
// across all supported formats and several geometries (identity + bilinear).
#include "dlss5_gpu_interop.hpp"
#include "dlss5_runtime.hpp"
#include "../../src/native_frame_input_check.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace dlss5;
using namespace dlss5::gpu;

// Verbatim copy of NativeResizeChannels from src/native_hip_client.h (portable part).
static void NativeResizeChannels(const float*src,unsigned sw,unsigned sh,float*dst,unsigned dw,unsigned dh,unsigned ch){
 if(!src||!dst||!sw||!sh||!dw||!dh||!ch)return;
 if(sw==dw&&sh==dh){std::memcpy(dst,src,size_t(dw)*dh*ch*sizeof(float));return;}
 for(unsigned y=0;y<dh;y++){
  float fy=(y+0.5f)*float(sh)/float(dh)-0.5f;
  int y0=int(floorf(fy)),y1=y0+1;float ty=fy-float(y0);
  if(y0<0){y0=y1=0;ty=0.f;}else if(y0>=int(sh-1)){y0=y1=int(sh-1);ty=0.f;}
  for(unsigned x=0;x<dw;x++){
   float fx=(x+0.5f)*float(sw)/float(dw)-0.5f;
   int x0=int(floorf(fx)),x1=x0+1;float tx=fx-float(x0);
   if(x0<0){x0=x1=0;tx=0.f;}else if(x0>=int(sw-1)){x0=x1=int(sw-1);tx=0.f;}
   const float*p00=src+(size_t(unsigned(y0))*sw+unsigned(x0))*ch;
   const float*p10=src+(size_t(unsigned(y0))*sw+unsigned(x1))*ch;
   const float*p01=src+(size_t(unsigned(y1))*sw+unsigned(x0))*ch;
   const float*p11=src+(size_t(unsigned(y1))*sw+unsigned(x1))*ch;
   float*o=dst+(size_t(y)*dw+x)*ch;
   for(unsigned c=0;c<ch;c++)
    o[c]=(1.f-tx)*(1.f-ty)*p00[c]+tx*(1.f-ty)*p10[c]+(1.f-tx)*ty*p01[c]+tx*ty*p11[c];
  }
 }
}

static const char* NativePixelFormatName(NativePixelFormat f){
 switch(f){
 case NativePixelFormat::rgba8_unorm:return "rgba8";
 case NativePixelFormat::bgra8_unorm:return "bgra8";
 case NativePixelFormat::rgba16_unorm:return "r16u";
 case NativePixelFormat::rgba16_float:return "rgba16f";
 default:return "?";
 }
}

static std::vector<unsigned char> make_pixels(unsigned w,unsigned h,unsigned dxgi,std::mt19937& rng){
 const size_t bpp=NativePixelBytes(NativePixelFormatFromDxgi(dxgi));
 std::vector<unsigned char> pix(size_t(w)*h*bpp);
 std::uniform_int_distribution<int> db(0,255);
 for(auto& c:pix) c=(unsigned char)db(rng);
 if(dxgi==10){ // RGBA16 float: force a spread of representable values (incl. denormals, extremes)
   std::uniform_real_distribution<float> df(-2.f,4.f);
   for(size_t i=0;i<pix.size();i+=2){
     float v=df(rng);
     uint16_t hbf=NativeFloatToHalf(v);
     pix[i]=(unsigned char)hbf; pix[i+1]=(unsigned char)(hbf>>8);
   }
 }
 if(dxgi==11){
   std::uniform_int_distribution<int> d16(0,65535);
   for(size_t i=0;i<pix.size();i+=2){
     uint16_t v=(uint16_t)d16(rng);
     pix[i]=(unsigned char)v; pix[i+1]=(unsigned char)(v>>8);
   }
 }
 return pix;
}

static int run_in_case(unsigned dxgi,unsigned w,unsigned h,std::mt19937& rng){
 auto pix=make_pixels(w,h,dxgi,rng);
 auto fmt=NativePixelFormatFromDxgi(dxgi);
 // CPU in-pipeline
 std::vector<float> rgba;
 if(!NativePixelsToRgbaF32(pix.data(),fmt,w,h,rgba))return -1;
 std::vector<float> net(1920ull*1080*4);
 NativeResizeChannels(rgba.data(),w,h,net.data(),1920,1080,4);
 // GPU in-pipeline
 DeviceBuf graw(pix.size()),gnet(1920ull*1080*16);
 graw.upload(pix.data(),pix.size());
 launch_px_to_rgba((const uint8_t*)graw.p,gnet.as<float>(),w,h,dxgi);
 hipStreamSynchronize(nullptr);
 std::vector<float> gnet_host(1920ull*1080*4);
 gnet.download(gnet_host.data(),gnet_host.size()*4);
 size_t bad=0; long long first=-1;
 for(size_t i=0;i<net.size();++i)if(std::memcmp(&net[i],&gnet_host[i],4)){if(first<0)first=(long long)i;++bad;}
 if(bad){printf("IN  %s %ux%u: mismatches=%zu first=%lld cpu=%.9g gpu=%.9g\n",NativePixelFormatName(fmt),w,h,bad,first,(double)net[(size_t)first],(double)gnet_host[(size_t)first]);return 1;}
 return 0;
}

static int run_out_case(unsigned dxgi,unsigned w,unsigned h,std::mt19937& rng){
 auto fmt=NativePixelFormatFromDxgi(dxgi);
 // CPU source: random RGBA-ish RGB in [-0.5, 1.5] (exercises clamping) + distinct alpha.
 std::uniform_real_distribution<float> df(-0.5f,1.5f);
 std::vector<float> net(1920ull*1080*3);
 for(auto& v:net)v=df(rng);
 // Original packed pixels (alpha source) — CPU requires dst contains the originals.
 std::vector<unsigned char> orig=make_pixels(w,h,dxgi,rng);
 std::vector<unsigned char> out_cpu=orig;
 std::vector<float> rgb(size_t(w)*h*3);
 NativeResizeChannels(net.data(),1920,1080,rgb.data(),w,h,3);
 const bool ok=NativeRgbF32ToPixels(rgb.data(),out_cpu.data(),fmt,w,h);
 // GPU out-pipeline
 DeviceBuf gnet(net.size()*4),gout(orig.size()),gorig(orig.size());
 gnet.upload(net.data(),net.size()*4);
 gorig.upload(orig.data(),orig.size());
 launch_rgb_to_pix(gnet.as<float>(),(const uint8_t*)gorig.p,(uint8_t*)gout.p,w,h,dxgi);
 hipStreamSynchronize(nullptr);
 std::vector<unsigned char> out_gpu(orig.size());
 gout.download(out_gpu.data(),orig.size());
 if(!ok){ // CPU rejected the frame (nonfinite/inf half): GPU must leave the original intact
   size_t bad=0;for(size_t i=0;i<orig.size();++i)if(out_gpu[i]!=orig[i])++bad;
   if(bad){printf("OUT %s %ux%u (cpu-reject): gpu modified %zu bytes\n",NativePixelFormatName(fmt),w,h,bad);return 1;}
   return 0;
 }
 size_t bad=0; long long first=-1;
 for(size_t i=0;i<orig.size();++i)if(out_gpu[i]!=out_cpu[i]){if(first<0)first=(long long)i;++bad;}
 if(bad){printf("OUT %s %ux%u: mismatches=%zu first=%lld cpu=%02x gpu=%02x\n",NativePixelFormatName(fmt),w,h,bad,first,out_cpu[(size_t)first],out_gpu[(size_t)first]);return 1;}
 return 0;
}

// Temporal blend kernels: the gated variant must blend where the frames agree,
// pass moving content through untouched, and scale linearly between; the plain
// variant keeps its legacy formula.
static int run_blend_case(){
 constexpr int n=12; // 4 pixels x RGB
 float hcur[12]={0.5f,0.5f,0.5f,   // p0: |d|=0.004 (<=lo) -> full gate
                 0.5f,0.2f,0.9f,   // p1: one channel |d|=0.3 (>=hi) -> per-channel gate
                 0.5f,0.5f,0.5f,   // p2: |d|=0.03 (mid) -> half gate
                 0.3f,0.3f,0.3f};  // p3: identical -> no-op
 float hprev[12]={0.496f,0.496f,0.496f,
                  0.8f,0.2f,0.9f,
                  0.53f,0.53f,0.53f,
                  0.3f,0.3f,0.3f};
 const float w=0.5f,lo=0.02f,hi=0.04f;
 DeviceBuf gcur(size_t(n)*4),gprev(size_t(n)*4);
 gcur.upload(hcur,size_t(n)*4);
 gprev.upload(hprev,size_t(n)*4);
 std::vector<float> gout(n);
 launch_temporal_blend_gated(gcur.as<float>(),gprev.as<float>(),n,w,lo,hi,nullptr);
 hipStreamSynchronize(nullptr);
 gcur.download(gout.data(),size_t(n)*4);
 for(int i=0;i<n;++i){
   const float d=fabsf(hcur[i]-hprev[i]);
   const float g=fminf(fmaxf((hi-d)*(1.f/(hi-lo)),0.f),1.f);
   const float k=w*g;
   const float exp=(1.f-k)*hcur[i]+k*hprev[i];
   if(std::memcmp(&gout[i],&exp,4)){
     printf("BLEND gated i=%d: gpu=%.9g expected=%.9g (cur=%.9g prev=%.9g)\n",
            i,(double)gout[i],(double)exp,(double)hcur[i],(double)hprev[i]);
     return 1;
   }
 }
 gcur.upload(hcur,size_t(n)*4);
 launch_temporal_blend(gcur.as<float>(),gprev.as<float>(),n,w,nullptr);
 hipStreamSynchronize(nullptr);
 gcur.download(gout.data(),size_t(n)*4);
 for(int i=0;i<n;++i){
   const float exp=(1.f-w)*hcur[i]+w*hprev[i];
   if(std::memcmp(&gout[i],&exp,4)){
     printf("BLEND plain i=%d: gpu=%.9g expected=%.9g\n",i,(double)gout[i],(double)exp);
     return 1;
   }
 }
 return 0;
}

int main(){
 std::mt19937 rng(0x5eed);
 struct Geo{unsigned w,h;};
 const Geo geos[]={ {1920,1080},{2560,1440},{1280,720},{1921,1081},{3840,2160},{16,9} };
 int fails=0;
 for(unsigned dxgi:{28u,87u,11u,10u})
  for(const auto& g:geos){
    int rc=run_in_case(dxgi,g.w,g.h,rng);
    if(rc){if(rc>0)++fails;continue;}
    rc=run_out_case(dxgi,g.w,g.h,rng);
    if(rc)++fails;
  }
 if(run_blend_case())++fails;
 printf("gpu_convert bit-exact: %s (%d failures)\n",fails?"FAIL":"PASS",fails);
 return fails?1:0;
}
