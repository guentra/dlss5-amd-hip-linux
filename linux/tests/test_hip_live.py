"""Executable CPU tests: verbatim live header, real pixel/resize/identity helpers.
Only Windows, D3D and HIP execution edges are simulated. No GPU or game.

The stub emulates the V3 GPU-resident design: shared handles resolve to the
resource's byte range, so RunFrameRaw simulates the .so writing network output
(RGB replaced, alpha exact) directly into the out buffer.
"""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STUB = r'''
#pragma once
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>
using UINT=unsigned;using UINT64=uint64_t;using ULONG=unsigned long;using DWORD=unsigned;using HRESULT=int;
using SIZE_T=size_t;using HMODULE=void*;using LPCWSTR=const wchar_t*;using HANDLE=void*;
#define STDMETHODCALLTYPE
#define S_OK 0
#define E_NOINTERFACE -1
#define E_POINTER -2
#define FAILED(x) ((x)<0)
#define SUCCEEDED(x) ((x)>=0)
struct GUID {unsigned Data1;unsigned short Data2,Data3;unsigned char Data4[8];};
using REFIID=const GUID&;inline const GUID IID_IUnknown{};
inline bool operator==(REFIID a,REFIID b){return !memcmp(&a,&b,sizeof a);}
#define IID_PPV_ARGS(x) IID_IUnknown,reinterpret_cast<void**>(x)
struct IUnknown {virtual HRESULT QueryInterface(REFIID id,void**p){if(id==IID_IUnknown){*p=this;AddRef();return 0;}*p=nullptr;return -1;}virtual ULONG AddRef()=0;virtual ULONG Release()=0;virtual ~IUnknown()=default;};
inline std::atomic<int> objects{0}, maps{0}, key_samples{0}, runs{0}, clients{0};
inline std::atomic<bool> key_down{false},fail_hip{false},block_run{false},entered_run{false};
inline bool fail_map=false,fail_private=false,accept_marker=true,fail_shared=false;inline int fail_create=0,creates=0;
inline bool srgb=false;inline int last_seed=-1;inline unsigned last_temporal_gen=0;inline std::string matched_adapter;
inline void(*desc_hook)()=nullptr;
inline std::atomic<bool> init_entered{false},block_init{false};
inline short GetAsyncKeyState(int){++key_samples;return key_down?short(0x8000):0;}
#define VK_F6 0x75
#define GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS 4
#define GET_MODULE_HANDLE_EX_FLAG_PIN 1
#define GENERIC_READ 0x80000000u
#define GENERIC_WRITE 0x40000000u
inline int CloseHandle(HANDLE){return 1;}
inline int GetModuleHandleExW(int,LPCWSTR,HMODULE*p){*p=(void*)1;return 1;}
inline unsigned long GetCurrentProcessId(){return 1;}
inline unsigned long long GetTickCount64(){return 1;}
inline const wchar_t* _wgetenv(const wchar_t*){return srgb?L"1":nullptr;}
inline FILE* _wfopen(const wchar_t*,const wchar_t*){return nullptr;}
inline int CreateDirectoryW(const wchar_t*,void*){return 1;}
#define CP_UTF8 65001
inline int WideCharToMultiByte(unsigned,int,const wchar_t*w,int,char*out,int n,void*,void*){std::string s;while(*w)s+=char(*w++);if(!out)return int(s.size()+1);if(n<int(s.size()+1))return 0;memcpy(out,s.c_str(),s.size()+1);return int(s.size()+1);}
struct Ref: IUnknown {std::atomic<ULONG> refs{1};Ref(){++objects;}~Ref(){--objects;}ULONG AddRef()override{return ++refs;}ULONG Release()override{auto n=--refs;if(!n)delete this;return n;}};
struct LUID{unsigned LowPart;int HighPart;};
using DXGI_FORMAT=unsigned;
struct DXGI_ADAPTER_DESC1{wchar_t Description[128];};
inline const wchar_t* adapter_description=L"Exact GPU Name";
struct IDXGIAdapter1:Ref{HRESULT GetDesc1(DXGI_ADAPTER_DESC1*d){wcscpy(d->Description,adapter_description);return 0;}};
struct IDXGIFactory4:Ref{HRESULT EnumAdapterByLuid(LUID,REFIID,void**p){*p=new IDXGIAdapter1;return 0;}};
inline HRESULT CreateDXGIFactory1(REFIID,void**p){*p=new IDXGIFactory4;return 0;}
using D3D12_RESOURCE_STATES=unsigned;
constexpr unsigned D3D12_RESOURCE_STATE_COMMON=0,D3D12_RESOURCE_STATE_RENDER_TARGET=4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS=8,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE=64,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE=128,D3D12_RESOURCE_STATE_COPY_DEST=1024,D3D12_RESOURCE_STATE_COPY_SOURCE=2048,D3D12_RESOURCE_STATE_GENERIC_READ=2755;
constexpr unsigned D3D12_RESOURCE_DIMENSION_TEXTURE2D=3,D3D12_RESOURCE_DIMENSION_BUFFER=1,D3D12_TEXTURE_LAYOUT_ROW_MAJOR=1,D3D12_RESOURCE_FLAG_NONE=0,D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS=32,D3D12_HEAP_TYPE_DEFAULT=0,D3D12_HEAP_TYPE_READBACK=3,D3D12_HEAP_TYPE_UPLOAD=2,D3D12_HEAP_FLAG_NONE=0,D3D12_HEAP_FLAG_SHARED=0x10,D3D12_RESOURCE_BARRIER_TYPE_TRANSITION=0,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES=~0u,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX=0,D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT=1,D3D12_COMMAND_LIST_TYPE_DIRECT=0;
struct D3D12_RESOURCE_DESC{unsigned Dimension{};UINT64 Alignment{},Width{};UINT Height{};unsigned short DepthOrArraySize{},MipLevels{};DXGI_FORMAT Format{};struct{UINT Count{},Quality{};}SampleDesc;unsigned Layout{},Flags{};};
struct D3D12_HEAP_PROPERTIES{unsigned Type{},CPUPageProperty{},MemoryPoolPreference{},CreationNodeMask{},VisibleNodeMask{};};
struct D3D12_PLACED_SUBRESOURCE_FOOTPRINT{UINT64 Offset{};struct{DXGI_FORMAT Format{};UINT Width{},Height{},Depth{},RowPitch{};}Footprint;};
struct D3D12_RANGE{SIZE_T Begin{},End{};};
struct ID3D12Device;
struct ID3D12Resource:Ref{ID3D12Device*d;D3D12_RESOURCE_DESC desc;std::vector<unsigned char> bytes;ID3D12Resource(ID3D12Device*,D3D12_RESOURCE_DESC);~ID3D12Resource();D3D12_RESOURCE_DESC GetDesc(){if(desc_hook){auto fn=desc_hook;desc_hook=nullptr;fn();}return desc;}HRESULT GetDevice(REFIID,void**p);HRESULT Map(UINT,const D3D12_RANGE*,void**p){++maps;if(fail_map){*p=nullptr;return -1;}*p=bytes.data();return 0;}void Unmap(UINT,const D3D12_RANGE*){}};
struct ID3D12Device:Ref{LUID GetAdapterLuid(){return {1,0};}HRESULT GetDeviceRemovedReason(){return 0;}void GetCopyableFootprints(const D3D12_RESOURCE_DESC*d,UINT,UINT,UINT64,D3D12_PLACED_SUBRESOURCE_FOOTPRINT*f,UINT*r,UINT64*b,UINT64*t){unsigned bp=(d->Format==10||d->Format==11)?8:4;f->Offset=0;f->Footprint={d->Format,UINT(d->Width),d->Height,1,UINT((d->Width*bp+255)&~255ull)};if(r)*r=d->Height;if(b)*b=d->Width*bp;*t=UINT64(f->Footprint.RowPitch)*d->Height;}HRESULT CreateCommittedResource(const D3D12_HEAP_PROPERTIES*,unsigned,const D3D12_RESOURCE_DESC*d,unsigned,const void*,REFIID,void**p){if(++creates==fail_create){*p=nullptr;return -1;}*p=new ID3D12Resource(this,*d);return 0;}HRESULT CreateSharedHandle(ID3D12Resource*r,const void*,DWORD,void*,HANDLE*h){if(fail_shared){*h=nullptr;return -1;}*h=r->bytes.data();return 0;}};
inline ID3D12Resource::ID3D12Resource(ID3D12Device*x,D3D12_RESOURCE_DESC a):d(x),desc(a),bytes(a.Dimension==1?a.Width:a.Width*a.Height*((a.Format==10||a.Format==11)?8:4)){d->AddRef();}
inline ID3D12Resource::~ID3D12Resource(){d->Release();}
inline HRESULT ID3D12Resource::GetDevice(REFIID,void**p){*p=d;d->AddRef();return 0;}
struct D3D12_RESOURCE_BARRIER{unsigned Type{},Flags{};struct{ID3D12Resource*pResource;UINT Subresource;unsigned StateBefore,StateAfter;}Transition;};
struct D3D12_TEXTURE_COPY_LOCATION{ID3D12Resource*pResource{};unsigned Type{};D3D12_PLACED_SUBRESOURCE_FOOTPRINT PlacedFootprint{};UINT SubresourceIndex{};};
#include "../hip/include/dlss5_submit.h"
struct ID3D12GraphicsCommandList:Ref{ID3D12Device*d;unsigned barriers=0,copies=0,markers=0;Dlss5SubmitMarker marker{};std::map<unsigned,IUnknown*> private_refs;ID3D12Resource*readback=nullptr,*upload=nullptr,*color=nullptr;ID3D12GraphicsCommandList(ID3D12Device*x):d(x){d->AddRef();}~ID3D12GraphicsCommandList(){ResetAllocator();for(auto&entry:private_refs)entry.second->Release();d->Release();}unsigned GetType(){return 0;}HRESULT GetDevice(REFIID,void**p){*p=d;d->AddRef();return 0;}HRESULT SetPrivateDataInterface(REFIID id,const IUnknown*p){if(fail_private)return -1;auto it=private_refs.find(id.Data1);if(p)const_cast<IUnknown*>(p)->AddRef();if(it!=private_refs.end()){it->second->Release();private_refs.erase(it);}if(p)private_refs[id.Data1]=const_cast<IUnknown*>(p);return 0;}void ResourceBarrier(UINT n,const D3D12_RESOURCE_BARRIER*){barriers+=n;}void CopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION*dst,UINT,UINT,UINT,const D3D12_TEXTURE_COPY_LOCATION*src,const void*){++copies;if(dst->Type==1){readback=dst->pResource;color=src->pResource;auto f=dst->PlacedFootprint;size_t row=color->bytes.size()/color->desc.Height;for(unsigned y=0;y<color->desc.Height;y++)memcpy(readback->bytes.data()+f.Offset+y*f.Footprint.RowPitch,color->bytes.data()+y*row,row);}else upload=src->pResource;}void CopyResource(ID3D12Resource*dst,ID3D12Resource*src){++copies;memcpy(dst->bytes.data(),src->bytes.data(),dst->bytes.size());}void SetMarker(UINT meta,const void*p,UINT n){++markers;assert(meta==DLSS5_SUBMIT_METADATA&&n==sizeof(Dlss5SubmitMarker));auto&m=*static_cast<const Dlss5SubmitMarker*>(p);assert(m.magic==DLSS5_SUBMIT_MAGIC&&m.version==1&&m.bytes==n);if(accept_marker){marker=m;marker.retain(marker.context);*m.accepted=1;marker.accepted=nullptr;}}int Run(uint64_t q=42){assert(marker.context);return marker.run(marker.context,q);}void ResetAllocator(){if(marker.context){marker.release(marker.context);marker={};}}};
'''
CLIENT = r'''
#pragma once
#include "windows.h"
#include "native_frame_input_check.h"
inline std::wstring NativeLabPath(const wchar_t*p){return p;}
class NativeHipClient{public:NativeHipClient(){++clients;}~NativeHipClient(){--clients;}
void Create(const char*n){matched_adapter=n;init_entered=true;while(block_init)std::this_thread::yield();}
bool Ready()const{return true;}
bool HasRawGpu()const{return true;}
int RunFrameRaw(void*in,void*out,unsigned w,unsigned h,unsigned dxgi,unsigned seed,bool display,bool host_ptrs,unsigned gen,unsigned temporal_gen){
(void)in;(void)host_ptrs;(void)gen;last_temporal_gen=temporal_gen;++runs;last_seed=int(seed);assert(display==srgb);
entered_run=true;while(block_run)std::this_thread::yield();
if(fail_hip)return -1;
unsigned bpp=(dxgi==10||dxgi==11)?8:4;
auto*o=static_cast<unsigned char*>(out);
for(size_t i=0;i<size_t(w)*h;i++){o[i*bpp]=191;o[i*bpp+1]=191;o[i*bpp+2]=191;}
return 0;}};
'''
BODY = r'''
#include "native_hip_live.h"
struct Fixture{ID3D12Device*d=new ID3D12Device;ID3D12Resource*c;ID3D12GraphicsCommandList*l=new ID3D12GraphicsCommandList(d);Fixture(){D3D12_RESOURCE_DESC a{};a.Dimension=3;a.Width=17;a.Height=19;a.DepthOrArraySize=a.MipLevels=a.SampleDesc.Count=1;a.Format=28;c=new ID3D12Resource(d,a);for(size_t i=0;i<c->bytes.size();i++)c->bytes[i]=static_cast<unsigned char>(i%251);}~Fixture(){l->Release();c->Release();d->Release();}};
static bool record(NativeHipLive&live,Fixture&f,uint64_t id=7){return live.Record(f.l,f.c,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,id);}
static void ready(NativeHipLive&live,Fixture&f){auto target=f.l->markers+1;assert(!record(live,f));for(int i=0;i<2000&&f.l->markers<target;i++){std::this_thread::sleep_for(std::chrono::milliseconds(1));record(live,f);}assert(f.l->markers==target);}
'''


def run_cpp(body):
    with tempfile.TemporaryDirectory() as directory:
        tmp = pathlib.Path(directory)
        (tmp / 'src').mkdir()
        (tmp / 'hip/include').mkdir(parents=True)
        for name in ('native_hip_live.h', 'native_frame_input_check.h', 'native_device_identity.h'):
            path = ROOT / 'src' / name
            if not path.exists():
                raise AssertionError(f'missing implementation: {name}')
            shutil.copyfile(path, tmp / 'src' / name)
        shutil.copyfile(ROOT / 'hip/include/dlss5_submit.h', tmp / 'hip/include/dlss5_submit.h')
        (tmp / 'src/windows.h').write_text(STUB)
        for name in ('d3d12.h', 'dxgi1_4.h', 'unknwn.h'):
            (tmp / 'src' / name).write_text('#include "windows.h"\n')
        # Keep the production resize implementation verbatim, not a fake resize.
        source = (ROOT / 'src/native_hip_client.h').read_text()
        resize = source[source.index('inline constexpr unsigned NativeHipNetWidth'):source.index('// Pixel conversion lives')]
        (tmp / 'src/native_hip_client.h').write_text(CLIENT + resize)
        (tmp / 'test.cpp').write_text(BODY + '\nint main(){\n' + body + '\n}\n')
        build = subprocess.run(['g++', '-std=c++17', '-O1', '-g', '-Wall', '-Wextra', '-pthread',
                                '-I', str(tmp / 'src'), str(tmp / 'test.cpp'), '-o', str(tmp / 'test')],
                               capture_output=True, text=True, timeout=40)
        if build.returncode:
            raise AssertionError('compile failed:\n' + build.stdout + build.stderr)
        result = subprocess.run([str(tmp / 'test')], capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError('live harness failed:\n' + result.stdout + result.stderr)


class HipLiveTests(unittest.TestCase):
    def test_radv_decoration_is_removed_for_exact_hip_match(self):
        run_cpp(r'''
adapter_description=L"Exact GPU Name (RADV GFX1201)";
{Fixture f;NativeHipLive live;ready(live,f);assert(matched_adapter=="Exact GPU Name");}
assert(objects==0&&clients==0);
''')
    def test_callback_network_rows_alpha_and_lifetime(self):
        run_cpp(r'''
{Fixture f;{NativeHipLive live;ready(live,f);assert(matched_adapter=="Exact GPU Name");
assert(f.l->copies==3&&f.l->barriers==7&&runs==0);assert(f.l->private_refs.empty());}
assert(clients==1);assert(f.l->Run()==0);assert(runs==1&&last_seed==7);
for(unsigned y=0;y<f.c->desc.Height;y++)for(unsigned x=0;x<f.c->desc.Width;x++){
auto*p=f.l->upload->bytes.data()+y*17*4+x*4;assert(p[0]==191&&p[1]==191&&p[2]==191);
assert(p[3]==f.c->bytes[(y*f.c->desc.Width+x)*4+3]);}
assert(f.l->Run()!=0);assert(runs==1);f.l->ResetAllocator();assert(clients==0);}
assert(objects==0);
''')

    def test_toggle_during_record_does_not_queue_when_now_bypassed(self):
        run_cpp(r'''
{Fixture f;NativeHipLive live;ready(live,f);f.l->ResetAllocator();
// Hold F6: exactly one edge, one sample per Record, no extra reads in callback.
int before=key_samples;key_down=true;assert(!record(live,f));assert(!record(live,f));
assert(key_samples==before+2);key_down=false;assert(!record(live,f));
key_down=true;assert(record(live,f,100));assert(f.l->Run(100)==0);assert(last_seed==100);}
assert(objects==0&&clients==0);
''')

    def test_temporal_generation_reaches_the_bridge_and_tracks_f6(self):
        """temporal_gen must actually be forwarded, and must change on bypass.

        The bridge resets its temporal blend whenever this token changes, so
        that a network frame is never mixed with a pre-bypass previous frame.
        A caller that forwards a constant - or drops the argument - produces no
        error anywhere: frames keep rendering and the blend is simply wrong
        across an F6 toggle. This pins it to the bypass generation instead."""
        run_cpp(r'''
{Fixture f;NativeHipLive live;ready(live,f);
assert(f.l->Run()==0&&runs==1);
const unsigned first=last_temporal_gen;
// Two toggles: bypass (odd generation), then enabled again (even, and
// different from where it started).
// Toggle through the null-list form, which only samples the key: it
// creates no job and so cannot leave one anchored.
key_down=true;live.Record(nullptr,nullptr,0,20);key_down=false;live.Record(nullptr,nullptr,0,21);
key_down=true;live.Record(nullptr,nullptr,0,22);key_down=false;
f.l->ResetAllocator();
bool ran=false;
for(int i=0;i<2000&&!ran;i++){
 record(live,f,uint64_t(30+i));
 if(f.l->marker.context){assert(f.l->Run()==0);ran=runs==2;}
 f.l->ResetAllocator();
}
assert(ran);assert(last_temporal_gen!=first);}
assert(objects==0&&clients==0);
''')

    def test_f6_toggle_after_record_copies_original_even_after_reenable(self):
        run_cpp(r'''
{Fixture f;NativeHipLive live;ready(live,f);int before=key_samples;
key_down=true;assert(!record(live,f));key_down=false;assert(!record(live,f));
key_down=true;live.Record(nullptr,nullptr,0,8);assert(key_samples==before+3);
assert(f.l->Run()==0&&runs==0);assert(key_samples==before+3);
for(unsigned y=0;y<f.c->desc.Height;y++)assert(!memcmp(f.l->upload->bytes.data()+y*17*4,f.c->bytes.data()+y*17*4,17*4));}
assert(objects==0&&clients==0);
''')

    def test_toggle_while_hip_running_keeps_new_pixels_and_bypasses_next(self):
        run_cpp(r'''
{Fixture f;NativeHipLive live;ready(live,f);block_run=true;
std::thread worker([&]{assert(f.l->Run()==0);});
while(!entered_run)std::this_thread::yield();key_down=true;assert(!record(live,f));
block_run=false;worker.join();assert(runs==1);
for(unsigned y=0;y<f.c->desc.Height;y++)for(unsigned x=0;x<f.c->desc.Width;x++){
auto*p=f.l->upload->bytes.data()+y*17*4+x*4;assert(p[0]==191&&p[1]==191&&p[2]==191);}
int before=key_samples;assert(!record(live,f));assert(!record(live,f));key_down=false;
assert(key_samples==before+2);}
assert(objects==0&&clients==0);
''')

    def test_hip_failure_copies_current_not_previous_frame(self):
        run_cpp(r'''
{Fixture f;NativeHipLive live;ready(live,f);assert(f.l->Run()==0);f.l->ResetAllocator();
for(auto&x:f.c->bytes)x=17;fail_hip=true;assert(record(live,f,9));assert(f.l->Run()==0);
for(unsigned y=0;y<f.c->desc.Height;y++)assert(!memcmp(f.l->upload->bytes.data()+y*17*4,f.c->bytes.data()+y*17*4,17*4));}
assert(objects==0&&clients==0);
''')

    def test_queue_conflict_duplicate_and_mapped_map_failure(self):
        run_cpp(r'''
{Fixture f;NativeHipLive live;ready(live,f);
assert(f.l->Run(111)==0);assert(f.l->Run(111)!=0);f.l->ResetAllocator();
assert(record(live,f,8));assert(f.l->Run(222)!=0);f.l->ResetAllocator();
int baseline=objects;
// A new geometry forces buffer recreation, so the setup failures fire.
f.c->desc.Width=15;f.c->desc.Height=21;
fail_shared=true;fail_map=true;assert(!record(live,f,9));
f.c->desc.Width=17;f.c->desc.Height=19;
// Failed setup discards the buffer pair and leaks nothing else.
fail_map=false;fail_shared=false;assert(objects==baseline-2);
assert(record(live,f,9));assert(f.l->Run(111)==0);}
assert(objects==0&&clients==0);
''')

    def test_failed_setup_does_not_record_or_leak(self):
        run_cpp(r'''
{Fixture f;NativeHipLive live;ready(live,f);f.l->ResetAllocator();
// Shared-handle failure falls back to mapped buffers; setup still succeeds.
f.c->desc.Width=15;f.c->desc.Height=21;
creates=0;fail_shared=true;assert(record(live,f,11));
f.c->desc.Width=17;f.c->desc.Height=19;
assert(f.l->Run(11)==0&&maps>=2);
f.l->ResetAllocator();
int baseline=objects;auto copies=f.l->copies;
// Full failure (shared + mapped) records nothing and leaks nothing.
f.c->desc.Width=13;f.c->desc.Height=23;creates=0;fail_shared=true;fail_map=true;
assert(!record(live,f,12));
f.c->desc.Width=17;f.c->desc.Height=19;fail_map=false;fail_shared=false;
assert(objects==baseline-2&&f.l->copies==copies);
fail_private=true;f.c->desc.Width=11;f.c->desc.Height=21;
assert(!record(live,f,13));
f.c->desc.Width=17;f.c->desc.Height=19;fail_private=false;assert(objects==baseline&&f.l->copies==copies);
assert(record(live,f,14));assert(f.l->Run(11)==0);}
assert(objects==0&&clients==0);
''')

    def test_capacity_tracks_allocator_refs_not_completed_callback(self):
        run_cpp(r'''
{Fixture f;NativeHipLive live;ready(live,f);auto*a=f.l;auto*b=new ID3D12GraphicsCommandList(f.d);auto*c=new ID3D12GraphicsCommandList(f.d);
f.l=b;assert(record(live,f,8));f.l=c;assert(record(live,f,9));
auto*d=new ID3D12GraphicsCommandList(f.d);f.l=d;assert(!record(live,f,10)&&d->copies==0);
assert(a->Run()==0);assert(!record(live,f,10));a->ResetAllocator();assert(record(live,f,10));
a->Release();b->Release();c->Release();}
assert(objects==0&&clients==0);
''')

    def test_unsupported_inputs_and_foreign_device_unchanged(self):
        run_cpp(r'''
{Fixture f;NativeHipLive live;auto original=f.c->desc;
for(unsigned format:{0u,27u,29u,90u}){f.c->desc.Format=format;assert(!record(live,f));}f.c->desc=original;
for(UINT64 width:{0ull,7681ull,0x100000011ull}){f.c->desc.Width=width;assert(!record(live,f));}f.c->desc=original;
f.c->desc.MipLevels=2;assert(!record(live,f));f.c->desc=original;
f.c->desc.SampleDesc.Count=4;assert(!record(live,f));f.c->desc=original;
f.c->desc.DepthOrArraySize=2;assert(!record(live,f));f.c->desc=original;
f.c->desc.Flags=32;assert(!record(live,f));f.c->desc=original;
assert(!live.Record(f.l,f.c,0xffffffff,7));
Fixture other;assert(!live.Record(f.l,other.c,8,7));assert(f.l->copies==0&&f.l->markers==0&&clients==0);}
assert(objects==0);
''')

    def test_explicit_srgb_and_bgra_fp16_unorm16_exact_alpha(self):
        run_cpp(r'''
for(unsigned format:{87u,10u,11u}){
Fixture f;f.c->Release();D3D12_RESOURCE_DESC desc{};desc.Dimension=3;desc.Width=17;desc.Height=19;
desc.DepthOrArraySize=desc.MipLevels=desc.SampleDesc.Count=1;desc.Format=format;f.c=new ID3D12Resource(f.d,desc);
for(auto&v:f.c->bytes)v=0x31;srgb=true;NativeHipLive live;ready(live,f);assert(f.l->Run()==0);
const unsigned bpp=format==87?4:8,alpha=bpp/4;
for(unsigned y=0;y<19;y++)for(unsigned x=0;x<17;x++)
assert(!memcmp(f.l->upload->bytes.data()+y*17*bpp+x*bpp+bpp-alpha,f.c->bytes.data()+(y*17+x)*bpp+bpp-alpha,alpha));}
assert(objects==0&&clients==0);
''')

    def test_reentrant_toggle_during_setup_bypasses_callback(self):
        run_cpp(r'''
{Fixture f;NativeHipLive live;ready(live,f);f.l->ResetAllocator();
static NativeHipLive* active;active=&live;
desc_hook=[]{key_down=true;active->Record(nullptr,nullptr,0,8);};
record(live,f,9);if(f.l->marker.context){assert(f.l->Run()==0);assert(runs==0);}}
assert(objects==0&&clients==0);
''')

    def test_missing_extension_anchor_survives_owner_recreation(self):
        run_cpp(r'''
accept_marker=false;
{Fixture f;{NativeHipLive live;ready(live,f);}int retained=objects;
{NativeHipLive next;ready(next,f);assert(f.l->private_refs.size()==2);assert(objects>retained);}}
assert(objects==0&&clients==0);
''')

    def test_background_init_retains_state_after_owner_destruction(self):
        run_cpp(r'''
block_init=true;
{Fixture f;{NativeHipLive live;assert(!record(live,f));while(!init_entered)std::this_thread::yield();}
assert(clients==1&&f.l->copies==0);}
assert(objects>0);block_init=false;
for(int i=0;i<2000&&objects;i++)std::this_thread::sleep_for(std::chrono::milliseconds(1));
assert(objects==0&&clients==0);
''')

    def test_4k_resize_roundtrip_preserves_alpha(self):
        run_cpp(r'''
{Fixture f;f.c->Release();D3D12_RESOURCE_DESC d{};d.Dimension=3;d.Width=3840;d.Height=2160;
d.DepthOrArraySize=d.MipLevels=d.SampleDesc.Count=1;d.Format=28;f.c=new ID3D12Resource(f.d,d);
for(size_t i=0;i<f.c->bytes.size();i++)f.c->bytes[i]=static_cast<unsigned char>(i%251);
NativeHipLive live;ready(live,f);assert(f.l->Run()==0&&runs==1);
for(size_t i=0;i<f.c->bytes.size();i+=4){assert(f.l->upload->bytes[i]==191);assert(f.l->upload->bytes[i+3]==f.c->bytes[i+3]);}}
assert(objects==0&&clients==0);
''')

    def test_missing_extension_no_suffix_and_prefix_resources_retained(self):
        run_cpp(r'''
accept_marker=false;
{Fixture f;NativeHipLive live;ready(live,f);assert(f.l->copies==2);assert(f.l->barriers==3);
assert(!f.l->private_refs.empty());assert(!record(live,f));assert(f.l->markers==1);assert(runs==0);
assert(f.l->readback->bytes[0]==f.c->bytes[0]);}
assert(objects==0&&clients==0);
''')


if __name__ == '__main__':
    unittest.main()
