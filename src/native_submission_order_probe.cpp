#include "native_text_overlay.h"
#include "native_input_geometry.h"
static bool fit_small_input();
#include <mutex>
#include <string>
#include "native_game_submission.h"
#include "native_lab_paths.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winver.h>
#include <d3d12.h>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <exception>
#include "reshade.hpp"
#include "MinHook.h"
#ifdef NATIVE_ORDER_NEURAL
#define NATIVE_ORDER_SNAPSHOT
#include "native_game_oneshot.h"
static NativeGameOneShot neural_oneshot;
#include "native_hip_live.h"
static NativeHipLive hip_live;
#include "native_fsr3_layout_gate.h"
// Which dispatch entry point owns the live path (see claim_live_route).
// Declared with hip_live because both dispatch handlers below consult it.
static std::atomic<int> live_route_owner{LiveRouteNone};
#endif
#ifdef NATIVE_ORDER_SNAPSHOT
#include "native_submitted_readback.h"
#include "native_snapshot_gate.h"
struct PendingSnapshot {ID3D12GraphicsCommandList*list{};ID3D12Resource*source{};DWORD thread{};unsigned frame{};ID3D12Resource*motion{};bool reset{};D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_UNORDERED_ACCESS};};
// Temporal contract observed on the first FFX dispatch (motion texture size, render size); zero until seen.
static std::atomic<unsigned>observed_motion_w{0},observed_motion_h{0},observed_render_w{0},observed_render_h{0};
static PendingSnapshot pending_snapshot;
static std::mutex snapshot_mutex;
static bool snapshot_taken{};
static thread_local bool snapshot_active{};
static bool cross_thread_submit{}; /* set for the XeSS path (Rise of the Ronin submits from a different thread than the one recording the upscaler) */
#endif
#ifdef NATIVE_ORDER_NEURAL
extern "C" __declspec(dllexport) const char*NAME="DLSS5 AMD single-frame verification";
extern "C" __declspec(dllexport) const char*DESCRIPTION="Background initialization then one reset-history neural frame; not a completed temporal renderer.";
#else
extern "C" __declspec(dllexport) const char*NAME="Native FFX submission order observer";
extern "C" __declspec(dllexport) const char*DESCRIPTION="Records FFX and command-list ordering; optional diagnostic snapshot build.";
#endif
struct Header {uint64_t type;Header*next;};
struct ResourcePayload {void*resource;uint32_t type,format,width,height,depth,mips,flags,usage,state,padding;};
static_assert(sizeof(ResourcePayload)==48,"FFX x64 resource layout");
static std::atomic<uint64_t>tracked_output{};
using Dispatch=uint32_t(*)(void**,const Header*);
static Dispatch original{};
static std::atomic<unsigned> frames{},events{};
// Flicker triage: how many FFX dispatches were armed for the neural path and how many actually ran it.
static std::atomic<unsigned> armed_frames{},neural_jobs{},dropped_pending{};
static SRWLOCK lock=SRWLOCK_INIT;
using ExecuteLists=void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*,UINT,ID3D12CommandList*const*);
static ExecuteLists original_execute{};
static std::atomic<bool>execute_install_attempted{};
static std::atomic<unsigned>native_batches{};
static constexpr GUID UnwrappedObject={0x7f2c9a11,0x3b4e,0x4d6a,{0x81,0x2f,0x5e,0x9c,0xd3,0x7a,0x1b,0x42}};
#ifdef NATIVE_ORDER_NEURAL
/* on-screen notice (native_text_overlay.h): the upscaler hook decides the text per frame; the notice is drawn from the ExecuteCommandLists hook
   on our own command list right after the game's batch (never recorded into the game's list: doing that crashed D3D12Core in Magpie). */
static NativeTextOverlay text_overlay;static std::mutex notice_mutex;static std::string notice_text;static ID3D12Resource*notice_target=nullptr;static unsigned notice_frame=0;
/* FSR 3.1 ffx_api resource state bits (FfxApiResourceState) -> D3D12 state; false = a combination we do not know */
static bool ffx_state_to_d3d12(uint32_t s,D3D12_RESOURCE_STATES&out){
 switch(s){
  case 1:out=D3D12_RESOURCE_STATE_COMMON;return true;case 2:out=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;return true;
  case 4:out=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;return true;case 8:out=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;return true;
  case 12:out=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;return true;
  case 16:out=D3D12_RESOURCE_STATE_COPY_SOURCE;return true;case 32:out=D3D12_RESOURCE_STATE_COPY_DEST;return true;
  case 20:out=D3D12_RESOURCE_STATE_GENERIC_READ;return true;case 128:out=D3D12_RESOURCE_STATE_PRESENT;return true;case 256:out=D3D12_RESOURCE_STATE_RENDER_TARGET;return true;
  default:return false;
 }
}
// Real DLL file version (e.g. games ship different FidelityFX SDK builds over
// time with different ABI-compatible-looking but differently-laid-out dispatch
// structs) - the same version stamp danielblnc's own runtime already reads and
// logs for these exact FFX DLLs. Static 4KB buffer: version resources are small.
static bool module_file_version(HMODULE m,wchar_t*out,size_t out_len){
 wchar_t path[MAX_PATH]{};if(!GetModuleFileNameW(m,path,MAX_PATH))return false;
 DWORD handle=0;DWORD size=GetFileVersionInfoSizeW(path,&handle);if(!size||size>4096)return false;
 unsigned char buf[4096];if(!GetFileVersionInfoW(path,handle,sizeof buf,buf))return false;
 VS_FIXEDFILEINFO*info=nullptr;UINT info_len=0;
 if(!VerQueryValueW(buf,L"\\",reinterpret_cast<void**>(&info),&info_len)||!info)return false;
 swprintf(out,out_len,L"%u.%u.%u.%u",HIWORD(info->dwFileVersionMS),LOWORD(info->dwFileVersionMS),
  HIWORD(info->dwFileVersionLS),LOWORD(info->dwFileVersionLS));
 return true;
}
static D3D12_RESOURCE_STATES notice_state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
static void draw_pending_notice(ID3D12CommandQueue*q){
 std::string text;ID3D12Resource*target=nullptr;
 {std::lock_guard<std::mutex>g(notice_mutex);if(notice_frame&&notice_frame==frames.load()&&notice_target){text=notice_text;target=notice_target;target->AddRef();}notice_frame=0;}
 if(!target)return;
 static NativeGameSubmission*submit=nullptr;static ID3D12CommandQueue*submit_queue=nullptr;
 try{
  if(submit_queue!=q){delete submit;submit=nullptr;submit_queue=nullptr;submit=new NativeGameSubmission;submit->Create(q,false);submit_queue=q;}
  const D3D12_RESOURCE_STATES st=notice_state;submit->Submit([&](ID3D12GraphicsCommandList*c){text_overlay.Draw(c,target,text.c_str(),24,96,3,st);});
 }catch(const std::exception&e){static std::atomic<bool>logged{false};if(!logged.exchange(true))if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu notice_submit_failed error=%s\n",GetCurrentProcessId(),e.what());fclose(f);}}
 target->Release();
}
#endif
using Barriers=void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*,UINT,const D3D12_RESOURCE_BARRIER*);
static Barriers original_barriers{};
static std::atomic<bool>barrier_install_attempted{};
static void STDMETHODCALLTYPE native_barriers(ID3D12GraphicsCommandList*c,UINT count,const D3D12_RESOURCE_BARRIER*b){
 original_barriers(c,count,b);
 if(!b)return;auto target=tracked_output.load();if(!target)return;
  for(UINT i=0;i<count;i++){
   const auto&v=b[i];
#ifdef NATIVE_ORDER_SNAPSHOT
  if(fit_small_input()&&!snapshot_active&&v.Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION&&v.Flags!=D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY){
   std::lock_guard<std::mutex>g(snapshot_mutex);
   if(pending_snapshot.list==c&&pending_snapshot.source==v.Transition.pResource&&v.Transition.Subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
    pending_snapshot.state=v.Transition.StateAfter;
  }
#endif
#ifdef NATIVE_ORDER_NEURAL
  if(fit_small_input()&&!snapshot_active&&v.Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION&&v.Flags!=D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY){
   std::lock_guard<std::mutex>g(notice_mutex);
   if(notice_frame&&notice_target==v.Transition.pResource&&v.Transition.Subresource==D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)notice_state=v.Transition.StateAfter;
  }
#endif
   bool relevant=v.Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION?reinterpret_cast<uint64_t>(v.Transition.pResource)==target:
   v.Type==D3D12_RESOURCE_BARRIER_TYPE_UAV?(!v.UAV.pResource||reinterpret_cast<uint64_t>(v.UAV.pResource)==target):
   v.Type==D3D12_RESOURCE_BARRIER_TYPE_ALIASING?(reinterpret_cast<uint64_t>(v.Aliasing.pResourceBefore)==target||reinterpret_cast<uint64_t>(v.Aliasing.pResourceAfter)==target):false;
  if(!relevant||events.fetch_add(1)>=8192)continue;
  AcquireSRWLockExclusive(&lock);
  if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
   if(v.Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
    fprintf(f,"pid=%lu thread=%lu tick=%llu kind=native_output_barrier list=%p resource=%llx before=%u after=%u subresource=%u flags=%u\n",GetCurrentProcessId(),GetCurrentThreadId(),GetTickCount64(),c,(unsigned long long)target,unsigned(v.Transition.StateBefore),unsigned(v.Transition.StateAfter),v.Transition.Subresource,unsigned(v.Flags));
   else if(v.Type==D3D12_RESOURCE_BARRIER_TYPE_UAV)
    fprintf(f,"pid=%lu thread=%lu tick=%llu kind=native_output_uav list=%p resource=%p global=%u no_state_transition=1\n",GetCurrentProcessId(),GetCurrentThreadId(),GetTickCount64(),c,v.UAV.pResource,v.UAV.pResource?0u:1u);
   else fprintf(f,"pid=%lu thread=%lu tick=%llu kind=native_output_alias list=%p before_resource=%p after_resource=%p\n",GetCurrentProcessId(),GetCurrentThreadId(),GetTickCount64(),c,v.Aliasing.pResourceBefore,v.Aliasing.pResourceAfter);
   fclose(f);
  }ReleaseSRWLockExclusive(&lock);
 }
}
static void install_native_barriers(void*list){
 if(!list||barrier_install_attempted.exchange(true))return;
 ID3D12GraphicsCommandList*native=nullptr;
 HRESULT hr=static_cast<IUnknown*>(list)->QueryInterface(UnwrappedObject,reinterpret_cast<void**>(&native));
 if(FAILED(hr)||!native)return;
 void**table=nullptr;void*target=nullptr;SIZE_T got=0;
 bool readable=ReadProcessMemory(GetCurrentProcess(),native,&table,sizeof(table),&got)&&got==sizeof(table)&&table&&ReadProcessMemory(GetCurrentProcess(),table+26,&target,sizeof(target),&got)&&got==sizeof(target)&&target;
 MH_STATUS s=MH_ERROR_NOT_EXECUTABLE;
 if(readable){s=MH_CreateHook(target,reinterpret_cast<void*>(&native_barriers),reinterpret_cast<void**>(&original_barriers));if(s==MH_OK)s=MH_EnableHook(target);}
 if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu kind=native_barrier_hook status=%u native_list=%p\n",GetCurrentProcessId(),unsigned(s),native);fclose(f);}native->Release();
}
static void device_identity(const char*origin,ID3D12Device*d){
 if(!d)return;
 IUnknown*identity=nullptr,*unwrapped=nullptr,*native_identity=nullptr;
 HRESULT identity_hr=d->QueryInterface(IID_PPV_ARGS(&identity));
 HRESULT unwrap_hr=d->QueryInterface(UnwrappedObject,reinterpret_cast<void**>(&unwrapped));
 HRESULT native_hr=unwrapped?unwrapped->QueryInterface(IID_PPV_ARGS(&native_identity)):E_NOINTERFACE;
 AcquireSRWLockExclusive(&lock);
 if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
  fprintf(f,"pid=%lu kind=device_identity origin=%s device=%p identity=%p identity_hr=%08x unwrapped=%p unwrap_hr=%08x native_identity=%p native_hr=%08x\n",GetCurrentProcessId(),origin,d,identity,unsigned(identity_hr),unwrapped,unsigned(unwrap_hr),native_identity,unsigned(native_hr));fclose(f);
 }ReleaseSRWLockExclusive(&lock);
 if(native_identity)native_identity->Release();if(unwrapped)unwrapped->Release();if(identity)identity->Release();
}
static void log(const char*kind,void*list,void*queue,unsigned value=0){
 if(!frames.load()||events.fetch_add(1)>=8192)return;
 AcquireSRWLockExclusive(&lock);
 if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
  fprintf(f,"pid=%lu thread=%lu tick=%llu kind=%s list=%p queue=%p value=%u observation_only=1\n",GetCurrentProcessId(),GetCurrentThreadId(),GetTickCount64(),kind,list,queue,value);fclose(f);
 }ReleaseSRWLockExclusive(&lock);
}
/* Read before initialization: the usual environment flags are applied by the background loader. */
static bool fit_small_input(){static const bool enabled=[]{unsigned v=0;if(FILE*f=_wfopen(NativeLabPath(L"native-game-flags.txt").c_str(),L"rb")){char line[256];while(fgets(line,sizeof line,f))sscanf(line,"DLSS5_FIT_INPUT=%u",&v);fclose(f);}return v==1;}();return enabled;}
static bool supported_input(unsigned w,unsigned h){return (w==1920&&h==1080)||(fit_small_input()&&NativeInputGeometry::Supported(w,h));}
static std::atomic<void**>fit_context{nullptr};
using DestroyContext=uint32_t(*)(void**,const void*);
static DestroyContext original_destroy{};
static uint32_t destroy_context(void**context,const void*allocation_callbacks){
 auto result=original_destroy(context,allocation_callbacks);
 if(result==0){auto expected=context;fit_context.compare_exchange_strong(expected,nullptr);}
 return result;
}
static bool neural_output_ok(void*resource,unsigned declared_w,unsigned declared_h,unsigned state){
 D3D12_RESOURCE_STATES mapped{};
 if(!resource||!ffx_state_to_d3d12(state,mapped)||declared_w<16||declared_h<16||declared_w>7680||declared_h>4320)return false;
 auto d=static_cast<ID3D12Resource*>(resource)->GetDesc();
 const unsigned w=unsigned(d.Width),h=d.Height;
 return d.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE2D&&w>=16&&h>=16&&w<=7680&&h<=4320
  &&d.MipLevels==1&&d.DepthOrArraySize==1&&d.SampleDesc.Count==1&&NativeIsGameColor(d.Format);
}
static bool neural_desc_ok(const D3D12_RESOURCE_DESC&d){
 const unsigned w=unsigned(d.Width),h=d.Height;
 return d.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE2D&&w>=16&&h>=16&&w<=7680&&h<=4320
  &&d.MipLevels==1&&d.DepthOrArraySize==1&&d.SampleDesc.Count==1&&NativeIsGameColor(d.Format);
}
static uint32_t dispatch(void**context,const Header*h){
 if(!h||(h->type&0x00ffffffu)!=0x00010001u)return original(context,h);
 /* Select the first upscaler context until it is destroyed (including its size-error notice). A following FSR4 (even when its output
    also fits 1080p) must not replace the input stage's motion, pending work or status text. */
 if(fit_small_input()){
  auto chosen=fit_context.load();if(chosen&&chosen!=context)return original(context,h);
  ResourcePayload candidate{};SIZE_T bytes=0;
  if(context&&ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<const char*>(h)+312,&candidate,sizeof candidate,&bytes)&&bytes==sizeof candidate&&candidate.resource){
   void**expected=nullptr;fit_context.compare_exchange_strong(expected,context);
   if(fit_context.load()!=context)return original(context,h);
  }
 }
 // Existing FFX x64 ABI: commandList follows the16-byte header. Payload only.
 void*list=nullptr;SIZE_T got=0;
 ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<const char*>(h)+16,&list,sizeof(list),&got);
 unsigned n=++frames;log("ffx_begin",got==sizeof(list)?list:nullptr,nullptr,n);
 ResourcePayload output{};ID3D12Resource*frame_motion=nullptr;bool frame_reset=false;
 tracked_output.store(0); // Never attribute later barriers to an unreadable frame.
 SIZE_T output_bytes=0;
 if(ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<const char*>(h)+312,&output,sizeof(output),&output_bytes)&&output_bytes==sizeof(output)){
  tracked_output.store(reinterpret_cast<uint64_t>(output.resource));
  if(n<=8&&output.resource){
   auto*r=static_cast<ID3D12Resource*>(output.resource);auto desc=r->GetDesc();ID3D12Device*device=nullptr;auto hr=r->GetDevice(IID_PPV_ARGS(&device));
   device_identity("output",device);
   AcquireSRWLockExclusive(&lock);
   if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
    fprintf(f,"pid=%lu kind=native_output_desc frame=%u resource=%p dimension=%u dxgi=%u size=%llux%u samples=%u mips=%u flags=%u device=%p device_hr=%08x\n",GetCurrentProcessId(),n,r,unsigned(desc.Dimension),unsigned(desc.Format),(unsigned long long)desc.Width,desc.Height,desc.SampleDesc.Count,desc.MipLevels,unsigned(desc.Flags),device,unsigned(hr));fclose(f);
   }ReleaseSRWLockExclusive(&lock);if(device)device->Release();
  }
  if(n<=8){AcquireSRWLockExclusive(&lock);
   if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
    fprintf(f,"pid=%lu thread=%lu tick=%llu kind=ffx_output frame=%u list=%p resource=%p format=%u size=%ux%u declared_state=%u payload_only=1\n",GetCurrentProcessId(),GetCurrentThreadId(),GetTickCount64(),n,list,output.resource,output.format,output.width,output.height,output.state);fclose(f);
   }ReleaseSRWLockExclusive(&lock);
  }
  // Temporal contract: motion vectors payload (+120), scale/sizes/reset (+360..) per the FFX upscale layout.
  ResourcePayload motion{};float mvscale[4]{};unsigned sizes[4]{};unsigned char reset=0;SIZE_T a=0,b=0,c=0,d=0;
  ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<const char*>(h)+120,&motion,sizeof(motion),&a);
  ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<const char*>(h)+360,mvscale,16,&b);
  ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<const char*>(h)+376,sizes,16,&c);
  ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<const char*>(h)+408,&reset,1,&d);
  if(a==sizeof(motion)&&c==16&&motion.resource){
   const bool size_changed=!observed_motion_w.load()||motion.width!=observed_motion_w.load()||motion.height!=observed_motion_h.load()||sizes[0]!=observed_render_w.load()||sizes[1]!=observed_render_h.load();
   if(size_changed){observed_motion_w=motion.width;observed_motion_h=motion.height;observed_render_w=sizes[0];observed_render_h=sizes[1];
    AcquireSRWLockExclusive(&lock);
    if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu kind=ffx_size_change frame=%u motion=%ux%u render=%ux%u output=%ux%u\n",GetCurrentProcessId(),n,motion.width,motion.height,sizes[0],sizes[1],output.width,output.height);fclose(f);}
    ReleaseSRWLockExclusive(&lock);}
   if(b==16){NativeMotionVectorScale()[0]=mvscale[0];NativeMotionVectorScale()[1]=mvscale[1];}
  }
  frame_motion=(a==sizeof(motion)&&motion.width==observed_motion_w.load()&&motion.height==observed_motion_h.load()&&sizes[0]==observed_render_w.load()&&sizes[1]==observed_render_h.load())?static_cast<ID3D12Resource*>(motion.resource):nullptr;
  frame_reset=reset!=0;
  if(n<=8){
   AcquireSRWLockExclusive(&lock);
   if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
    fprintf(f,"pid=%lu kind=ffx_temporal frame=%u motion=%p format=%u size=%ux%u state=%u jitter=%g,%g mvscale=%g,%g render=%ux%u upscale=%ux%u reset=%u\n",GetCurrentProcessId(),n,motion.resource,motion.format,motion.width,motion.height,motion.state,mvscale[0],mvscale[1],mvscale[2],mvscale[3],sizes[0],sizes[1],sizes[2],sizes[3],unsigned(reset));fclose(f);
   }ReleaseSRWLockExclusive(&lock);
  }
 }
 if(output.resource)install_native_barriers(list);
 auto result=original(context,h);log("ffx_end",list,nullptr,result);
 // HIP_LIVE_ROUTE_BEGIN
 if(NativeHipRequested()){
  D3D12_RESOURCE_STATES live_state{};
  if(result==0&&list&&output.resource&&ffx_state_to_d3d12(output.state,live_state)&&
     claim_live_route(live_route_owner,LiveRouteUnified))
   hip_live.Record(static_cast<ID3D12GraphicsCommandList*>(list),static_cast<ID3D12Resource*>(output.resource),live_state,n);
  return result; // Never enter the old post-ECL readback/wait path for HIP.
 }
 // HIP_LIVE_ROUTE_END
#ifdef NATIVE_ORDER_NEURAL
 /* on-screen notice (native_text_overlay.h): why the picture is not changing -- the input is not 1920x1080 (the network only knows that
    size; the hook never arms), the network is still initializing, or its initialization failed (developer mode off, wrong driver...) */
 static const unsigned notice_mode=[]{unsigned v=2;if(FILE*f=_wfopen(NativeLabPath(L"native-game-flags.txt").c_str(),L"rb")){char line[256];while(fgets(line,sizeof line,f)){unsigned x;if(sscanf(line,"DLSS5_NOTICE=%u",&x)==1)v=x;}fclose(f);}return v;}();
 D3D12_RESOURCE_STATES declared_state{};const bool state_known=output_bytes==sizeof(output)&&ffx_state_to_d3d12(output.state,declared_state);
 if(output_bytes==sizeof(output)&&!state_known){static std::atomic<bool>logged{false};if(!logged.exchange(true))if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-game-oneshot.txt").c_str(),L"ab")){fprintf(f,"pid=%lu tick=%llu event=output_state_unknown detail=the upscaler declares its output in ffx state %u, which the hook cannot map to a D3D12 state; please report this line\n",GetCurrentProcessId(),GetTickCount64(),output.state);fclose(f);}}
 if(notice_mode&&output_bytes==sizeof(output)&&output.resource&&state_known&&list){
  static std::atomic<bool>size_logged{false};char notice[80]{};
  if(!supported_input(output.width,output.height)){snprintf(notice,sizeof notice,fit_small_input()?"DLSS5-AMD: INPUT MAX 1920X1080 (NOW %uX%u)":"DLSS5-AMD: INPUT MUST BE 1920X1080 (NOW %uX%u)",output.width,output.height);
   if(!size_logged.exchange(true))if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-game-oneshot.txt").c_str(),L"ab")){fprintf(f,"pid=%lu tick=%llu event=input_size_unsupported detail=upscaler runs at %ux%u, expected first-stage output within 1920x1080 with DLSS5_FIT_INPUT=1, otherwise exactly 1920x1080 (Magpie: FSR3 scale relative to input = 1)\n",GetCurrentProcessId(),GetTickCount64(),output.width,output.height);fclose(f);}}
  else{const unsigned ph=neural_oneshot.Phase();if(ph==0)text_overlay.Prepare(static_cast<ID3D12Resource*>(output.resource)); /* pipeline built before the initializer thread starts (same frame arms it) */
   if(ph==1&&text_overlay.Ready())snprintf(notice,sizeof notice,"DLSS5-AMD: INITIALIZING...");else if(ph==5)snprintf(notice,sizeof notice,"DLSS5-AMD: INIT FAILED - SEE DLSS5-AMD\\LOGS");
  }
  if(notice[0]&&notice_mode>=2){std::lock_guard<std::mutex>g(notice_mutex);notice_text=notice;notice_state=declared_state;auto*r=static_cast<ID3D12Resource*>(output.resource);if(notice_target!=r){if(notice_target)notice_target->Release();notice_target=r;notice_target->AddRef();}notice_frame=n;}
 }
#endif
#ifdef NATIVE_ORDER_SNAPSHOT
 /* DLSS5_SNAPSHOT_FRAME=<n> in native-game-flags.txt (read here directly: the flag file is applied to the environment only when the network
    initializes, which is what this frame triggers): the upscaler frame that arms the network. Default 120 (the game build); 1 for Magpie. */
 static const unsigned snapshot_frame=[]{unsigned v=120;if(FILE*f=_wfopen(NativeLabPath(L"native-game-flags.txt").c_str(),L"rb")){char line[256];while(fgets(line,sizeof line,f)){unsigned x;if(sscanf(line,"DLSS5_SNAPSHOT_FRAME=%u",&x)==1&&x>=1)v=x;}fclose(f);}return v;}();
 bool request=n==snapshot_frame;
#ifdef NATIVE_ORDER_NEURAL
 /* idle (first time, or after a session reset) and failed states re-arm from the snapshot frame on; the first arming is still exactly snapshot_frame */
 request=neural_oneshot.WantsFrame()||(n>=snapshot_frame&&(neural_oneshot.Phase()==0||neural_oneshot.Phase()==5));
#endif
 /* any declared output state we can map is taken over (the frame transitions from / back to it); 2 = UAV is what Stellar Blade and Magpie declare */
  if(request&&result==0&&output_bytes==sizeof(output)&&output.resource&&supported_input(output.width,output.height)&&neural_output_ok(output.resource,output.width,output.height,output.state)&&state_known&&list){
  ID3D12GraphicsCommandList*native=nullptr;
  if(SUCCEEDED(static_cast<IUnknown*>(list)->QueryInterface(UnwrappedObject,reinterpret_cast<void**>(&native)))&&native){
   std::lock_guard<std::mutex>guard(snapshot_mutex);
   bool eligible=!snapshot_taken;
#ifdef NATIVE_ORDER_NEURAL
   eligible=eligible||neural_oneshot.WantsFrame()||neural_oneshot.Phase()==0||neural_oneshot.Phase()==5;
#endif
   if(eligible&&!pending_snapshot.list){auto*r=static_cast<ID3D12Resource*>(output.resource);r->AddRef();if(frame_motion)frame_motion->AddRef();pending_snapshot={native,r,GetCurrentThreadId(),n,frame_motion,frame_reset,declared_state};++armed_frames;
    AcquireSRWLockExclusive(&lock);
    if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){auto d=r->GetDesc();fprintf(f,"pid=%lu kind=neural_arm frame=%u size=%llux%u format=%u\n",GetCurrentProcessId(),n,(unsigned long long)d.Width,d.Height,unsigned(d.Format));fclose(f);}
    ReleaseSRWLockExclusive(&lock);}
   else native->Release();
  }
 }
 if(n%100==0){AcquireSRWLockExclusive(&lock);
  if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu kind=neural_coverage ffx_frames=%u armed=%u ran=%u dropped_pending=%u\n",GetCurrentProcessId(),n,armed_frames.load(),neural_jobs.load(),dropped_pending.load());fclose(f);}
  ReleaseSRWLockExclusive(&lock);}
#endif
 return result;
}
#ifdef NATIVE_ORDER_NEURAL
// Older, non-unified FSR3 upscaler dispatch (ffx_fsr3upscaler_x64.dll!
// ffxFsr3UpscalerContextDispatch) - some titles (Cyberpunk 2077) call this
// directly instead of the unified ffxDispatch hooked above. Field order from
// AMD's public FidelityFX-SDK (FfxFsr3UpscalerDispatchDescription); confirmed
// live against a real capture (docs/linux-support-spec.md) that this game's
// build carries an embedded per-resource debug-name buffer AMD's current SDK
// header no longer has, making each resource entry 176 bytes (48 + a
// FFX_RESOURCE_NAME_SIZE=64 wchar_t name), not 48. Confirmed via `output`
// landing at exactly commandList+9*176 with real 2560x1440 dimensions.
struct Fsr3ResourceEntry : ResourcePayload { wchar_t name[64]; };
static_assert(sizeof(Fsr3ResourceEntry)==176,"FSR3 named-resource layout (this game's SDK build)");
struct Fsr3UpscalerDispatchDescription {
 void*commandList;
 Fsr3ResourceEntry color,depth,motionVectors,exposure,reactive,transparencyAndComposition,
  dilatedDepth,dilatedMotionVectors,reconstructedPrevNearestDepth,output;
 float jitterOffsetX,jitterOffsetY,motionVectorScaleX,motionVectorScaleY;
 uint32_t renderWidth,renderHeight,upscaleWidth,upscaleHeight;
 bool enableSharpening;float sharpness,frameTimeDelta,preExposure;bool reset;
 float cameraNear,cameraFar,cameraFovAngleVertical,viewSpaceToMetersFactor;
 uint32_t flags;
};
// No fixed-size assert on the outer struct here - the real total (tail scalar
// fields past `output`) hasn't been independently confirmed yet, only the
// resource-entry stride and field order above. sizeof(*d) is still logged
// per-dispatch below so a wrong guess here would be caught immediately.
// This offset IS independently confirmed (a real memory capture showed a
// valid 2560x1440 resource landing exactly here) - assert it directly.
static_assert(offsetof(Fsr3UpscalerDispatchDescription,output)==1592,"FSR3 output field position (this game's SDK build)");
using Fsr3UpscalerDispatch=uint32_t(*)(void*,const Fsr3UpscalerDispatchDescription*);
static Fsr3UpscalerDispatch original_fsr3upscaler{};
// ffx_fsr3upscaler_x64.dll itself ships with no embedded version resource in
// this game (module_file_version reads "unknown" for it live) - the 176-byte
// named-resource layout above was instead confirmed against the FFX loader
// dll that ships alongside it (amd_fidelityfx_dx12.dll), whose version IS
// readable. Only trust the hardcoded layout - and only then feed captures
// into HIP - when that companion dll matches the exact build this was
// confirmed against; a different game or a different FFX SDK drop is not
// guaranteed to share this same struct shape. fsr3_layout_matches() lives in
// its own header so it stays testable on the host (see
// linux/tests/test_fsr3_layout_gate.py) without pulling in D3D12/Windows.h.
static std::atomic<bool>fsr3_layout_trusted{false};
static uint32_t fsr3_upscaler_dispatch(void*ctx,const Fsr3UpscalerDispatchDescription*d){
 if(!d)return original_fsr3upscaler(ctx,d);
 unsigned n=++frames;log("fsr3upscaler_begin",d->commandList,nullptr,n);
 if(n<=8){
  AcquireSRWLockExclusive(&lock);
  if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
   fprintf(f,"pid=%lu kind=fsr3upscaler_dispatch frame=%u list=%p color=%p output=%p out_size=%ux%u out_state=%u motion=%p render=%ux%u upscale=%ux%u reset=%u\n",
    GetCurrentProcessId(),n,d->commandList,d->color.resource,d->output.resource,d->output.width,d->output.height,
    d->output.state,d->motionVectors.resource,d->renderWidth,d->renderHeight,d->upscaleWidth,d->upscaleHeight,unsigned(d->reset));
   fclose(f);
  }ReleaseSRWLockExclusive(&lock);
 }
 auto result=original_fsr3upscaler(ctx,d);log("fsr3upscaler_end",d->commandList,nullptr,result);
 if(NativeHipRequested()){
  D3D12_RESOURCE_STATES live_state{};
  const bool state_known=ffx_state_to_d3d12(d->output.state,live_state);
  if(fsr3_route_to_live(fsr3_layout_trusted.load(),result==0,d->commandList,d->output.resource,state_known)&&
     claim_live_route(live_route_owner,LiveRouteFsr3))
   hip_live.Record(static_cast<ID3D12GraphicsCommandList*>(d->commandList),static_cast<ID3D12Resource*>(d->output.resource),live_state,n);
 }
 return result;
}
// XeSS titles (Rise of the Ronin): the same contract read from xessD3D12Execute's parameters (XeSS 1.x/2.x SDK layout)
// instead of the FFX dispatch description. XeSS runs first; the network then refines its 1080p output after submission,
// exactly as with FSR. Velocity is XeSS-scaled by (-w,-h) in this title, so the feed's motion sign is flipped.
struct XessExecuteParams{ID3D12Resource*color,*velocity,*depth,*exposure,*responsive,*output;float jitter_x,jitter_y,exposure_scale;uint32_t reset,input_w,input_h;int32_t bases[10];ID3D12DescriptorHeap*heap;uint32_t heap_offset;};
using XessExecute=int(*)(void*,ID3D12GraphicsCommandList*,const XessExecuteParams*);
static XessExecute original_xess{};
static int xess_execute(void*ctx,ID3D12GraphicsCommandList*list,const XessExecuteParams*p){
 if(!p||!p->output)return original_xess(ctx,list,p);
 unsigned n=++frames;log("xess_begin",list,nullptr,n);
 auto*out=p->output;auto od=out->GetDesc();tracked_output.store(reinterpret_cast<uint64_t>(out));
 D3D12_RESOURCE_DESC md{};if(p->velocity)md=p->velocity->GetDesc();
 if(p->velocity&&(!observed_motion_w.load()||unsigned(md.Width)!=observed_motion_w.load()||md.Height!=observed_motion_h.load()||p->input_w!=observed_render_w.load()||p->input_h!=observed_render_h.load())){observed_motion_w=unsigned(md.Width);observed_motion_h=md.Height;observed_render_w=p->input_w;observed_render_h=p->input_h;}
 ID3D12Resource*frame_motion=(p->velocity&&unsigned(md.Width)==observed_motion_w.load()&&md.Height==observed_motion_h.load()&&p->input_w==observed_render_w.load()&&p->input_h==observed_render_h.load())?p->velocity:nullptr;
 const bool frame_reset=p->reset!=0;
 if(n<=8){AcquireSRWLockExclusive(&lock);
  if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
   fprintf(f,"pid=%lu kind=xess_execute frame=%u list=%p output=%p format=%u size=%llux%u flags=%u velocity=%p vformat=%u vsize=%llux%u input=%ux%u jitter=%g,%g exposure=%g reset=%u\n",GetCurrentProcessId(),n,list,out,unsigned(od.Format),(unsigned long long)od.Width,od.Height,unsigned(od.Flags),p->velocity,unsigned(md.Format),(unsigned long long)md.Width,md.Height,p->input_w,p->input_h,p->jitter_x,p->jitter_y,p->exposure_scale,p->reset);fclose(f);
  }ReleaseSRWLockExclusive(&lock);}
 install_native_barriers(list);
 int result=original_xess(ctx,list,p);log("xess_end",list,nullptr,unsigned(result));
 if(result==0&&neural_desc_ok(od)&&list&&(n==120||neural_oneshot.WantsFrame()||neural_oneshot.Phase()==0||neural_oneshot.Phase()==5)){ /* first eligible frame starts background initialization, as on the FFX path */
  ID3D12GraphicsCommandList*native=nullptr;
  if(SUCCEEDED(static_cast<IUnknown*>(list)->QueryInterface(UnwrappedObject,reinterpret_cast<void**>(&native)))&&native){
   std::lock_guard<std::mutex>guard(snapshot_mutex);
   if(!pending_snapshot.list){out->AddRef();if(frame_motion)frame_motion->AddRef();pending_snapshot={native,out,GetCurrentThreadId(),n,frame_motion,frame_reset};++armed_frames;}
   else native->Release();
  }
 }
 if(n%100==0){AcquireSRWLockExclusive(&lock);
  if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu kind=neural_coverage xess_frames=%u armed=%u ran=%u dropped_pending=%u\n",GetCurrentProcessId(),n,armed_frames.load(),neural_jobs.load(),dropped_pending.load());fclose(f);}
  ReleaseSRWLockExclusive(&lock);}
 return result;
}
#endif
static void STDMETHODCALLTYPE execute_native(ID3D12CommandQueue*q,UINT count,ID3D12CommandList*const*lists){
 const unsigned batch=++native_batches;
 if(q&&batch<=8){
  auto desc=q->GetDesc();ID3D12Device*device=nullptr;auto hr=q->GetDevice(IID_PPV_ARGS(&device));
  device_identity("queue",device);
  AcquireSRWLockExclusive(&lock);
  if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
   fprintf(f,"pid=%lu kind=native_queue_desc batch=%u queue=%p type=%u flags=%u device=%p device_hr=%08x\n",GetCurrentProcessId(),batch,q,unsigned(desc.Type),unsigned(desc.Flags),device,unsigned(hr));fclose(f);
  }ReleaseSRWLockExclusive(&lock);if(device)device->Release();
 }
 log("execute_native_begin",nullptr,q,count);
 if(lists&&count<=64)for(UINT i=0;i<count;i++)log("execute_native_item",lists[i],q,i);
 original_execute(q,count,lists);
 // This proves CPU submission returned, NOT GPU completion. No fence is added.
 log("execute_native_return",nullptr,q,count);
#ifdef NATIVE_ORDER_NEURAL
 draw_pending_notice(q);
#endif
#ifdef NATIVE_ORDER_SNAPSHOT
 if(!snapshot_active){
  PendingSnapshot job{};
  {std::lock_guard<std::mutex>guard(snapshot_mutex);
   if(pending_snapshot.list&&(cross_thread_submit?frames.load()-pending_snapshot.frame>2:pending_snapshot.frame!=frames.load())){ /* cross-thread: the render thread may be one or two upscaler calls ahead of the submit */
    pending_snapshot.list->Release();pending_snapshot.source->Release();if(pending_snapshot.motion)pending_snapshot.motion->Release();pending_snapshot={};++dropped_pending;
   }
   uintptr_t items[64]{};if(lists&&count<=64)for(UINT i=0;i<count;i++)items[i]=reinterpret_cast<uintptr_t>(lists[i]);
   bool eligible=!snapshot_taken;
#ifdef NATIVE_ORDER_NEURAL
   eligible=eligible||neural_oneshot.WantsFrame()||neural_oneshot.Phase()==0||neural_oneshot.Phase()==5; /* idle/failed: a new snapshot (re)initializes */
#endif
   if(eligible&&NativeSnapshotBatchMatch(cross_thread_submit?GetCurrentThreadId():pending_snapshot.thread,GetCurrentThreadId(),reinterpret_cast<uintptr_t>(pending_snapshot.list),lists?items:nullptr,count)){ /* XeSS titles record on the render thread and submit from another: match by list identity only */
    job=pending_snapshot;pending_snapshot={};snapshot_taken=true;
   }
  }
  if(job.list){
   snapshot_active=true;
#ifdef NATIVE_ORDER_NEURAL
   ++neural_jobs;
   try{neural_oneshot.OnSubmitted(q,job.source,job.motion,job.reset,observed_motion_w.load(),observed_motion_h.load(),observed_render_w.load(),observed_render_h.load(),job.state);}
   catch(const std::exception&e){if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-game-oneshot.txt").c_str(),L"ab")){fprintf(f,"pid=%lu tick=%llu event=execute_catch detail=%s\n",GetCurrentProcessId(),GetTickCount64(),e.what());fclose(f);}}
   if(job.motion)job.motion->Release();
#else
   try{
    auto pixels=NativeReadSubmittedFrame(q,job.source,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    wchar_t path[MAX_PATH];swprintf(path,MAX_PATH,NativeLabPath(L"logs\\ffx-submitted-%lu.f16").c_str(),GetCurrentProcessId());
    FILE*f=_wfopen(path,L"wb");if(!f)throw std::runtime_error("snapshot file open");auto written=fwrite(pixels.data(),1,pixels.size(),f);fclose(f);if(written!=pixels.size())throw std::runtime_error("snapshot short write");
    if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-snapshot-result.txt").c_str(),L"ab")){fprintf(f,"pid=%lu frame=120 bytes=%zu queue=%p source=%p state_restored=UAV success=1\n",GetCurrentProcessId(),pixels.size(),q,job.source);fclose(f);}
   }catch(const std::exception&e){if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-snapshot-result.txt").c_str(),L"ab")){fprintf(f,"pid=%lu success=0 error=%s\n",GetCurrentProcessId(),e.what());fclose(f);}}
#endif
   job.list->Release();job.source->Release();snapshot_active=false;
  }
 }
#endif
}
static void install_execute(reshade::api::command_queue*q){
 if(!frames.load()||execute_install_attempted.exchange(true))return;
 auto*native=reinterpret_cast<ID3D12CommandQueue*>(q->get_native());
 void**table=nullptr;void*target=nullptr;SIZE_T got=0;
 if(!ReadProcessMemory(GetCurrentProcess(),native,&table,sizeof(table),&got)||got!=sizeof(table)||!table||
    !ReadProcessMemory(GetCurrentProcess(),table+10,&target,sizeof(target),&got)||got!=sizeof(target)||!target){log("execute_hook_unreadable",native,nullptr);return;}
 // ID3D12CommandQueue's SDK vtable slot10 is ExecuteCommandLists.
 auto s=MH_CreateHook(target,reinterpret_cast<void*>(&execute_native),reinterpret_cast<void**>(&original_execute));
 if(s==MH_OK)s=MH_EnableHook(target);
 log("execute_hook_status",target,native,unsigned(s));
}
static void close_list(reshade::api::command_list*c){log("close_api",c,nullptr);log("close_native",reinterpret_cast<void*>(c->get_native()),nullptr);}
static void execute(reshade::api::command_queue*q,reshade::api::command_list*c){if(NativeHipRequested())return;install_execute(q);log("before_execute_api",c,q);log("before_execute_native",reinterpret_cast<void*>(c->get_native()),reinterpret_cast<void*>(q->get_native()));}
static bool compute(reshade::api::command_list*c,uint32_t,uint32_t,uint32_t){log("dispatch_api",c,nullptr);return false;}
static bool draw(reshade::api::command_list*c,uint32_t,uint32_t,uint32_t,uint32_t){log("draw_api",c,nullptr);return false;}
static void barrier(reshade::api::command_list*c,uint32_t count,const reshade::api::resource*r,const reshade::api::resource_usage*before,const reshade::api::resource_usage*after){
 if(!r||!before||!after)return;auto target=tracked_output.load();if(!target)return;
 for(uint32_t i=0;i<count;i++)if(r[i].handle==target&&events.fetch_add(1)<8192){
  AcquireSRWLockExclusive(&lock);
  if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
   fprintf(f,"pid=%lu thread=%lu tick=%llu kind=output_barrier list=%p resource=%llx before=%u after=%u observation_only=1\n",GetCurrentProcessId(),GetCurrentThreadId(),GetTickCount64(),c,(unsigned long long)target,unsigned(before[i]),unsigned(after[i]));fclose(f);
  }ReleaseSRWLockExclusive(&lock);
 }
}
static DWORD WINAPI worker(void*){
  // Whichever upscaler dll the title loads first: the FFX SDK 1.x single dll (Stellar Blade), the FSR 4 SDK loader (Magpie's FSR3/FSR4
  // effects: amd_fidelityfx_loader_dx12.dll exports ffxDispatch and forwards to the provider dll), or the self-contained FFX provider
  // dlls that UE's FSR plugin loads directly from Engine/Plugins/Marketplace/FSR (signedbin: upscaler, then frame generation; both
  // export the ffx* API, the dispatch handler filters on the upscaler header type) or XeSS (Rise of the Ronin; its FSR is linked into the exe).
  // A host that ships an FFX dll next to its exe (Magpie: libxess.dll is loaded at startup, the FFX loader only when the FSR3/FSR4 effect
  // starts) waits for the FFX one; XeSS is taken only when no FFX dll file is present (Rise of the Ronin). DLSS5_UPSCALER=ffx|xess overrides.
  bool wait_ffx=false;{wchar_t exe[MAX_PATH]{};GetModuleFileNameW(nullptr,exe,MAX_PATH);if(wchar_t*slash=wcsrchr(exe,L'\\'))slash[1]=0;std::wstring dir=exe;
   const wchar_t*const ffx_files[]={L"amd_fidelityfx_dx12.dll",L"amd_fidelityfx_loader_dx12.dll",L"amd_fidelityfx_upscaler_dx12.dll",L"amd_fidelityfx_framegeneration_dx12.dll"};
   for(const wchar_t*name:ffx_files)if(GetFileAttributesW((dir+name).c_str())!=INVALID_FILE_ATTRIBUTES){wait_ffx=true;break;}}
 if(const wchar_t*u=_wgetenv(L"DLSS5_UPSCALER")){if(!wcscmp(u,L"ffx"))wait_ffx=true;else if(!wcscmp(u,L"xess"))wait_ffx=false;}
 /* weights into memory while we wait for the upscaler dll / the user's hotkey (see NativePrefetchWeights) */
 {std::wstring assets=NativeLabPath(L"native-game-tiled-assets");if(GetFileAttributesW(assets.c_str())!=INVALID_FILE_ATTRIBUTES)NativePrefetchWeights(assets);}
 /* No deadline: Magpie loads the FFX dll only when the user starts scaling, which can be any time after launch (the old 10-minute limit gave up before that). */
  HMODULE module=nullptr,xess=nullptr;for(unsigned i=0;!module&&!xess;i++){module=GetModuleHandleW(L"amd_fidelityfx_dx12.dll");if(!module)module=GetModuleHandleW(L"amd_fidelityfx_loader_dx12.dll");if(!module)module=GetModuleHandleW(L"amd_fidelityfx_upscaler_dx12.dll");if(!module)module=GetModuleHandleW(L"amd_fidelityfx_framegeneration_dx12.dll");if(!wait_ffx)xess=GetModuleHandleW(L"libxess.dll");if(!module&&!xess)Sleep(100);}if(!module&&!xess)return 1;
 auto target=module?GetProcAddress(module,"ffxDispatch"):GetProcAddress(xess,"xessD3D12Execute");if(!target)return 2;
  auto s=MH_Initialize();if(s!=MH_OK&&s!=MH_ERROR_ALREADY_INITIALIZED)return 3;
  if(module&&fit_small_input()){
   /* Magpie unloads the FFX loader when scaling stops. Hook trampolines must remain executable across restarts. */
   HMODULE pinned=nullptr;
   if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,reinterpret_cast<LPCWSTR>(target),&pinned))return 5;
   auto destroy=GetProcAddress(module,"ffxDestroyContext");
   auto ds=destroy?MH_CreateHook(reinterpret_cast<void*>(destroy),reinterpret_cast<void*>(&destroy_context),reinterpret_cast<void**>(&original_destroy)):MH_ERROR_NOT_EXECUTABLE;
   if(ds==MH_OK)ds=MH_EnableHook(reinterpret_cast<void*>(destroy));
   if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu fit_context_destroy_hook=%u\n",GetCurrentProcessId(),unsigned(ds));fclose(f);}
   if(ds!=MH_OK)return 5;
  }
#ifdef NATIVE_ORDER_NEURAL
 if(!module){NativeMotionSign()=-1.f;cross_thread_submit=true;s=MH_CreateHook(reinterpret_cast<void*>(target),reinterpret_cast<void*>(&xess_execute),reinterpret_cast<void**>(&original_xess));}else
#endif
 s=MH_CreateHook(reinterpret_cast<void*>(target),reinterpret_cast<void*>(&dispatch),reinterpret_cast<void**>(&original));if(s==MH_OK)s=MH_EnableHook(reinterpret_cast<void*>(target));
 // Do not retry an existing-hook conflict or modify another addon's hook.
 wchar_t loader_ver[64]=L"unknown";if(module)module_file_version(module,loader_ver,64);
 {if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu hook_status=%u upscaler=%s version=%ls\n",GetCurrentProcessId(),unsigned(s),module?"ffx":"xess",loader_ver);fclose(f);}}
#ifdef NATIVE_ORDER_NEURAL
 // Some titles (Cyberpunk 2077) call the older, non-unified FSR3 upscaler
 // entry point directly rather than the unified ffxDispatch hooked above -
 // hook that too. Independent and best-effort: never blocks or fails the
 // primary hook, own log line, bounded poll in case it loads later.
 {
  HMODULE fsr3upscaler=nullptr;
  for(unsigned i=0;!fsr3upscaler&&i<50;i++){fsr3upscaler=GetModuleHandleW(L"ffx_fsr3upscaler_x64.dll");if(!fsr3upscaler)Sleep(100);}
  if(fsr3upscaler){
   // ffx_fsr3upscaler_x64.dll carries no version resource of its own in this
   // title; the FFX loader dll found above (amd_fidelityfx_dx12.dll et al.)
   // is what the hardcoded 176-byte struct layout was actually confirmed
   // against, so gate on its version instead.
   fsr3_layout_trusted=module&&fsr3_layout_matches(loader_ver);
   if(auto fsr3target=GetProcAddress(fsr3upscaler,"ffxFsr3UpscalerContextDispatch")){
    auto fs=MH_CreateHook(reinterpret_cast<void*>(fsr3target),reinterpret_cast<void*>(&fsr3_upscaler_dispatch),
     reinterpret_cast<void**>(&original_fsr3upscaler));
    if(fs==MH_OK)fs=MH_EnableHook(reinterpret_cast<void*>(fsr3target));
    if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
     fprintf(f,"pid=%lu hook_status=%u upscaler=fsr3upscaler loader_version=%ls layout_trusted=%u\n",GetCurrentProcessId(),unsigned(fs),loader_ver,unsigned(fsr3_layout_trusted.load()));fclose(f);
    }
   }else if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
    fprintf(f,"pid=%lu kind=fsr3upscaler_symbol_not_found loader_version=%ls\n",GetCurrentProcessId(),loader_ver);fclose(f);
   }
  }else if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){
   fprintf(f,"pid=%lu kind=fsr3upscaler_dll_not_loaded_after_5s\n",GetCurrentProcessId());fclose(f);
  }
 }
#endif
 return s==MH_OK?0:4;
}
// Before the game creates its D3D12 device: select the private Agility 721 runtime shipped in
// the game folder and enable the experimental shader-model feature so SM6.10 wave-matrix PSOs
// can be created on the game device. Gated by D:\DLSSNR-Lab\enable-game-sdk721.txt.
static bool on_create_device(reshade::api::device_api api,uint32_t&){
 if(api!=reshade::api::device_api::d3d12||GetFileAttributesW(NativeLabPath(L"enable-game-sdk721.txt").c_str())==INVALID_FILE_ATTRIBUTES)return false;
 static std::atomic<bool>attempted{false};if(attempted.exchange(true))return false;
 /* Diagnostic (D:\DLSSNR-Lab\enable-dred.txt): Device Removed Extended Data, breadcrumbs + page faults, dumped by the frame when initialization fails. */
 if(GetFileAttributesW(NativeLabPath(L"enable-dred.txt").c_str())!=INVALID_FILE_ATTRIBUTES){ID3D12DeviceRemovedExtendedDataSettings*dred=nullptr;HRESULT dh=D3D12GetDebugInterface(IID_PPV_ARGS(&dred));if(SUCCEEDED(dh)&&dred){dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);dred->Release();}
  if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu dred_enable hr=%08x\n",GetCurrentProcessId(),unsigned(dh));fclose(f);}}
 const GUID clsid={0x7cda6aca,0xa03e,0x49c8,{0x94,0x58,0x03,0x34,0xd2,0x0e,0x07,0xce}};
 using GetInterfaceFn=HRESULT(WINAPI*)(REFCLSID,REFIID,void**);HMODULE d3d=GetModuleHandleW(L"d3d12.dll");auto get_interface=d3d?reinterpret_cast<GetInterfaceFn>(GetProcAddress(d3d,"D3D12GetInterface")):nullptr;
 ID3D12SDKConfiguration*configuration=nullptr;HRESULT get=get_interface?get_interface(clsid,IID_PPV_ARGS(&configuration)):HRESULT_FROM_WIN32(ERROR_PROC_NOT_FOUND),set=E_ABORT,experimental=E_ABORT;
 if(SUCCEEDED(get)){set=configuration->SetSDKVersion(721,".\\DLSS5-D3D12-721\\");configuration->Release();}
 /* (after SetSDKVersion so the Agility folder's d3d12SDKLayers.dll is used) Diagnostic (D:\DLSSNR-Lab\enable-d3d12-debug.txt): the D3D12 debug layer (d3d12SDKLayers.dll from the Agility folder); its messages are dumped when initialization fails. */
#if NATIVE_HAVE_SDKLAYERS
 if(GetFileAttributesW(NativeLabPath(L"enable-d3d12-debug.txt").c_str())!=INVALID_FILE_ATTRIBUTES){ID3D12Debug*dbg=nullptr;HRESULT bh=D3D12GetDebugInterface(IID_PPV_ARGS(&dbg));if(SUCCEEDED(bh)&&dbg){dbg->EnableDebugLayer();dbg->Release();}
  if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu debug_layer hr=%08x\n",GetCurrentProcessId(),unsigned(bh));fclose(f);}}
#endif
 if(SUCCEEDED(set)){const GUID feature={0x76f5573e,0xf13a,0x40f5,{0xb2,0x97,0x81,0xce,0x9e,0x18,0x93,0x3f}};experimental=D3D12EnableExperimentalFeatures(1,&feature,nullptr,nullptr);}
 if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu sdk721_before_device get=%08x set=%08x experimental=%08x\n",GetCurrentProcessId(),unsigned(get),unsigned(set),unsigned(experimental));fclose(f);}
 return false;
}
// VRAM reservation (DLSS5_RESERVE_VRAM_MB=N in native-game-flags.txt): right after the game creates its device, hold N MB of
// video memory in a placeholder buffer so the game sizes its texture pool with that much less; the placeholder is released
// just before the network allocates its own buffers, which then take that room instead of pushing the game's textures out.
static ID3D12Resource*reserved_vram=nullptr;
void NativeReleaseReservedVram(){if(reserved_vram){reserved_vram->Release();reserved_vram=nullptr;if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu reserved_vram_released\n",GetCurrentProcessId());fclose(f);}}}
static void on_init_device(reshade::api::device*device){
 if(!device||device->get_api()!=reshade::api::device_api::d3d12||reserved_vram)return;
 unsigned long mb=0;if(FILE*flags=_wfopen(NativeLabPath(L"native-game-flags.txt").c_str(),L"rb")){char line[256];while(fgets(line,sizeof line,flags))if(!strncmp(line,"DLSS5_RESERVE_VRAM_MB=",22))mb=strtoul(line+22,nullptr,10);fclose(flags);}
 if(!mb)return;auto*d=reinterpret_cast<ID3D12Device*>(device->get_native());
 D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC rd{};rd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;rd.Width=UINT64(mb)<<20;rd.Height=1;rd.DepthOrArraySize=rd.MipLevels=1;rd.SampleDesc.Count=1;rd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
 HRESULT hr=d->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&reserved_vram));
 if(FILE*f=_wfopen(NativeLabPath(L"logs\\native-submission-order.txt").c_str(),L"ab")){fprintf(f,"pid=%lu reserved_vram_mb=%lu hr=%08x\n",GetCurrentProcessId(),mb,unsigned(hr));fclose(f);}
}
BOOL WINAPI DllMain(HINSTANCE h,DWORD reason,LPVOID){
 if(reason==DLL_PROCESS_ATTACH){
  DisableThreadLibraryCalls(h);wchar_t path[MAX_PATH]{};GetModuleFileNameW(nullptr,path,MAX_PATH);
  /* Optional process filter: DLSS5_ONLY_EXE=<substring of the host exe path> makes every other process refuse the add-on (the old
     hard-coded SB-Win64-Shipping.exe / Ronin.exe list returned FALSE here = ReShade error 1114 in any other game). */
  if(const wchar_t*only=_wgetenv(L"DLSS5_ONLY_EXE"))if(*only&&!wcsstr(path,only))return FALSE;
  if(!reshade::register_addon(h))return FALSE;
  reshade::register_event<reshade::addon_event::create_device>(on_create_device);
  reshade::register_event<reshade::addon_event::init_device>(on_init_device);
  reshade::register_event<reshade::addon_event::close_command_list>(close_list);
  reshade::register_event<reshade::addon_event::execute_command_list>(execute);
  reshade::register_event<reshade::addon_event::dispatch>(compute);
  reshade::register_event<reshade::addon_event::draw>(draw);
  reshade::register_event<reshade::addon_event::barrier>(barrier);
  HMODULE pinned=nullptr;if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,reinterpret_cast<LPCWSTR>(&worker),&pinned))return FALSE;
  HANDLE thread=CreateThread(nullptr,0,worker,nullptr,0,nullptr);if(!thread)return FALSE;CloseHandle(thread);
 }return TRUE;
}
