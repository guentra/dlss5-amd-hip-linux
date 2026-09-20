#pragma once
#include <atomic>
#include <cwchar>
// ffx_fsr3upscaler_x64.dll's named-resource layout (176-byte stride, `output`
// at offset 1592 - see native_submission_order_probe.cpp) was confirmed
// against a real live capture of one specific FFX loader build. The upscaler
// dll itself carries no version resource in that game, so the loader dll's
// version (amd_fidelityfx_dx12.dll et al., read via module_file_version) is
// what the layout is actually gated on. No Windows dependency here so this
// stays host-testable (see linux/tests/test_fsr3_layout_gate.py). Both
// helpers are inline rather than static: a translation unit that uses only
// one of them would otherwise warn about the other being defined and
// unused, which -Werror turns into a build failure.
static constexpr wchar_t kConfirmedFsr3LoaderVersion[]=L"1.0.1.41314";
inline bool fsr3_layout_matches(const wchar_t*loader_version){
 return loader_version&&!wcscmp(loader_version,kConfirmedFsr3LoaderVersion);
}
// Whether one dispatch may be handed to the live HIP path. Kept here, in
// plain types, because the call site lives in a translation unit that pulls
// in D3D12, ReShade and MinHook and so cannot be compiled by the host test
// suite - while this decision is the part that actually has to be right.
// Each condition is a safety condition, and dropping any of them fails
// quietly rather than loudly:
//   layout_trusted     - another FFX build lays the dispatch description out
//                        differently; reading it hands D3D12 garbage pointers
//   dispatch_ok        - the upscaler itself failed, so `output` holds
//                        nothing the network should consume
//   command_list       - nothing to record the copies into
//   output_resource    - nothing to read from or write back to
//   output_state_known - the declared FFX resource state has no D3D12
//                        equivalent, so the barriers cannot be formed
inline bool fsr3_route_to_live(bool layout_trusted,bool dispatch_ok,
                               const void*command_list,const void*output_resource,
                               bool output_state_known){
 return layout_trusted&&dispatch_ok&&command_list&&output_resource&&output_state_known;
}
// Exactly one dispatch entry point may feed the live path.
//
// A title can plausibly reach both: the unified ffxDispatch is free to be
// implemented on top of the FSR3-specific entry point, in which case both
// hooks fire for the same frame. Routing both would run the network twice on
// one frame, the second pass consuming the first's output - a doubled cost
// and a doubly-processed image, with nothing reporting an error. The
// version gate makes that unlikely but cannot rule it out, since a title
// shipping the confirmed loader build could still use either API.
//
// So the first route to reach a frame claims the path for the life of the
// process and the other stays off. Re-claiming by the owner is free, so this
// costs one uncontended compare-exchange per frame.
enum LiveRouteOwner{LiveRouteNone=0,LiveRouteUnified=1,LiveRouteFsr3=2};
inline bool claim_live_route(std::atomic<int>&owner,int who){
 int expected=LiveRouteNone;
 return owner.compare_exchange_strong(expected,who)||expected==who;
}
