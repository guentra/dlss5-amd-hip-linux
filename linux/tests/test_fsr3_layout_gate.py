"""Compile the real FSR3 layout-version gate natively and exercise it directly.

fsr3_layout_matches() decides whether the hardcoded 176-byte FSR3 upscaler
struct layout (native_submission_order_probe.cpp) is trusted for the FFX
loader build actually running the game - and therefore whether captured
frames are fed into HIP at all. It has no Windows dependency, so it is
included and run as plain C++ here rather than through Wine.
"""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class Fsr3LayoutGateTests(unittest.TestCase):
    def _build_and_run(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / 'test.cpp'
            src.write_text(source)
            exe = pathlib.Path(tmp) / 'test'
            build = subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                                    '-I', str(ROOT), str(src), '-o', str(exe)],
                                   text=True, capture_output=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(exe)], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_routes_only_when_every_safety_condition_holds(self):
        """The live route must open only when all five conditions hold.

        Each one prevents a different concrete failure, and each fails
        silently if dropped: an untrusted layout hands D3D12 pointers read
        out of a differently-shaped struct, a failed dispatch means `output`
        holds nothing worth consuming, a missing command list or output
        resource means there is nothing to record or write back, and an
        unmappable FFX resource state means the barriers around the copy
        cannot be formed. None of these produce an error on their own - they
        produce a wrong frame, or a corrupted one."""
        source = r'''
#include <cassert>
#include "src/native_fsr3_layout_gate.h"
int main(){
 int list_object=0,output_object=0;
 void*list=&list_object;void*output=&output_object;
 // Everything in order: this is the only combination that may route.
 assert(fsr3_route_to_live(true,true,list,output,true));
 // Each condition alone withdraws the route.
 assert(!fsr3_route_to_live(false,true,list,output,true));   // untrusted layout
 assert(!fsr3_route_to_live(true,false,list,output,true));   // upscaler failed
 assert(!fsr3_route_to_live(true,true,nullptr,output,true)); // no command list
 assert(!fsr3_route_to_live(true,true,list,nullptr,true));   // no output resource
 assert(!fsr3_route_to_live(true,true,list,output,false));   // state not mappable
 // And nothing routes when nothing is right.
 assert(!fsr3_route_to_live(false,false,nullptr,nullptr,false));
 return 0;
}
'''
        self._build_and_run(source)

    def test_only_one_dispatch_route_ever_owns_the_live_path(self):
        """Two entry points can reach the same frame; only one may route it.

        The unified ffxDispatch may be implemented on top of the FSR3-specific
        entry point, so both hooks can fire for one frame. Routing both would
        run the network twice on that frame, the second pass consuming the
        first's output - doubled cost and a doubly-processed image, with
        nothing reporting an error. The first route to arrive claims the path
        and keeps it; the other is refused for the life of the process."""
        source = r'''
#include <cassert>
#include "src/native_fsr3_layout_gate.h"
int main(){
 {  // Unified arrives first and keeps the path.
  std::atomic<int> owner{LiveRouteNone};
  assert(claim_live_route(owner,LiveRouteUnified));
  assert(!claim_live_route(owner,LiveRouteFsr3));
  assert(claim_live_route(owner,LiveRouteUnified));   // owner re-claims freely
  assert(!claim_live_route(owner,LiveRouteFsr3));     // and the loser stays out
 }
 {  // Exactly the same the other way round - neither entry point is favoured.
  std::atomic<int> owner{LiveRouteNone};
  assert(claim_live_route(owner,LiveRouteFsr3));
  assert(!claim_live_route(owner,LiveRouteUnified));
  assert(claim_live_route(owner,LiveRouteFsr3));
 }
 return 0;
}
'''
        self._build_and_run(source)

    def test_matches_only_the_confirmed_loader_build(self):
        source = r'''
#include <cassert>
#include "src/native_fsr3_layout_gate.h"
int main(){
 assert(fsr3_layout_matches(kConfirmedFsr3LoaderVersion));
 assert(!fsr3_layout_matches(L"unknown"));
 assert(!fsr3_layout_matches(L""));
 assert(!fsr3_layout_matches(L"1.0.1.41313"));
 assert(!fsr3_layout_matches(L"1.0.1.413140"));
 assert(!fsr3_layout_matches(nullptr));
 return 0;
}
'''
        self._build_and_run(source)


if __name__ == '__main__':
    unittest.main()
