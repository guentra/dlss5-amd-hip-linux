#!/usr/bin/env bash
# Isolated native GPU tests. Never installs/launches a game or executes vendor code.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=${DLSS5_GRAPH_TEST_BUILD:-linux/build/graph-audit}
export DLSS5_GRAPH_TEST_BUILD="$B"
ARCH="${ARCH:-gfx1201,gfx1200}"
if [[ -z "${ROCM_PATH:-}" && -n "${HIP_PATH:-}" ]]; then
  ROCM_PATH="$HIP_PATH"
fi
if [[ -z "${HIPCC:-}" ]]; then
  if [[ -n "${ROCM_PATH:-}" && -x "${ROCM_PATH}/bin/hipcc" ]]; then
    HIPCC="${ROCM_PATH}/bin/hipcc"
  elif [[ -x /opt/rocm/bin/hipcc ]]; then
    HIPCC=/opt/rocm/bin/hipcc
    ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
  else
    found=
    while IFS= read -r p; do
      [[ -x "$p/bin/hipcc" ]] && found=$p
    done < <(ls -1d /opt/rocm-[0-9]* 2>/dev/null | sort -V; true)
    if [[ -n "$found" ]]; then
      HIPCC="$found/bin/hipcc"
      ROCM_PATH="${ROCM_PATH:-$found}"
    else
      HIPCC="$(command -v hipcc || true)"
    fi
  fi
fi
if [[ -z "${HIPCC:-}" ]]; then
  printf '%s\n' 'hipcc not found. Set HIPCC= or ROCM_PATH= to your ROCm install.' >&2
  exit 2
fi
if [[ -z "${ROCM_PATH:-}" ]]; then
  ROCM_PATH="$(dirname "$(dirname "$(readlink -f "$HIPCC" 2>/dev/null || printf '%s' "$HIPCC")")")"
fi
mkdir -p "$B"
OFFLOAD=()
IFS=',' read -ra _archs <<< "$ARCH"
for _arch in "${_archs[@]}"; do
  _arch="${_arch// /}"
  [[ -n "$_arch" ]] && OFFLOAD+=(--offload-arch="$_arch")
done
FLAGS=("${OFFLOAD[@]}" -O3 -std=c++17 -Ihip/include)
if [[ -d "${ROCM_PATH}/include" ]]; then
  FLAGS+=(-I"${ROCM_PATH}/include")
fi
LINK_FLAGS=()
if [[ -d "${ROCM_PATH}/lib" ]]; then
  LINK_FLAGS+=(-L"${ROCM_PATH}/lib" -Wl,-rpath,"${ROCM_PATH}/lib")
fi
"$HIPCC" "${FLAGS[@]}" -c hip/src/kernels.hip -o "$B/kernels.o"
"$HIPCC" "${FLAGS[@]}" -DHIP_PREPACKED_WEIGHTS -c hip/src/kernels_prod.hip -o "$B/kernels_prod.o"
"$HIPCC" "${FLAGS[@]}" -c hip/src/kernels_mh1.hip -o "$B/kernels_mh1.o"
"$HIPCC" "${FLAGS[@]}" -c hip/src/kernels_mh2.hip -o "$B/kernels_mh2.o"
"$HIPCC" "${FLAGS[@]}" -c hip/src/network.hip -o "$B/network.o"
"$HIPCC" "${FLAGS[@]}" -c hip/src/gpu_interop.hip -o "$B/gpu_interop.o"
"$HIPCC" "${FLAGS[@]}" -c hip/src/frame.hip -o "$B/frame.o"
"$HIPCC" "${FLAGS[@]}" -c hip/src/capi.hip -o "$B/capi.o"
for t in test_wmma test_stages test_quantize test_attention test_residual test_linear test_logical_ops test_ffn_f32 test_graph_ops test_graph_blocks test_graph_stream test_graph_vit test_network test_configured_path test_gpu_convert test_raw_pipeline; do
  precision=(-ffp-contract=off)
  # The precise C32 regression must run under production contraction settings.
  if [[ "$t" == test_configured_path ]]; then precision=(); fi
  "$HIPCC" "${FLAGS[@]}" "${precision[@]}" -c "hip/tests/$t.hip" -o "$B/$t.o"
  objects=("$B/$t.o" "$B/kernels.o" "$B/kernels_prod.o" "$B/kernels_mh1.o" "$B/kernels_mh2.o")
  # Block and ViT tests include network.hip to reach the real internal scheduler.
  if [[ "$t" == test_network ]]; then objects+=("$B/network.o"); fi
  # Pixel conversion test links only the interop object (kernels inside it).
  if [[ "$t" == test_gpu_convert ]]; then objects=("$B/$t.o" "$B/gpu_interop.o"); fi
  # Raw pipeline test links the full C ABI frame path.
  if [[ "$t" == test_raw_pipeline ]]; then
    objects=("$B/$t.o" "$B/capi.o" "$B/frame.o" "$B/kernels.o" "$B/kernels_prod.o" "$B/kernels_mh1.o" "$B/kernels_mh2.o" "$B/network.o" "$B/gpu_interop.o")
  fi
  "$HIPCC" "${objects[@]}" "${LINK_FLAGS[@]}" -o "$B/$t"
  log="$B/$t.log"
  if [[ "$t" == test_network ]]; then log="$B/network-final.log"; fi
  timeout "${DLSS5_GRAPH_TEST_TIMEOUT:-300}" "$B/$t" > "$log" 2>&1
  printf '%s PASS\n' "$t"
done
python3 hip/tests/test_graph_evidence.py
