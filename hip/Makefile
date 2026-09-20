# Override any of: HIPCC, ROCM_PATH, HIP_PATH, ARCH, MINGW_CC, MINGW_PREFIX, TOOLCHAIN_DIR.
ARCH ?= gfx1201
BUILD ?= build

# Prefer an explicit ROCm prefix over `hipcc` on PATH: pip/venv `rocm-sdk-core`
# wrappers often shadow the device compiler. GNU make treats `)` as the end of
# $(shell ...), so discovery stays in make functions / simple commands.
ifeq ($(origin ROCM_PATH), undefined)
  ifneq ($(HIP_PATH),)
    ROCM_PATH := $(HIP_PATH)
  else
    ROCM_PATH := $(firstword \
      $(if $(wildcard /opt/rocm/bin/hipcc),/opt/rocm) \
      $(patsubst %/bin/hipcc,%,$(wildcard /opt/rocm-[0-9]*/bin/hipcc)))
  endif
endif
ifeq ($(ROCM_PATH),)
  ROCM_PATH := $(shell hipconfig --rocmpath 2>/dev/null)
  ifneq ($(findstring site-packages,$(ROCM_PATH)),)
    ROCM_PATH :=
  endif
  ifneq ($(findstring rocm_sdk,$(ROCM_PATH)),)
    ROCM_PATH :=
  endif
endif
ifeq ($(ROCM_PATH),)
  _HIPCC_ON_PATH := $(shell command -v hipcc 2>/dev/null)
  ifeq ($(findstring site-packages,$(_HIPCC_ON_PATH))$(findstring rocm_sdk,$(_HIPCC_ON_PATH)),)
    ifneq ($(_HIPCC_ON_PATH),)
      ROCM_PATH := $(abspath $(dir $(_HIPCC_ON_PATH))/..)
    endif
  endif
endif

ifeq ($(origin HIPCC), undefined)
  ifneq ($(wildcard $(ROCM_PATH)/bin/hipcc),)
    HIPCC := $(ROCM_PATH)/bin/hipcc
  else
    HIPCC := $(shell command -v hipcc 2>/dev/null)
  endif
endif

INCLUDES := -Iinclude
LDFLAGS := -lamdhip64
ifneq ($(ROCM_PATH),)
  INCLUDES += -I$(ROCM_PATH)/include
  LDFLAGS := -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib
endif
CXXFLAGS := -O3 -std=c++17 --offload-arch=$(ARCH) $(INCLUDES)
TESTS := test_wmma test_stages test_quantize test_attention test_residual test_linear test_logical_ops test_ffn_f32

# Preference: explicit MINGW_PREFIX, then the self-contained llvm-mingw
# toolchains (their PEs carry no libgcc runtime imports), and only as a last
# resort the distribution mingw found on PATH. A distro mingw GCC emits a
# libgcc_s_seh-1.dll import that Proton prefixes do not provide, breaking
# LoadLibrary of the trampoline in-game.
ifeq ($(origin MINGW_CC), undefined)
  ifneq ($(MINGW_PREFIX),)
    MINGW_CC := $(MINGW_PREFIX)/bin/x86_64-w64-mingw32-gcc
  else
    MINGW_CC := $(firstword $(wildcard \
      $(CURDIR)/../toolchain/llvm-mingw*/bin/x86_64-w64-mingw32-gcc \
      $(CURDIR)/../../toolchain/llvm-mingw*/bin/x86_64-w64-mingw32-gcc))
  endif
endif
ifeq ($(MINGW_CC),)
  MINGW_CC := $(firstword $(wildcard \
    $(if $(TOOLCHAIN_DIR),$(TOOLCHAIN_DIR)/bin/x86_64-w64-mingw32-gcc \
                          $(TOOLCHAIN_DIR)/llvm-mingw*/bin/x86_64-w64-mingw32-gcc)))
endif
ifeq ($(MINGW_CC),)
  MINGW_CC := $(shell command -v x86_64-w64-mingw32-gcc 2>/dev/null)
endif

export HIPCC ROCM_PATH ARCH

.PHONY: all clean test game test-api bench-ffn32

bench-ffn32: tests/bench_ffn32.hip $(BUILD)/kernels.o $(BUILD)/kernels_prod.o $(BUILD)/kernels_mh1.o $(BUILD)/kernels_mh2.o include/*.hpp
	$(HIPCC) $(CXXFLAGS) -c tests/bench_ffn32.hip -o $(BUILD)/bench_ffn32.o
	$(HIPCC) $(CXXFLAGS) $(BUILD)/bench_ffn32.o $(BUILD)/kernels.o $(BUILD)/kernels_prod.o $(BUILD)/kernels_mh1.o $(BUILD)/kernels_mh2.o $(LDFLAGS) -o $@

all: $(TESTS) hip-network70 libdlss5_hip.so

game: libdlss5_hip.so dlss5_hip.dll

test_wmma: tests/test_wmma.hip include/dlss5_linalg.hpp include/dlss5_common.hpp include/dlss5_runtime.hpp
	$(HIPCC) $(CXXFLAGS) tests/test_wmma.hip $(LDFLAGS) -o $@

$(BUILD)/kernels.o: src/kernels.hip include/*.hpp kernels/*.hip Makefile
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -fPIC -c src/kernels.hip -o $@

$(BUILD)/kernels_prod.o: src/kernels_prod.hip include/*.hpp kernels/c32_prod.hip Makefile
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -fPIC -DHIP_PREPACKED_WEIGHTS -c src/kernels_prod.hip -o $@

$(BUILD)/kernels_mh1.o: src/kernels_mh1.hip include/*.hpp kernels/mh_prod1.hip Makefile
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -fPIC -c src/kernels_mh1.hip -o $@

$(BUILD)/kernels_mh2.o: src/kernels_mh2.hip include/*.hpp kernels/mh_prod2.hip Makefile
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -fPIC -c src/kernels_mh2.hip -o $@

$(BUILD)/network.o: src/network.hip include/*.hpp Makefile
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -fPIC -c src/network.hip -o $@

$(BUILD)/gpu_interop.o: src/gpu_interop.hip include/*.hpp include/*.h kernels/gpu_convert.hip Makefile
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -fPIC -c $< -o $@

$(BUILD)/%.o: src/%.hip include/*.hpp include/*.h Makefile
	mkdir -p $(BUILD)
	$(HIPCC) $(CXXFLAGS) -fPIC -c $< -o $@

$(filter-out test_wmma,$(TESTS)): %: tests/%.hip $(BUILD)/kernels.o $(BUILD)/kernels_prod.o $(BUILD)/kernels_mh1.o $(BUILD)/kernels_mh2.o include/*.hpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $(BUILD)/$@.o
	$(HIPCC) $(CXXFLAGS) $(BUILD)/$@.o $(BUILD)/kernels.o $(BUILD)/kernels_prod.o $(BUILD)/kernels_mh1.o $(BUILD)/kernels_mh2.o $(LDFLAGS) -o $@

hip-network70: $(BUILD)/bench.o $(BUILD)/kernels.o $(BUILD)/kernels_prod.o $(BUILD)/kernels_mh1.o $(BUILD)/kernels_mh2.o $(BUILD)/network.o
	$(HIPCC) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

libdlss5_hip.so: $(BUILD)/capi.o $(BUILD)/frame.o $(BUILD)/kernels.o $(BUILD)/kernels_prod.o $(BUILD)/kernels_mh1.o $(BUILD)/kernels_mh2.o $(BUILD)/network.o $(BUILD)/gpu_interop.o
	$(HIPCC) $(CXXFLAGS) -shared -fPIC $^ $(LDFLAGS) -o $@

dlss5_hip.dll: src/trampoline.c include/dlss5_capi.h
	@test -n "$(MINGW_CC)" || { printf '%s\n' 'x86_64-w64-mingw32-gcc not found. Set MINGW_CC= or MINGW_PREFIX= or install mingw-w64.'; exit 2; }
	# -static: a GCC-flavoured mingw (Debian/Ubuntu's gcc-mingw-w64-x86-64,
	# which is what the discovery above finds when no llvm-mingw drop is
	# installed) links this against libgcc_s_seh-1.dll. That file is not in
	# the game's directory, so LoadLibrary fails and the add-on reports the
	# trampoline as MISSING - which reads as "the file is absent" while it
	# sits right there next to the caller. -static pulls in libgcc,
	# libwinpthread and libstdc++ all at once; llvm-mingw needs no such flag
	# and is unaffected by it.
	$(MINGW_CC) -shared -O2 -static -o $@ src/trampoline.c

test: $(TESTS)
	@set -e; for t in $(TESTS); do ./$$t; done

test-api: libdlss5_hip.so
	$(HIPCC) $(CXXFLAGS) tests/test_buffer.hip $(LDFLAGS) -o $(BUILD)/test_buffer
	$(BUILD)/test_buffer
	$(HIPCC) $(CXXFLAGS) tests/test_capi.hip -L. -ldlss5_hip -Wl,-rpath,'$$ORIGIN/..' $(LDFLAGS) -o $(BUILD)/test_capi
	$(BUILD)/test_capi
	python3 tests/test_publish.py "$(CURDIR)/libdlss5_hip.so"

.PHONY: test-frame test-graph
test-frame: $(BUILD)/frame.o
	$(HIPCC) $(CXXFLAGS) -c tests/test_frame.hip -o $(BUILD)/test_frame.o
	$(HIPCC) $(BUILD)/test_frame.o $(BUILD)/frame.o $(LDFLAGS) -o $(BUILD)/test_frame
	$(BUILD)/test_frame
	$(HIPCC) $(CXXFLAGS) -c tests/test_temporal.hip -o $(BUILD)/test_temporal.o
	$(HIPCC) $(BUILD)/test_temporal.o $(BUILD)/frame.o $(LDFLAGS) -o $(BUILD)/test_temporal
	$(BUILD)/test_temporal

test-graph:
	HIPCC="$(HIPCC)" ROCM_PATH="$(ROCM_PATH)" ARCH="$(ARCH)" bash tests/run_graph_tests.sh

.PHONY: check test-python test-model
# Run the GPU families serially even when the build uses -j.
check:
	$(MAKE) game hip-network70
	$(MAKE) test-api
	$(MAKE) test-frame
	$(MAKE) test-graph
	$(MAKE) test-python

test-python: hip-network70
	python3 tests/test_bench.py "$(CURDIR)/hip-network70"
	python3 tests/test_infer_image.py
	python3 tests/test_make_targets.py
	python3 -m unittest discover -s ../linux/tests -v

# Explicit real weights only: synthetic graph fixtures belong to test-graph.
test-model:
	@test -n "$(WEIGHTS)" || { printf '%s\n' 'Set WEIGHTS=/path/to/complete/tables'; exit 2; }
	$(MAKE) libdlss5_hip.so
	$(HIPCC) $(CXXFLAGS) -c tests/test_capi_lifecycle.hip -o $(BUILD)/test_capi_lifecycle.o
	$(HIPCC) $(BUILD)/test_capi_lifecycle.o -L. -ldlss5_hip -Wl,-rpath,"$(CURDIR)" $(LDFLAGS) -o $(BUILD)/test_capi_lifecycle
	$(BUILD)/test_capi_lifecycle "$(WEIGHTS)"
	$(HIPCC) $(CXXFLAGS) -c tests/test_frame_api.hip -o $(BUILD)/test_frame_api.o
	$(HIPCC) $(BUILD)/test_frame_api.o -L. -ldlss5_hip -Wl,-rpath,"$(CURDIR)" $(LDFLAGS) -o $(BUILD)/test_frame_api
	$(BUILD)/test_frame_api "$(WEIGHTS)"

clean:
	rm -f $(TESTS) hip-network70 libdlss5_hip.so dlss5_hip.dll $(BUILD)/*.o $(BUILD)/test_capi
