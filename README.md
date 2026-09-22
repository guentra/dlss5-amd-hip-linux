# DLSS5-AMD HIP for Linux

> **It still needs substantial optimization.** Expect low frame rates (~25-28 fps on SILENT HILL f, RX 9070 XT), high latency and possible rendering problems like ghosting and shimmering. It is not an official NVIDIA DLSS implementation or a claim of equivalent image quality.

A native Linux HIP/rocWMMA implementation of the complete 71-block DLSS5 network for AMD `gfx1200` (RX 9060) and `gfx1201` (RX 9070), with a ReShade add-on, Wine bridge and modified vkd3d-proton submission path. NVIDIA DLLs and weights are **not included**.

## Download and install

**[Download v0.2.6 .tar.gz](https://github.com/guentra/dlss5-amd-hip-linux/releases/download/v0.2.6/dlss5-amd-hip-linux.tar.gz)** · [SHA256](https://github.com/guentra/dlss5-amd-hip-linux/releases/download/v0.2.6/dlss5-amd-hip-linux.tar.gz.sha256) · [Release notes](https://github.com/guentra/dlss5-amd-hip-linux/releases/tag/v0.2.6)

1. Close the game. Extract the archive inside its directory, keeping the `dlss5-amd-hip-linux` subfolder.
2. Put your legitimately obtained `nvngx_dlssnr.dll` **310.8.0.0** beside the game executable or in the game root (or select it in the wizard).
3. Open a terminal in the extracted subfolder and run `./install.sh` — **not sudo**.
4. Follow the wizard, explicitly accept experimental reconstructed weight layouts, then use the printed launch wrapper in your launcher. In Steam, paste the printed launch options. Steam is optional.

The archive contains the prebuilt HIP library, Windows bridge, add-on, ReShade loader, matching modified vkd3d DLL pair and offline benchmark. No compiler is needed to use it. ROCm must already be installed; this installer does not install it.

## Requirements and limitations

- Linux x86_64, Python 3.10+, ROCm/HIP 7 and a compatible Wine/Proton runner. Binaries are built for **gfx1200 (RX 9060) and gfx1201 (RX 9070) only**; `gfx1200` is untested and may simply not work. No other AMD GPU or distribution.
- An in-game DX12 FSR/FidelityFX hook. **No anti-cheat games, Magpie support or frame generation in this release.**
- The network operates at 1920×1080 internally; the live hook resizes other supported frame sizes. Motion history is reset for each live frame, so this is **not temporally complete**.
- Some weight layouts are reconstructed from AMD-consumer evidence. Opting in does not establish NVIDIA numerical or visual equivalence.
- The prototype uses CPU readback/upload and HIP execution at a split vkd3d submission boundary. **The game still waits for neural rendering.** It is not an asynchronous performance fix.
- General gameplay stability, HDR behavior and broad game compatibility are not certified. F6 toggles the live path's bypass when the hook is active; it cannot fix a missing hook or failed initialization.

The offline bench (network only, fixed 1080p input, real converted weights, RX 9070 XT `gfx1201`, warm runs) measures **~25 ms GPU time per inference** on the current `0.2.6` build (`0.2.5` ~33 ms, `0.2.3` ~63–66 ms, `0.2.1` ~98–105 ms, `0.1.0-poc` 210–216 ms). That is network-only timing, **not in-game FPS**: in-game SILENT HILL f runs at **~28 fps** on `0.2.6` (up from ~25 fps on `0.2.5`, 15–20 fps on `0.2.3`). Kernel work is tracked in the [Changelog](#changelog); data transfers and memory use remain open targets.

## Changelog

Unless a scenario is specified, timings are the offline bench (network only, fixed 1920×1080 input, real converted weights, RX 9070 XT `gfx1201`); "bit-exact" means the output did not change by a single bit against the reference 71-block network, which stays the judge.

| Version | Date | What changed | Result |
|---|---|---|---|
| `0.1.0-poc` | 09-13 | First complete Linux implementation: all 71 blocks as HIP/rocWMMA wave-matrix kernels; weights converted from the user's `nvngx_dlssnr.dll` 310.8.0.0; Win64→SysV trampoline + ReShade add-on + modified vkd3d-proton hook; offline `hip-network70` bench | bench 210–216 ms, in-game staging only |
| `0.2.0` | 09-14 | add-on sync 0.12 → 0.15 (≤1920×1080 fit, notice + FPS display, driver trap); installer retargeted at the local add-on (wizard, backups, sha256 manifest, test suites); bit-exact kernel work: fused QKV+LayerNorm+quant, window-attention rewrite (padded scores, E4M3 P·V operand, 4 → 2 barriers), exact f16 squares via opaque asm | bench 175 → 122 ms (−30 %), bit-identical |
| `0.2.1` | 09-14 | bit-exact: C32 FFN activation spill eliminated (opaque asm, 2.0 → 0.45 ms/call), LDS aliasing in FFN/QKV (−512 B/block, C32 FFN occupancy 78 → 100 %), `bench_ffn32`. Add-on: recognizes self-contained FFX provider dlls of UE FSR-plugin titles | bench 122 → ~98–105 ms; FSR3/FSR4 takeover verified on Beast of Reincarnation |
| `0.2.2` | 09-15 | bit-exact: E4M3 siblings kept across launch boundaries (C64–C512), C32 one-CTA QKV (`NH=3`) + two 16-token FFN groups, ViT E4M3 feed-through, f8/f16 reframe/pack variants | bench ~98–105 → ~63–66 ms, bit-identical |
| `0.2.3` | 09-19 | Installer: bridge `LD_PRELOAD`ed from a whitespace-free per-digest cache (game paths with spaces broke the bridge); doctor/runtime/status UX, GPU+ROCm detection, non-Steam support; opt-in `DLSS5_FIXED_SEED` | first in-game neural rendering verified: SILENT HILL f, 15–20 fps, F6 bypass |
| `0.2.4` | 09-20 | live path: gated-less temporal blend `out=(1−w)·prev+w·cur` (`DLSS5_TEMPORAL_BLEND`) fixes shimmer; byte-exact fused C64/C128/C256 MH-prod chain on by default | bench ~63–66 → ~50 ms, bit-identical |
| `0.2.5` | 09-21 | frame-difference-gated blend weight (anti-ghosting, `DLSS5_TEMPORAL_BLEND_GATE`); post-70 f16 chain (dead f32 raster dropped); installer UX (overwrite prompt, weights consent, progress); legacy FSR3 dispatch route (PR #4); CI + live test suite fixes | in-game SHf ~25 fps; bench ~33 ms, bit-identical (16/16) |
| `0.2.6` | 09-22 | dual-arch build for `gfx1200` (RX 9060) + `gfx1201` (RX 9070), installer accepts both (fat-container target scan; a `gfx1250` metadata false positive is no longer mistaken for an image); bit-exact C32 work: e4m3 rasters chained between blocks, raster/window siblings read in place, workgroup fences replacing the barriers' implicit SE-scope L1 invalidation, dead decoder group-end rasters dropped | bench ~33 → **~25 ms**; in-game SHf **~28 fps**; bit-identical (16/16). `gfx1200` untested |

## Troubleshooting and removal

If executable detection is ambiguous, see `./install.sh install --help` and use `--exe '/path/to/Game/Binaries/Win64/Game.exe'`. Keep the game's existing prefix and launch arguments. Do not reuse the old proprietary `version.dll` mod; this HIP wrapper selects Wine's builtin version library.

Logs: `DLSS5-AMD/logs/native-hip-live.txt` and `.dlssnr-linux/logs/`. Installation success means files were copied and verified, not that rendering works in your game. Stop testing if the game freezes or produces invalid output.

Uninstall: `./install.sh uninstall --exe /path/to/Game.exe --yes`, then remove the wrapper from the launcher. Managed original files are restored from backups. Preserve backups if integrity checks report changed files.

## Development and credits

[Build from source](docs/BUILD.md) · [HIP backend / offline inference](hip/README.md) · [Third-party notices](linux/THIRD-PARTY.md)

Linux HIP implementation, bridge and packaging: **guentra and AI collaborators**. Third-party components and upstream references remain covered by [Third-party notices](linux/THIRD-PARTY.md).

## Legal

Not affiliated with NVIDIA or AMD. No NVIDIA DLL or weights are distributed.
Use a copy of `nvngx_dlssnr.dll` you obtained legitimately, converted with the project's weight-conversion scripts, then packaged. Provided as-is.

Project code: [MIT](LICENSE). Modified vkd3d-proton: LGPL-2.1-or-later, with corresponding source provided alongside the binaries. Third-party components retain their own licenses.
