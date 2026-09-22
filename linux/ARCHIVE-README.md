# DLSS5-AMD HIP — Linux proof of concept

Prebuilt native HIP network, Wine bridge and ReShade add-on for `gfx1200`
(RX 9060) and `gfx1201` (RX 9070). No HIP or MinGW compiler is needed to use
this archive.

**This is a slow proof of concept requiring substantial optimization, not a
production-ready gaming mod.** The live hook uses the included modified vkd3d
submission boundary, CPU readback/upload and HIP; the game still waits for the
network. Motion history is reset each frame. General gameplay stability, HDR and
NVIDIA numerical/visual equivalence are not verified. No frame generation.

## Requirements

- Linux x86_64, Python 3.10+, AMD RDNA 4 `gfx1201`, ROCm/HIP 7.
- Your own `nvngx_dlssnr.dll` 310.8.0.0, or complete coefficient tables.
- NumPy and Pillow for the image command (`requirements.txt`).
- For optional game-directory staging: the game's existing compatible Proton/Wine
  runner and a supported DX12 upscaler hook. No anti-cheat games.

Weights are not bundled. Some layouts are reconstructed from AMD consumer evidence;
`--allow-derived-layouts` explicitly accepts them, not NVIDIA equivalence.

## Install

Close the game, extract this archive inside its directory, keep this subfolder,
and run `./install.sh` (not sudo). The wizard selects the executable, existing
runner and weights and asks for experimental-layout consent. Put your own
`nvngx_dlssnr.dll` beside the executable or in the game root for automatic
detection. Use the printed wrapper as your launch command prefix; Steam is optional.

The installer deploys the matching vkd3d pair and configures ReShade's proxy.
F6 bypasses/re-enables the live HIP path if the FSR hook is active. Expect very
low frame rates. Magpie and anti-cheat games are not supported by this HIP release.

Optional arguments: `./install.sh install --help`. `doctor` and `--dry-run` do
not convert or install files. For ambiguous detection, use
`--exe '/path/to/Game/Binaries/Win64/Game.exe'`.

Uninstall with `./install.sh uninstall --exe /path/to/Game.exe --yes`, then remove
the wrapper from the launcher. Existing files are restored from managed backups.

## Optional offline image inference

From this extracted folder, using a 1920×1080 input image:

```sh
python3 infer_image.py input.png output.png \
  --nvidia-dll /absolute/path/to/nvngx_dlssnr.dll \
  --allow-derived-layouts --runs 2
```

Use `--weights /absolute/path/to/complete/tables` instead of `--nvidia-dll` to reuse
a cache. Output is never overwritten. `output.png.json` records actual timing,
finiteness, replay and SHA256 values. `bin/hip-network70 --help` describes the
packed-float benchmark. `HIP.md` contains source-checkout build/test details.


## Contents and notices

`bin/` contains `libdlss5_hip.so`, `dlss5_hip.dll`, `hip-network70`,
`dlss5-amd.addon64`, the ReShade 6.8 add-on loader, `dlss5-d3d12.dll` and
`d3d12core.dll`. The native library
requires ROCm installed separately. `manifest.json` records every packaged file's
digest; it is an integrity inventory, not a test certificate.

Code license: `LICENSE`. ReShade and MinHook notices: `licenses/` and
`THIRD-PARTY.md`. No NVIDIA DLL, weights, or proprietary AMD inference binaries are included. Corresponding source (including modified LGPL
vkd3d-proton) is available with this release:
https://github.com/guentra/dlss5-amd-hip-linux/releases/tag/v0.1.0-poc
