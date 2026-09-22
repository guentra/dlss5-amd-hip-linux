# DLSS5-AMD HIP on Linux

The complete native HIP network is available for **offline inference**; see
[`hip/README.md`](../hip/README.md) for the image command and tests.
This folder stages the ReShade add-on and Wine bridge locally. It does not use
danielblnc's `setup.exe`, inference DLLs or fatbins.

**Slow proof of concept: substantial optimization is required.** The packaged
modified vkd3d pair executes HIP at a split submission boundary. The game still
waits for inference and motion history resets each frame. Installation and
finite pixels do not prove gameplay stability or NVIDIA equivalence.

## Binary release

With the game closed, extract `dlss5-amd-hip-linux.tar.gz` inside its directory,
keep the extracted subfolder, and run `./install.sh` there. The wizard asks for
the executable, existing runner and weights. Steam is optional.

Requirements: Linux x86_64, Python 3.10+, RDNA 4 `gfx1200` (RX 9060) or
`gfx1201` (RX 9070), ROCm/HIP 7 and a
compatible Wine/Proton runner. `gfx1200` is untested and may simply not work. The native ROCm runtime is not included or installed
automatically. No anti-cheat games. The archive includes the ReShade 6.8 add-on
loader, native `.so` and PE bridge; compilers are not required to use it.

The converter accepts your SHA-pinned `nvngx_dlssnr.dll` 310.8.0.0. Complete
reconstructed caches require explicit `--allow-derived-layouts` consent; original
NVIDIA maps and visual equivalence are not claimed. A complete external coefficient
folder may also be supplied. `doctor` and `--dry-run` never convert or install.

Optional paths and arguments: `./install.sh install --help`. If automatic
executable selection is ambiguous, use `--exe '/path/to/Game/Binaries/Win64/Game.exe'`.
The wizard installs the matching runtime pair and ReShade proxy configuration.
F6 toggles the active live hook's bypass. Do not use the legacy activation files
to work around a missing hook or a freeze. This release does not support Magpie.

## Build from source

From the repository root. See [`../docs/BUILD.md`](../docs/BUILD.md) for HIP
toolchain discovery, the vkd3d source archive, ReShade placement and packager
layout.

```sh
make -C hip -j3 game hip-network70
linux/install.sh build-addon
python3 linux/build_release.py
```

Artifacts are written to `dist/`. The archive has a per-file SHA256
manifest, offline image CLI and benchmark; no weights or project tests are packaged.

## Removal and logs

`./install.sh uninstall --exe /path/to/Game.exe --yes` restores managed backups.
Remove the printed wrapper from the launcher afterward. The diagnostic add-on
logs to `DLSS5-AMD/logs/native-hip-live.txt`; native API errors are also
available through `dlss5_last_error()`.

## Legal

Not affiliated with NVIDIA or AMD. No NVIDIA DLL or weights are distributed.
Use a copy of `nvngx_dlssnr.dll` you obtained legitimately, converted with the
project weight-conversion scripts, then packaged. Provided as-is.
