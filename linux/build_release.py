#!/usr/bin/env python3
"""Pack a standalone Linux HIP installer tarball (prebuilt binaries, no compiler)."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARCHIVE = 'dlss5-amd-hip-linux'
sys.path.insert(0, str(HERE))

from dlssnr import VERSION as PACKAGE_VERSION


def _add(tar: tarfile.TarFile, src: Path, dest: str, mode: int | None = None) -> dict:
    if src.is_symlink() or not src.is_file():
        raise RuntimeError(f'Not a regular build artifact: {src}')
    data = src.read_bytes()
    info = tar.gettarinfo(str(src), arcname=dest)
    info.size = len(data)
    info.uid = info.gid = 0
    info.uname = info.gname = 'root'
    if mode is not None:
        info.mode = mode
    elif src.suffix == '.so' or src.name.endswith('.addon64') or src.name == 'install.sh':
        info.mode = 0o755
    else:
        info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def _add_bytes(tar: tarfile.TarFile, dest: str, data: bytes, mode: int = 0o644) -> dict:
    info = tarfile.TarInfo(dest)
    info.size = len(data)
    info.mode = mode
    info.uid = info.gid = 0
    info.uname = info.gname = 'root'
    tar.addfile(info, io.BytesIO(data))
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def main() -> None:
    so = REPO / 'hip' / 'libdlss5_hip.so'
    dll = REPO / 'hip' / 'dlss5_hip.dll'
    addon = HERE / 'build' / 'dlss5-amd.addon64'
    reshade = HERE / 'vendor' / 'reshade' / 'ReShade64.dll'
    reshade_lic = HERE / 'vendor' / 'reshade' / 'LICENSE.md'
    reshade_prov = HERE / 'vendor' / 'reshade' / 'PROVENANCE.txt'
    bench = REPO / 'hip' / 'hip-network70'
    minhook_lic = REPO / 'third_party' / 'minhook' / 'LICENSE.txt'
    vkd3d = HERE / 'build/live/build-vkd3d/libs/d3d12/d3d12.dll'
    vkd3dcore = HERE / 'build/live/build-vkd3d/libs/d3d12core/d3d12core.dll'
    vkd3d_source = HERE / 'build/live/vkd3d-proton'
    missing = [str(p) for p in (so, dll, addon, reshade, bench, reshade_lic, reshade_prov, minhook_lic,
                               vkd3d, vkd3dcore, *(vkd3d_source / n for n in ('LICENSE', 'COPYING', 'AUTHORS'))) if not p.is_file()]
    if missing:
        raise SystemExit('Build artifacts missing: ' + ', '.join(missing) + '\nRun: make -C hip game hip-network70 && linux/install.sh build-addon')

    dest_dir = REPO / 'dist'
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / (ARCHIVE + '.tar.gz')

    py_files = [
        'installer.py',
        'install.sh',
        'THIRD-PARTY.md',
        'dlssnr/__init__.py',
        'dlssnr/cli.py',
        'dlssnr/deploy.py',
        'dlssnr/games.py',
        'dlssnr/package.py',
        'dlssnr/addon.py',
        'dlssnr/convert_dll.py',
        'dlssnr/runtime.py',
        'dlssnr/assets.py',
        'dlssnr/kernels.py',
    ]
    readme = (HERE / 'ARCHIVE-README.md').read_bytes() if (HERE / 'ARCHIVE-README.md').is_file() else (HERE / 'README.md').read_bytes()
    flags = (REPO / 'scripts' / 'game-flags.txt').read_text()
    if 'DLSS5_HIP=' not in flags:
        flags += '\nDLSS5_HIP=1\nDLSS5_SNAPSHOT_FRAME=60\n'
    license_text = (REPO / 'LICENSE').read_bytes()

    tmp = dest_dir / ('.' + ARCHIVE + '.tmp')
    try:
        with tarfile.open(tmp, 'w:gz') as tar:
            records = {}

            def add_file(src, name, mode=0o644):
                records[name] = _add(tar, src, f'{ARCHIVE}/{name}', mode)

            def add_bytes(name, data):
                records[name] = _add_bytes(tar, f'{ARCHIVE}/{name}', data)

            add_bytes('README.md', readme)
            add_bytes('LICENSE', license_text)
            add_bytes('flags/native-game-flags.txt', flags.encode())
            for name in py_files:
                add_file(HERE / name, name, 0o755 if name == 'install.sh' else 0o644)
            add_file(so, 'bin/libdlss5_hip.so', 0o755)
            add_file(dll, 'bin/dlss5_hip.dll')
            add_file(addon, 'bin/dlss5-amd.addon64')
            add_file(reshade, 'bin/ReShade64.dll')
            add_file(vkd3d, 'bin/dlss5-d3d12.dll')
            add_file(vkd3dcore, 'bin/d3d12core.dll')
            for name in ('LICENSE', 'COPYING', 'AUTHORS'):
                add_file(vkd3d_source / name, 'licenses/vkd3d-' + name)
            add_file(bench, 'bin/hip-network70', 0o755)
            add_file(REPO / 'hip/infer_image.py', 'infer_image.py')
            add_file(REPO / 'hip/requirements.txt', 'requirements.txt')
            add_file(REPO / 'hip/README.md', 'HIP.md')
            add_file(REPO / 'hip/include/dlss5_capi.h', 'include/dlss5_capi.h')
            add_file(reshade_lic, 'licenses/ReShade-LICENSE.md')
            add_file(reshade_prov, 'licenses/ReShade-PROVENANCE.txt')
            add_file(minhook_lic, 'licenses/MinHook-LICENSE.txt')
            for notice in sorted((HERE / 'vendor' / 'notices').glob('*')):
                add_file(notice, 'licenses/' + notice.name)
            manifest = {'schema': 1, 'version': PACKAGE_VERSION, 'minimum_glibc': '2.34',
                        'purpose': 'experimental native HIP DLSS5 network; per-frame optimization ongoing',
                        'source_url': 'https://github.com/guentra/dlss5-amd-hip-linux/tree/main',
                        'gameplay_verified': False, 'nvidia_equivalence_verified': False,
                        'files': records}
            _add_bytes(tar, f'{ARCHIVE}/manifest.json',
                       (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
        tmp.replace(archive)
    finally:
        if tmp.exists():
            tmp.unlink()

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (dest_dir / (archive.name + '.sha256')).write_text(f'{digest}  {archive.name}\n')
    print(archive)
    print(digest, archive.stat().st_size)


if __name__ == '__main__':
    main()
