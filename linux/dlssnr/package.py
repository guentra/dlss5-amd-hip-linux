"""Validate a user package (ReShade addon + lab folder + optional D3D12 runtime)."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import stat

MAX_FILES = 8000
MAX_FILE = 2 * 1024**3
REQUIRED_FLAGS = 'DLSS5-AMD/native-game-flags.txt'
ADDON = 'dlss5-amd.addon64'
LOADER_GAME = 'd3d12.dll'
LOADER_MAGPIE = 'dxgi.dll'


class DerivedLayoutsConsentRequired(RuntimeError):
    """The weights use the experimental reconstructed layout and the user has not
    consented yet. An interactive caller may offer to reuse them instead of failing;
    a non-interactive caller must pass --allow-derived-layouts explicitly."""


def _safe_file(path: Path) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f'Missing or unsafe file: {path}')
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f'Not a regular file: {path}')
    if info.st_size > MAX_FILE:
        raise RuntimeError(f'File too large: {path}')
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _safe_file(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _rel(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    if relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError(f'Refusing path outside package: {path}')
    return str(relative).replace('\\', '/')


def iter_files(root: Path):
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f'Package must be a real directory: {root}')
    count = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        if current.is_symlink():
            raise RuntimeError(f'Symlink directory in package: {current}')
        dirnames[:] = [name for name in dirnames if name not in {'.git', '__pycache__'}]
        for name in filenames:
            path = current / name
            if path.is_symlink():
                raise RuntimeError(f'Symlink in package: {path}')
            count += 1
            if count > MAX_FILES:
                raise RuntimeError('Package has too many files')
            yield path


def local_weights(exe: Path) -> Path | None:
    """Only executable-adjacent assets and an established Unreal game root."""
    directory = Path(exe).parent
    roots = [directory]
    if directory.name.casefold() == 'win64' and directory.parent.name.casefold() == 'binaries':
        roots.append(directory.parents[2])
    for root in roots:
        assets = root / 'DLSS5-AMD/native-game-tiled-assets'
        if assets.is_dir() and not assets.is_symlink():
            if any(p.suffix in {'.f32', '.f16', '.i32'} for p in assets.iterdir()):
                return assets
        dll = root / 'nvngx_dlssnr.dll'
        if dll.is_file() and not dll.is_symlink():
            return dll
    return None


def inspect_weights(path: Path, *, allow_derived_layouts=False) -> dict:
    """Read-only discovery: never invoke a converter or create a cache."""
    root = Path(path).expanduser().absolute()
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise RuntimeError(f'Unsafe symlink in weights path: {parent}')
    if root.is_file() and root.name.lower() == 'nvngx_dlssnr.dll':
        from .convert_dll import KNOWN_NVIDIA_SHA
        if sha256(root) != KNOWN_NVIDIA_SHA:
            raise RuntimeError('unrecognized nvngx_dlssnr.dll; pinned SHA256 required')
        return {'root': str(root), 'conversion_required': True,
                'runtime_ready': False, 'layout_mode': 'amd-consumer-derived'}
    weights = find_weights(root, allow_derived_layouts=allow_derived_layouts)
    manifest_path = weights / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return {'root': str(root), 'weights_dir': str(weights),
            'conversion_required': False, 'runtime_ready': manifest.get('runtime_ready', False),
            'layout_mode': manifest.get('layout_mode', 'external-tables')}


def find_weights(path: Path, *, allow_derived_layouts=False, progress=None) -> Path:
    """Directory that holds .f32/.f16/.i32 network tables (package, assets, or NVIDIA DLL)."""
    root = Path(path).expanduser().absolute()
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise RuntimeError(f'Unsafe symlink in weights path: {parent}')
    if root.is_file() and root.name.lower() == 'nvngx_dlssnr.dll':
        if not allow_derived_layouts:
            raise DerivedLayoutsConsentRequired('Conversion requires --allow-derived-layouts (experimental, not NVIDIA equivalence)')
        from .convert_dll import convert_nvidia_dll
        return convert_nvidia_dll(root, layout_mode='amd-consumer-derived', progress=progress)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(
            'Pass a weights directory, an dlss5 package folder, or nvngx_dlssnr.dll '
            '(version 310.8.0.0), not a random file.'
        )
    for candidate in (
        root / 'DLSS5-AMD' / 'native-game-tiled-assets',
        root / 'native-game-tiled-assets',
        root,
    ):
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        count = 0
        for entry in candidate.iterdir():
            if entry.is_symlink() or not entry.is_file():
                continue
            if entry.suffix.lower() in {'.f32', '.f16', '.i32'}:
                count += 1
        if count:
            manifest_path = candidate / 'manifest.json'
            if manifest_path.exists() or manifest_path.is_symlink():
                try:
                    manifest = json.loads(_safe_file(manifest_path).read_text())
                    mode = manifest.get('layout_mode')
                    if mode == 'amd-consumer-derived' and not allow_derived_layouts:
                        raise DerivedLayoutsConsentRequired('Reconstructed cache requires --allow-derived-layouts')
                    from .convert_dll import validate_cache
                    # The converter validates an exact expected filename set
                    # before reading entries; never dereference manifest paths.
                    if (candidate / 'full-network.ok').is_symlink():
                        raise RuntimeError('Unsafe weights manifest marker symlink')
                    validate_cache(candidate, layout_mode=mode)
                except (ValueError, TypeError, AttributeError) as exc:
                    raise RuntimeError(f'Invalid weights manifest: {exc}') from exc
            return candidate
    raise RuntimeError(
        'No network weights (.f32/.f16/.i32) found. Pass an dlss5 package or '
        'DLSS5-AMD/native-game-tiled-assets with --package / --weights.'
    )


def validate(package: Path, *, magpie: bool = False) -> dict:
    """Return a description of a drop-in dlss5 release directory."""
    root = Path(package).expanduser().resolve()
    files = list(iter_files(root))
    names = {_rel(root, path) for path in files}
    loader = LOADER_MAGPIE if magpie else LOADER_GAME
    missing = [name for name in (ADDON, loader, REQUIRED_FLAGS) if name not in names]
    if missing:
        raise RuntimeError(
            'Not a valid user package. Missing: '
            + ', '.join(missing)
            + f'. Pass the extracted folder that contains {ADDON} next to DLSS5-AMD/.'
        )
    assets = [name for name in names if name.startswith('DLSS5-AMD/native-game-tiled-assets/')]
    weights = [name for name in assets if name.endswith(('.f16', '.f32', '.i32'))]
    shaders = [name for name in assets if name.endswith(('.cso', '.hlsl', '.hlsli'))]
    if not weights:
        raise RuntimeError('Package has no network weights under DLSS5-AMD/native-game-tiled-assets/.')
    agility = [name for name in names if name.startswith('DLSS5-D3D12-721/') and name.lower().endswith('d3d12core.dll')]
    addon = root / ADDON
    if addon.stat().st_size < 64 * 1024:
        raise RuntimeError(f'{ADDON} is too small to be the ReShade add-on')
    return {
        'root': str(root),
        'mode': 'magpie' if magpie else 'game',
        'addon': ADDON,
        'loader': loader,
        'files': sorted(names),
        'file_count': len(names),
        'weights': len(weights),
        'shaders': len(shaders),
        'agility_sdk': bool(agility),
        'addon_sha256': sha256(addon),
        'flags_sha256': sha256(root / REQUIRED_FLAGS),
    }
