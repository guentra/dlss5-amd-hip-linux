"""Copy an dlss5 user package into a game directory and write a Proton wrapper."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import configparser
import io
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile

from . import package

STORE = '.dlssnr-linux'
SCHEMA = 2
# Files the game may rewrite between install and the next upgrade/uninstall
# (ReShade re-saves its own settings on exit); a changed copy must not block
# a reinstall, and install regenerates them from the proxy template anyway.
RUNTIME_MUTABLE = {'ReShade.ini'}


class ChangedDeploymentError(RuntimeError):
    """A previous deployment differs from its journal; overwriting needs consent."""


NOTES = [
    'D3D path: this deploys the ReShade add-on and package shaders without native HIP.',
    'SM 6.10 wave-matrix requires the corresponding D3D12 runtime and driver support; successful installation does not verify rendering under vkd3d-proton.',
    'Remove the wrapper from your launcher after uninstall.',
]
HIP_NOTES = [
    'HIP proof of concept: slow, experimental and in need of substantial optimization; not ready for normal gameplay.',
    'The modified vkd3d submission worker waits for HIP. Live history resets each frame; gameplay and NVIDIA equivalence remain unverified.',
    'Weights stay in DLSS5-AMD/native-game-tiled-assets/. Remove the wrapper from your launcher after uninstall.',
]


def bridge_cache_path(digest):
    """Shared LD_PRELOAD target for the HIP bridge library.

    The ELF loader splits LD_PRELOAD on whitespace, so the preloaded library
    can never live under a path containing spaces. Game install directories
    often do (for example 'SILENT HILL f'), so the preload target is a
    per-digest copy under the user data directory, shared between games and
    retained on uninstall.
    """
    root = Path(os.environ.get('XDG_DATA_HOME') or (Path.home() / '.local' / 'share'))
    path = root / 'dlss5-hip' / 'bridges' / (digest + '.so')
    if any(ch.isspace() for ch in str(path)) or ':' in str(path):
        raise RuntimeError(
            'Bridge cache path must not contain whitespace or colon (LD_PRELOAD '
            'limit): ' + str(path) + '; set XDG_DATA_HOME to a safe path and reinstall')
    return path


def _safe(path, *, directory=False, missing=False):
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if missing:
                return path
            raise RuntimeError(f'Missing path: {current}') from None
        wanted_dir = current != path or directory
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) if wanted_dir else stat.S_ISREG(info.st_mode)):
            raise RuntimeError(f'Unsafe symlink or wrong path type: {current}')
    return path


def _exe(exe):
    path = _safe(exe)
    if path.suffix.lower() != '.exe':
        raise RuntimeError('Expected a game .exe')
    return path.parent.resolve() / path.name


def _digest(path):
    return package.sha256(_safe(path))


def _sync(path):
    fd = os.open(_safe(path, directory=True), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _lock(directory):
    fd = os.open(_safe(directory, directory=True), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another deployment transaction is running') from None
        yield
    finally:
        os.close(fd)


def running_game(exe):
    basename = Path(exe).name.casefold()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode('utf-8', 'surrogateescape').rstrip('\0').split('\0')
        except (OSError, ValueError):
            continue
        if args and Path(args[0].replace('\\', '/')).name.casefold() == basename:
            return True
        if len(args) > 1 and Path(args[0]).name.lower().startswith('wine') and Path(args[1].replace('\\', '/')).name.casefold() == basename:
            return True
    return False


def wrapper_bytes(exe, *, magpie=False, gpu_name=None, hip=False, hip_so=None,
                  gpu=None, so_digest=None, hip_library=None, hip_library_digest=None):
    store = exe.parent / STORE
    overrides = 'dxgi=n,b;d3d12=n,b;d3d12core=n,b' if magpie else 'd3d12=n,b;d3d12core=n,b'
    if hip:
        overrides = 'version=b;dlss5_hip=n;' + overrides
    q = shlex.quote
    weights = exe.parent / 'DLSS5-AMD' / 'native-game-tiled-assets'
    so = hip_so or (store / 'lib' / 'libdlss5_hip.so')
    lines = [
        '#!/bin/bash',
        'set -euo pipefail',
        '# Generated Proton prefix for dlss5-amd-hip (' + ('HIP diagnostic' if hip else 'D3D backend') + ').',
        'if (( $# == 0 )); then',
        '  printf "%s\\n" "Usage: launch.sh RUNNER [ARGUMENTS...] (Steam: launch.sh %command%)." >&2',
        '  exit 2',
        'fi',
        'addon=' + q(str(exe.parent / package.ADDON)),
        'if [[ ! -f "$addon" ]]; then',
        '  printf "DLSS5: missing add-on: %s\\n" "$addon" >&2',
        '  exit 1',
        'fi',
        'kept=()',
        'IFS=";" read -r -a entries <<< "${WINEDLLOVERRIDES-}"',
        'for entry in "${entries[@]}"; do',
        '  [[ "$entry" == *=* ]] || continue',
        '  names=${entry%%=*}; value=${entry#*=}',
        '  IFS="," read -r -a names_array <<< "$names"',
        '  for name in "${names_array[@]}"; do',
        '    clean=${name//[[:space:]]/}; clean=${clean,,}; clean=${clean#\\*}; clean=${clean%.dll}',
        '    case "$clean" in d3d12|d3d12core|dxgi|dlss5_hip' + ('|version' if hip else '') + ') ;;',
        '      *) kept+=("$name=$value") ;;',
        '    esac',
        '  done',
        'done',
        'saved=$(IFS=";"; printf "%s" "${kept[*]}")',
        f'export WINEDLLOVERRIDES="${{saved:+$saved;}}{overrides}"',
        'export STEAM_COMPAT_MOUNTS="${STEAM_COMPAT_MOUNTS:+$STEAM_COMPAT_MOUNTS:}"' + q(str(exe.parent)),
    ]
    if hip:
        lines += ['so=' + q(str(so))]
        if so_digest:
            lines += [
                'if [[ ! -f "$so" || ! -r "$so" ]] || ! actual=$(sha256sum < "$so") || '
                '[[ "${actual%% *}" != ' + q(so_digest) + ' ]]; then',
                '  printf "DLSS5: HIP library missing or changed: %s; restore it or reinstall before launching.\\n" "$so" >&2',
                '  exit 1',
                'fi',
            ]
        if hip_library:
            library = str(hip_library)
            lines += ['hip_library=' + q(library)]
            if hip_library_digest:
                lines += [
                    'if [[ ! -f "$hip_library" || ! -r "$hip_library" ]] || ! actual=$(sha256sum < "$hip_library") || '
                    '[[ "${actual%% *}" != ' + q(hip_library_digest) + ' ]]; then',
                    '  printf "DLSS5: HIP runtime missing or changed: %s; restore it or reinstall before launching.\\n" "$hip_library" >&2',
                    '  exit 1',
                    'fi',
                    'export LD_LIBRARY_PATH=' + q(str(Path(library).parent)) + '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
                ]
        lines += [
            'export LD_PRELOAD="$so${LD_PRELOAD:+:$LD_PRELOAD}"',
            'export DLSS5_HIP=1',
            'export DLSS5_HIP_WEIGHTS=' + q(str(weights)),
            # Temporal blend weight (previous-frame share) for the non-temporal live
            # path; smooths the per-frame shimmer. 0 disables it (legacy behavior).
            'export DLSS5_TEMPORAL_BLEND="${DLSS5_TEMPORAL_BLEND:-0.5}"',
            # MH prod chain: byte-exact fused path for the C64/C128/C256 blocks.
            # ~26% faster network (67->50 ms) with identical output; 0 to disable.
            'export DLSS5_MH_PROD="${DLSS5_MH_PROD:-1}"',
        ]
    if gpu is not None:
        gpu_name = gpu.get('name') or gpu_name
    if gpu_name:
        lines += [
            'export DXVK_FILTER_DEVICE_NAME=' + q(gpu_name),
            'export VKD3D_FILTER_DEVICE_NAME=' + q(gpu_name),
        ]
    lines += [
        'export VKD3D_DEBUG="${VKD3D_DEBUG:-info}"',
        'export VKD3D_LOG_FILE=' + q('Z:' + str(store / 'logs/vkd3d.log')),
        'exec "$@"',
        '',
    ]
    return '\n'.join(lines).encode()


def _atomic_copy(source, target, expected=None, mode=None):
    target = _safe(target, missing=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.copy-', dir=str(target.parent))
    temporary = Path(temporary)
    try:
        with _safe(source).open('rb') as src, os.fdopen(fd, 'wb') as dst:
            fd = -1
            digest = hashlib.sha256()
            for block in iter(lambda: src.read(1024 * 1024), b''):
                digest.update(block)
                dst.write(block)
            if expected is not None and digest.hexdigest() != expected:
                raise RuntimeError(f'Hash changed during copy: {source}')
            os.fchmod(dst.fileno(), mode if mode is not None else 0o644)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, target)
        _sync(target.parent)
    finally:
        if fd != -1:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()


def _atomic_bytes(target, data, mode=0o600):
    target = _safe(target, missing=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.write-', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            os.fchmod(stream.fileno(), mode)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
        _sync(target.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _journal(store, data):
    _atomic_bytes(store / 'manifest.json', json.dumps(data, sort_keys=True, indent=2).encode() + b'\n')


def _load(exe):
    store = _safe(exe.parent / STORE, directory=True)
    data = json.loads(_safe(store / 'manifest.json').read_text())
    if type(data) is not dict:
        raise RuntimeError('Not an dlss5 Linux deployment journal; refusing mutation')
    # A terminal tombstone owns no files, so a fresh install may reset it even
    # when an older release wrote it under a previous journal kind.
    if (data.get('state') == 'removed' and not data.get('files')
            and data.get('exe') in (None, str(exe))):
        return data
    if data.get('schema') != SCHEMA or data.get('kind') != 'dlss5':
        raise RuntimeError('Not an dlss5 Linux deployment journal; refusing mutation')
    if data.get('exe') != str(exe):
        raise RuntimeError('Deployment journal exe mismatch')
    if type(data.get('files')) is not dict:
        raise RuntimeError('Invalid deployment journal files')
    if data.get('state') == 'removed' and data['files']:
        raise RuntimeError('Removed deployment journal must not own files')
    for name, item in data['files'].items():
        parts = name.split('/')
        if (not name or name.startswith('/') or any(p in {'', '.', '..'} for p in parts)
                or '\\' in name or ':' in name or name == STORE + '/manifest.json'
                or name.startswith(STORE + '/backups/')):
            raise RuntimeError(f'Invalid deployment journal path: {name}')
        if (type(item) is not dict or not isinstance(item.get('sha256'), str)
                or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise RuntimeError(f'Invalid deployment journal digest: {name}')
        original = item.get('original')
        if original is not None and (not isinstance(original, str) or len(original) != 64
                                     or any(c not in '0123456789abcdef' for c in original)):
            raise RuntimeError(f'Invalid deployment original digest: {name}')
    return data


FOREIGN_STORE_KEYS = frozenset({'schema', 'exe', 'state', 'files', 'request', 'cache', 'undo'})


def foreign_deployment(exe):
    """Detect a live deployment owned by the foreign dlssnr_on_amd product.

    Its journal is schema 1 with a distinct key set; we never mutate it
    ourselves, only remove it through its own staged uninstaller.
    """
    exe = _exe(exe)
    store = exe.parent / STORE
    manifest = store / 'manifest.json'
    if not store.is_dir() or manifest.is_symlink() or not manifest.is_file():
        return None
    try:
        data = json.loads(_safe(manifest).read_text())
    except (OSError, ValueError):
        return None
    if type(data) is not dict or set(data) != FOREIGN_STORE_KEYS or data.get('schema') != 1:
        return None
    if data.get('exe') != str(exe) or type(data.get('files')) is not dict or not data['files']:
        return None
    if data.get('state') not in {'installed', 'installing', 'uninstalling'}:
        return None
    for name, item in data['files'].items():
        parts = name.split('/')
        if (type(name) is not str or not name or name.startswith('/')
                or any(p in {'', '.', '..'} for p in parts) or '\\' in name or ':' in name
                or type(item) is not dict or 'sha256' not in item or 'original' not in item
                or 'touched' not in item or 'preserve' not in item or 'mode' not in item):
            return None
    return data


def remove_foreign_deployment(exe, *, timeout=300):
    """Remove the foreign deployment by running its own staged uninstaller.

    Must be called without holding the game directory lock: the foreign
    uninstaller takes the same directory flock.
    """
    exe = _exe(exe)
    data = foreign_deployment(exe)
    if data is None:
        raise RuntimeError('No foreign deployment found for this game.')
    staging = exe.parent / 'dlssnr-linux-portable'
    if staging.is_symlink() or not staging.is_dir():
        raise RuntimeError('The foreign installer directory is missing or incomplete; run its '
                           'own uninstall (dlssnr-linux-portable/install.sh uninstall) manually.')
    _safe(staging, directory=True)
    for required in ('installer.py', 'dlssnr', 'PROVENANCE.json'):
        path = staging / required
        present = path.is_dir() if required == 'dlssnr' else path.is_file()
        if path.is_symlink() or not present:
            raise RuntimeError('The foreign installer directory is missing or incomplete; run its '
                               'own uninstall (dlssnr-linux-portable/install.sh uninstall) manually.')
        _safe(path, directory=required == 'dlssnr')
    try:
        result = subprocess.run([sys.executable, '-B', str(staging / 'installer.py'),
                                 'uninstall', '--exe', str(exe), '--yes', '--json'],
                                capture_output=True, text=True, timeout=timeout,
                                cwd=str(staging), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Foreign uninstaller timed out; finish it manually '
                           '(dlssnr-linux-portable/install.sh uninstall) and re-run.') from exc
    if result.returncode != 0:
        raise RuntimeError('Foreign deployment removal failed; finish it with the original '
                           'uninstaller (dlssnr-linux-portable/install.sh uninstall):\n'
                           + (result.stdout + result.stderr)[-2000:])
    manifest = exe.parent / STORE / 'manifest.json'
    if manifest.is_file() and not manifest.is_symlink():
        try:
            after = json.loads(manifest.read_text())
        except ValueError:
            after = {}
        if after.get('state') != 'removed' or after.get('files'):
            raise RuntimeError('Foreign removal left a live journal; inspect '
                               '.dlssnr-linux/manifest.json before continuing.')
    for name, item in data['files'].items():
        if item.get('preserve'):
            continue
        target = exe.parent / name
        if target.exists() or target.is_symlink():
            raise RuntimeError(f'Foreign removal incomplete: {name} is still present; '
                               'inspect the game directory before continuing.')
    return data


def _previous_or_foreign(exe, store, *, dry_run):
    if not store.exists():
        return {}
    if foreign_deployment(exe) is not None:
        if not dry_run:
            raise RuntimeError('A foreign dlssnr_on_amd deployment owns this game directory; '
                               're-run with --replace-foreign to remove it via its own uninstaller.')
        return {}
    return _load(exe)


def prefix_paths(exe):
    store = exe.parent / STORE
    launch = store / 'launch.sh'
    return {
        'command_prefix': shlex.quote(str(launch)),
        'launch_options': shlex.quote(str(launch)) + ' %command%',
    }


def status_game(exe):
    exe = _exe(exe)
    store = exe.parent / STORE
    if not store.exists():
        return {'installed': False, 'valid': False, 'pending': False, 'notes': []}
    foreign = foreign_deployment(exe)
    if foreign is not None:
        return {'installed': False, 'valid': False, 'pending': False, 'foreign': True,
                'notes': ['Foreign dlssnr_on_amd deployment present (not managed by this installer); '
                          'use install --replace-foreign to remove it via its own uninstaller.']}
    try:
        data = _load(exe)
        if data.get('state') == 'removed':
            return {'installed': False, 'valid': False, 'pending': False, 'notes': []}
        if data.get('state') != 'installed':
            return {'installed': True, 'valid': False, 'pending': True,
                    'notes': ['Incomplete deployment journal; preserve backups and recover before launch.']}
        missing = [name for name in data['files'] if not (exe.parent / name).is_file()]
        if missing:
            return {'installed': True, 'valid': False, 'pending': True,
                    'notes': ['Missing deployed files: ' + ', '.join(missing[:8])]}
        changed = [name for name, item in data['files'].items()
                   if name not in RUNTIME_MUTABLE and _digest(exe.parent / name) != item['sha256']]
        if changed:
            return {'installed': True, 'valid': False, 'pending': True, 'changed': changed,
                    'notes': ['Changed deployed files: ' + ', '.join(changed[:8])]}
        if data.get('bridge_cache'):
            cache = Path(data['bridge_cache'])
            if not (cache.is_file() and _digest(cache) == data.get('bridge_cache_sha256')):
                return {'installed': True, 'valid': False, 'pending': True,
                        'notes': ['Bridge cache missing or changed: ' + str(cache)
                                  + '; reinstall before launching']}
        return dict({'installed': True, 'valid': True, 'pending': False,
                     'hip': data.get('hip', False),
                     'notes': list(HIP_NOTES if data.get('hip', False) else NOTES),
                     'mode': data.get('mode'), 'package': data.get('package'),
                     'weights': data.get('weights'), 'gpu': data.get('gpu'),
                     'hip_library': data.get('hip_library'),
                     'bridge_cache': data.get('bridge_cache')}, **prefix_paths(exe))
    except (RuntimeError, OSError, ValueError) as exc:
        return {'installed': True, 'valid': False, 'pending': True, 'notes': [str(exc)]}


def previous_deployment(exe):
    """Describe a managed deployment left by a previous operation, if any."""
    exe = _exe(exe)
    store = exe.parent / STORE
    if not store.exists():
        return {'present': False, 'state': None, 'valid': False, 'reusable': False, 'changed': []}
    try:
        data = _load(exe)
    except (RuntimeError, OSError, ValueError):
        data = {}
    state = data.get('state')
    if state in (None, 'removed'):
        return {'present': False, 'state': state, 'valid': False, 'reusable': False, 'changed': []}
    status = status_game(exe)
    return {'present': True, 'state': state, 'valid': status['valid'],
            'reusable': state == 'installed' and status['valid'],
            'changed': status.get('changed', []), 'notes': status.get('notes', [])}


def _first_file(*candidates):
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def hip_paths():
    installer = Path(__file__).resolve().parents[1]
    repo = installer.parent
    return {
        'so': _first_file(installer / 'bin' / 'libdlss5_hip.so', repo / 'hip' / 'libdlss5_hip.so'),
        'dll': _first_file(installer / 'bin' / 'dlss5_hip.dll', repo / 'hip' / 'dlss5_hip.dll'),
        'addon': _first_file(installer / 'bin' / 'dlss5-amd.addon64', installer / 'build' / 'dlss5-amd.addon64'),
        'vkd3d': _first_file(installer / 'bin' / 'dlss5-d3d12.dll', installer / 'build/live/build-vkd3d/libs/d3d12/d3d12.dll'),
        'vkd3dcore': _first_file(installer / 'bin' / 'd3d12core.dll', installer / 'build/live/build-vkd3d/libs/d3d12core/d3d12core.dll'),
        'reshade': _first_file(
            installer / 'bin' / 'ReShade64.dll',
            installer / 'vendor' / 'reshade' / 'ReShade64.dll',
            installer / 'bin' / 'd3d12.dll',
        ),
        'flags': _first_file(installer / 'flags' / 'native-game-flags.txt', repo / 'scripts' / 'game-flags.txt'),
    }


def ensure_hip_artifacts():
    paths = hip_paths()
    needed = ('so', 'dll', 'addon', 'reshade', 'vkd3d', 'vkd3dcore')
    missing = [str(paths[name]) for name in needed if not paths[name].is_file()]
    if missing:
        raise RuntimeError(
            'HIP game artifacts missing: ' + ', '.join(missing)
            + '. From the repo: make -C hip game && linux/install.sh build-addon')
    return paths


def _backup_original(exe, relative, files, store):
    target = exe.parent / relative
    _safe(target, missing=True)
    if relative in files:
        original = files[relative].get('original')
        if original is not None:
            backup = store / 'backups' / relative.replace('/', '__')
            if _digest(backup) != original:
                raise RuntimeError(f'Backup missing or changed: {relative}')
        return original
    original = _digest(target) if target.is_file() else None
    if original is not None:
        backup = store / 'backups' / relative.replace('/', '__')
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            _atomic_copy(target, backup, original)
        elif _digest(backup) != original:
            raise RuntimeError(f'Backup collision: {relative}')
    return original


def _prune_weights(exe, files, store, data, keep, *, remove_unmanaged):
    """Drop superseded network tables before staging a fresh set.

    Upgrading an existing install must not leave stale tables behind (a
    foreign game's network, a superseded revision, an old manifest): the
    runtime would mix them with the new set. Files owned by the deployment
    journal are always pruned (digest-checked first); files the journal does
    not own are only removed when the caller asked to replace existing files,
    so a manual layout is never clobbered silently.
    """
    prefix = 'DLSS5-AMD/native-game-tiled-assets/'
    assets = exe.parent / prefix
    if not assets.is_dir():
        return []
    removed = []
    for entry in sorted(assets.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            continue
        relative = prefix + entry.name
        owned = relative in files
        if entry.name in keep and owned:
            continue
        if not owned and not remove_unmanaged:
            continue
        if owned and _digest(entry) != files[relative]['sha256']:
            raise RuntimeError(f'Refusing to remove a changed weights file: {relative}; uninstall first')
        entry.unlink()
        if owned:
            del files[relative]
        removed.append(relative)
    if removed:
        data['files'] = files
        _journal(store, data)
    return removed


def _record_write(exe, relative, digest, original, files, store):
    target = exe.parent / relative
    previous = _digest(target) if target.is_file() else None
    files[relative] = {'sha256': digest, 'original': original, 'previous_sha256': previous}
    data = _load(exe)
    data['files'] = files
    _journal(store, data)  # Write-ahead record survives failure before/after rename.


def _overlay(exe, relative, source, files, store):
    target = exe.parent / relative
    original = _backup_original(exe, relative, files, store)
    digest = package.sha256(source)
    _record_write(exe, relative, digest, original, files, store)
    _atomic_copy(source, target, digest)


def _overlay_bytes(exe, relative, content, files, store, mode=0o644):
    original = _backup_original(exe, relative, files, store)
    _record_write(exe, relative, hashlib.sha256(content).hexdigest(), original, files, store)
    _atomic_bytes(exe.parent / relative, content, mode)


def _overwrite_previous(exe, store):
    """Wipe a previous managed deployment (any state, modifications included)."""
    if not store.exists():
        return
    try:
        previous = _load(exe)
    except (RuntimeError, OSError, ValueError):
        previous = {}
    if previous.get('state') not in (None, 'removed') and previous.get('files'):
        _teardown(exe, store, previous, tolerant=True)


def _begin_install(exe, *, hip, mode, force=False, **metadata):
    store = exe.parent / STORE
    files = {}
    if store.exists():
        previous = _load(exe)
        if previous.get('state') != 'removed':
            status = status_game(exe)
            if previous.get('state') != 'installed' or not status['valid']:
                if not force:
                    detail = '; '.join(status.get('notes', [])) or 'incomplete journal'
                    raise ChangedDeploymentError(
                        'Previous installation is modified or incomplete: ' + detail
                        + '. Overwrite it with --overwrite (or answer yes when prompted), or uninstall first.')
            elif previous.get('hip', False) != hip or previous.get('mode') != mode:
                raise RuntimeError('Uninstall before switching deployment backend or loader mode')
            files = dict(previous['files'])
    store.mkdir(mode=0o700, exist_ok=True)
    data = dict(schema=SCHEMA, kind='dlss5', exe=str(exe), state='installing',
                hip=hip, mode=mode, files=files, **metadata)
    _journal(store, data)
    return data


def install_hip(exe, weights_root, *, magpie=False, replace_existing=False,
                acknowledge_risk=False, dry_run=False, gpu_name=None, allow_derived_layouts=False,
                gpu=None, hip_library=None, force=False, progress=None):
    if not acknowledge_risk:
        raise RuntimeError('Explicit --accept-risk is required; this injects a ReShade add-on into the game directory')
    exe = _exe(exe)
    if gpu is not None:
        if type(gpu.get('index')) is not int or gpu['index'] < 0 or not isinstance(gpu.get('name'), str) or not gpu['name']:
            raise RuntimeError('Select an explicit GPU index and name')
    weights = (Path(package.inspect_weights(weights_root, allow_derived_layouts=allow_derived_layouts).get('weights_dir', weights_root))
               if dry_run else package.find_weights(weights_root, allow_derived_layouts=allow_derived_layouts,
                                                    progress=progress))
    hip_files = ensure_hip_artifacts() if not dry_run else hip_paths()
    live_pair = 'vkd3d' in hip_files or 'vkd3dcore' in hip_files
    # Only when something is actually being installed. hip_paths() names where
    # the artifacts WOULD go, so in a source checkout none of them exist yet -
    # enforcing the pair on a dry run made `doctor` and `--dry-run` fail with
    # "the matching modified vkd3d DLL pair is required", which is both wrong
    # (a dry run installs nothing) and unhelpful (it reads as a broken package
    # rather than an unbuilt tree). ensure_hip_artifacts() above already
    # guarantees presence on the real path.
    if not dry_run and live_pair and not all(name in hip_files and hip_files[name].is_file() for name in ('vkd3d', 'vkd3dcore')):
        raise RuntimeError('The matching modified vkd3d DLL pair is required')
    if live_pair and magpie:
        raise RuntimeError('The HIP proof of concept currently supports the in-game FSR hook, not Magpie')
    loader_name = package.LOADER_MAGPIE if magpie else package.LOADER_GAME
    with _lock(exe.parent):
        if running_game(exe):
            raise RuntimeError('Close the game before installation')
        store = exe.parent / STORE
        for name in ('continuous-every-frame.txt', 'continuous-reset-preview.txt',
                     'neural-frame-request.txt'):
            marker = exe.parent / 'DLSS5-AMD' / name
            if marker.exists() or marker.is_symlink():
                raise RuntimeError(f'Existing activation marker must be removed manually before HIP staging: {marker}')
        collisions = []
        targets = [package.ADDON, loader_name, 'dlss5_hip.dll']
        if live_pair:
            targets += ['dlss5-d3d12.dll', 'd3d12core.dll', 'ReShade.ini']
        for relative in targets:
            if (exe.parent / relative).exists():
                collisions.append(relative)
        previous = _previous_or_foreign(exe, store, dry_run=dry_run)
        managed = previous.get('files', {}) if previous.get('state') == 'installed' else {}
        unmanaged = [name for name in collisions if name not in managed]
        if unmanaged and not replace_existing:
            raise RuntimeError('Existing files need --replace-existing: ' + ', '.join(unmanaged[:12]))
        proxy_ini = None
        if live_pair:
            ini = configparser.RawConfigParser(strict=True)
            ini.optionxform = lambda optionstr: optionstr
            ini_path = exe.parent / 'ReShade.ini'
            if ini_path.exists():
                ini.read_string(_safe(ini_path).read_text(encoding='utf-8-sig'))
            if not ini.has_section('PROXY'):
                ini.add_section('PROXY')
            ini.set('PROXY', 'EnableProxyLibrary', '1')
            ini.set('PROXY', 'ProxyLibrary', '.\\dlss5-d3d12.dll')
            text = io.StringIO()
            ini.write(text, space_around_delimiters=False)
            proxy_ini = text.getvalue().encode()
        library = Path(hip_library).expanduser().absolute() if hip_library else None
        so_digest = package.sha256(hip_files['so']) if hip_files['so'].is_file() else None
        library_digest = package.sha256(library) if library is not None and library.is_file() else None
        # LD_PRELOAD cannot contain whitespace, so the preload target is a
        # per-digest cache copy, not the game-directory staged library.
        cache = bridge_cache_path(so_digest) if so_digest else None
        wrapper = wrapper_bytes(exe, magpie=magpie, gpu_name=gpu_name, hip=True,
                                hip_so=cache or (store / 'lib' / 'libdlss5_hip.so'), gpu=gpu,
                                so_digest=so_digest, hip_library=library,
                                hip_library_digest=library_digest)
        result = dict({'installed': False, 'valid': False, 'dry_run': dry_run, 'notes': list(HIP_NOTES),
                       'mode': 'magpie' if magpie else 'game', 'hip': True,
                       'weights': str(weights), 'gpu': gpu,
                       'hip_library': str(library) if library else None,
                       'bridge_cache': str(cache) if cache else None}, **prefix_paths(exe))
        if dry_run:
            result['collisions'] = collisions
            return result
        if force:
            _overwrite_previous(exe, store)
        data = _begin_install(exe, hip=True, mode='magpie' if magpie else 'game', weights=str(weights),
                              gpu=gpu, hip_library=str(library) if library else None,
                              hip_library_sha256=library_digest,
                              bridge_cache=str(cache) if cache else None,
                              bridge_cache_sha256=so_digest, force=force)
        for name in ('backups', 'logs', 'lib'):
            (store / name).mkdir(mode=0o700, exist_ok=True)
        files = data['files']
        dest_assets = exe.parent / 'DLSS5-AMD' / 'native-game-tiled-assets'
        dest_assets.mkdir(parents=True, exist_ok=True)
        keep = {entry.name for entry in weights.iterdir()
                if entry.is_file() and not entry.is_symlink()
                and (entry.suffix.lower() in {'.f32', '.f16', '.i32', '.hlsl', '.hlsli', '.cso'}
                     or entry.name in {'manifest.json', 'full-network.ok'})}
        pruned = _prune_weights(exe, files, store, data, keep, remove_unmanaged=replace_existing)
        if pruned:
            result['pruned_weights'] = pruned
        for entry in sorted(weights.iterdir()):
            if entry.is_symlink() or not entry.is_file():
                continue
            if (entry.suffix.lower() not in {'.f32', '.f16', '.i32', '.hlsl', '.hlsli', '.cso'}
                    and entry.name not in {'manifest.json', 'full-network.ok'}):
                continue
            rel = 'DLSS5-AMD/native-game-tiled-assets/' + entry.name
            _overlay(exe, rel, entry, files, store)
        _overlay(exe, package.ADDON, hip_files['addon'], files, store)
        _overlay(exe, loader_name, hip_files['reshade'], files, store)
        _overlay(exe, 'dlss5_hip.dll', hip_files['dll'], files, store)
        if live_pair:
            _overlay(exe, 'dlss5-d3d12.dll', hip_files['vkd3d'], files, store)
            _overlay(exe, 'd3d12core.dll', hip_files['vkd3dcore'], files, store)
            _overlay_bytes(exe, 'ReShade.ini', proxy_ini, files, store)
        _overlay(exe, STORE + '/lib/libdlss5_hip.so', hip_files['so'], files, store)
        if cache is not None:
            _atomic_copy(hip_files['so'], cache, expected=so_digest, mode=0o644)
        flags_src = hip_files.get('flags')
        flags_text = flags_src.read_text() if flags_src.is_file() else ''
        if 'DLSS5_HIP=' not in flags_text:
            flags_text += ('\n' if flags_text and not flags_text.endswith('\n') else '') + 'DLSS5_HIP=1\n'
        if 'DLSS5_SNAPSHOT_FRAME=' not in flags_text:
            flags_text += 'DLSS5_SNAPSHOT_FRAME=60\n'
        _overlay_bytes(exe, 'DLSS5-AMD/native-game-flags.txt', flags_text.encode(), files, store)
        # Staging is not permission to arm the existing synchronous game path.
        # Motion history is available through the host API, not this add-on.
        (exe.parent / 'DLSS5-AMD' / 'logs').mkdir(mode=0o755, exist_ok=True)
        _overlay_bytes(exe, STORE + '/launch.sh', wrapper, files, store, 0o700)
        data['state'] = 'installed'
        _journal(store, data)
        return dict(status_game(exe), installed=True, dry_run=False, hip=True)


def install_package(exe, package_root, *, magpie=False, replace_existing=False,
                    acknowledge_risk=False, dry_run=False, gpu_name=False, hip=False,
                    allow_derived_layouts=False, force=False, progress=None):
    if hip:
        return install_hip(exe, package_root, magpie=magpie, replace_existing=replace_existing,
                           acknowledge_risk=acknowledge_risk, dry_run=dry_run, gpu_name=gpu_name,
                           allow_derived_layouts=allow_derived_layouts, force=force, progress=progress)
    if not acknowledge_risk:
        raise RuntimeError('Explicit --accept-risk is required; this injects a ReShade add-on into the game directory')
    exe = _exe(exe)
    info = package.validate(package_root, magpie=magpie)
    root = Path(info['root'])
    with _lock(exe.parent):
        if running_game(exe):
            raise RuntimeError('Close the game before installation')
        store = exe.parent / STORE
        collisions = []
        planned = []
        for relative in info['files']:
            target = exe.parent / relative
            planned.append(relative)
            if target.exists():
                collisions.append(relative)
        previous = _previous_or_foreign(exe, store, dry_run=dry_run)
        managed = previous.get('files', {}) if previous.get('state') == 'installed' else {}
        unmanaged = [name for name in collisions if name not in managed]
        if unmanaged and not replace_existing:
            raise RuntimeError('Existing files need --replace-existing: ' + ', '.join(unmanaged[:12]))
        wrapper = wrapper_bytes(exe, magpie=magpie, gpu_name=gpu_name, hip=False)
        result = dict({'installed': False, 'valid': False, 'dry_run': dry_run, 'notes': list(NOTES),
                       'mode': info['mode'], 'file_count': info['file_count'],
                       'agility_sdk': info['agility_sdk'], 'hip': False}, **prefix_paths(exe))
        if dry_run:
            result['collisions'] = collisions
            return result
        if force:
            _overwrite_previous(exe, store)
        data = _begin_install(exe, hip=False, mode=info['mode'], package=info['root'], force=force)
        for name in ('backups', 'logs'):
            (store / name).mkdir(mode=0o700, exist_ok=True)
        files = data['files']
        for relative in planned:
            source = root / relative
            _overlay(exe, relative, source, files, store)
        _overlay_bytes(exe, STORE + '/launch.sh', wrapper, files, store, 0o700)
        data['state'] = 'installed'
        _journal(store, data)
        return dict(status_game(exe), installed=True, dry_run=False, hip=False)


def _teardown(exe, store, data, *, tolerant=False):
    # Preflight every path before the first restore/delete. A changed late
    # entry must not leave the earlier half of the installation removed.
    backups = []
    changed = []
    for relative, item in data['files'].items():
        target = _safe(exe.parent / relative, missing=True)
        original = item.get('original')
        if original:
            backup = store / 'backups' / relative.replace('/', '__')
            if _digest(backup) != original:
                raise RuntimeError(f'Backup missing or changed: {relative}')
            backups.append(backup)
        if target.is_file():
            allowed = {item['sha256']}
            if data.get('state') in {'installing', 'uninstalling'}:
                allowed.update((original, item.get('previous_sha256')))
            # ReShade re-saves its settings at runtime; the file is
            # regenerated on install and deleted/restored on uninstall.
            if relative not in RUNTIME_MUTABLE and _digest(target) not in allowed:
                changed.append(relative)
    if changed and not tolerant:
        raise ChangedDeploymentError(
            'Changed deployed files would be removed or restored: '
            + ', '.join(changed[:12]) + (' ...' if len(changed) > 12 else '')
            + '. Overwrite them with --overwrite (or answer yes when prompted); the changed content is discarded.')
    data['state'] = 'uninstalling'
    _journal(store, data)
    for relative, item in data['files'].items():
        target = exe.parent / relative
        original = item.get('original')
        if original:
            backup = store / 'backups' / relative.replace('/', '__')
            if not backup.is_file() or package.sha256(backup) != original:
                raise RuntimeError(f'Backup missing or changed: {relative}')
            _atomic_copy(backup, target, original)
        elif target.is_file():
            target.unlink()
            parent = target.parent
            while parent != exe.parent and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
    for backup in backups:
        backup.unlink()
    data['state'] = 'removed'
    data['files'] = {}
    _journal(store, data)
    journal = store / 'manifest.json'
    for leftover in sorted(store.rglob('*'), reverse=True):
        if leftover.is_dir() and not leftover.is_symlink() and not any(leftover.iterdir()):
            leftover.rmdir()
    if set(store.iterdir()) == {journal}:
        journal.unlink()
        store.rmdir()
    return changed


def uninstall_game(exe, *, yes=False, force=False):
    if not yes:
        raise RuntimeError('Pass --yes to restore backups and remove the dlss5 files')
    exe = _exe(exe)
    with _lock(exe.parent):
        if running_game(exe):
            raise RuntimeError('Close the game before uninstall')
        data = _load(exe)
        store = exe.parent / STORE
        _teardown(exe, store, data, tolerant=force)
        return {'installed': False, 'removed': True, 'notes': ['Remove the wrapper from your launcher settings.']}
