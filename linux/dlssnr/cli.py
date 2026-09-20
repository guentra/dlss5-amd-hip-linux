"""Terminal interface for the per-game dlss5-amd-hip Linux/Proton installer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

from . import VERSION, TAGLINE
from . import addon, assets, deploy, games, kernels, package, runtime

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


def parser():
    result = argparse.ArgumentParser(description='dlss5-amd-hip Linux / Proton: per-game installer')
    commands = result.add_subparsers(dest='command')
    for name in ('list-games', 'list-protons'):
        command = commands.add_parser(name)
        command.add_argument('--steam-root', type=Path)
        command.add_argument('--json', action='store_true')
    build = commands.add_parser('build-addon', help='Cross-compile the ReShade add-on with mingw-w64')
    build.add_argument('--output', type=Path)
    build.add_argument('--json', action='store_true')
    hipb = commands.add_parser('build-hip', help='Build libdlss5_hip.so and dlss5_hip.dll')
    hipb.add_argument('--json', action='store_true')
    for name in ('install', 'doctor', 'status', 'uninstall'):
        command = commands.add_parser(name)
        command.add_argument('--appid', help='Exact Steam application ID')
        command.add_argument('--exe', type=Path, help='Game Windows x64 executable')
        command.add_argument('--game-dir', type=Path, help='Game directory (default: directory containing install.sh)')
        command.add_argument('--steam-root', type=Path)
        command.add_argument('--json', action='store_true', help='Machine-readable JSON output')
        if name == 'uninstall':
            command.add_argument('--yes', action='store_true', help='Confirm restoring original files')
        if name in ('install', 'uninstall'):
            command.add_argument('--overwrite', action='store_true',
                                 help='Overwrite a previous installation, even if modified, without asking')
        if name in ('install', 'doctor'):
            command.add_argument('--runner', '--proton', dest='proton', type=Path,
                                 help='Installed Wine/Proton runner directory used by this game')
            command.add_argument('--confirm-runner', '--confirm-proton', dest='confirm_proton', action='store_true',
                                 help='Confirm this runner is selected in your launcher')
            command.add_argument('--gpu', help='HIP index or exact GPU name')
            command.add_argument('--hip-library', type=Path, help='Existing HIP7 library or SDK directory')
            command.add_argument('--data-dir', type=Path, help='Persistent user-local runtime/cache directory')
            command.add_argument('--install-rocm', action='store_true', help='Allow a verified AMD wheel download if needed')
            inputs = command.add_mutually_exclusive_group()
            inputs.add_argument('--weights', type=Path, help='Existing native-game-tiled-assets weights folder')
            inputs.add_argument('--nvidia-dll', type=Path, help='Your legitimately obtained nvngx_dlssnr.dll (310.8.0.0)')
            inputs.add_argument('--package', type=Path, help='Alias of --weights (package or weights folder)')
            command.add_argument('--allow-derived-layouts', action='store_true',
                                 help='Accept experimental reconstructed weight layouts, not NVIDIA equivalence')
            command.add_argument('--magpie', action='store_true',
                                 help='Magpie edition: require dxgi.dll, do not require in-game FSR')
            command.add_argument('--accept-risk', action='store_true', help='Accept experimental injection risks')
            command.add_argument('--replace-existing', action='store_true', help='Back up and replace conflicting files')
            command.add_argument('--dry-run', action='store_true', help='No writes, downloads or conversion')
        if name == 'install':
            command.add_argument('--replace-foreign', action='store_true',
                                 help='Remove a foreign dlssnr_on_amd deployment via its own uninstaller')
    command = commands.add_parser('runtime', help='Check or install a user-local HIP7 runtime')
    command.add_argument('--hip-library', type=Path)
    command.add_argument('--data-dir', type=Path)
    command.add_argument('--install-rocm', action='store_true', help='Allow the pinned official AMD wheel')
    command.add_argument('--self-test', action='store_true', help='Explicit GPU memory-copy test (256 bytes)')
    command.add_argument('--json', action='store_true')
    return result


def choose(items, label, display):
    for index, item in enumerate(items, 1):
        print(f'  {index}. {display(item)}')
    answer = input(f'{label} (number; empty to cancel): ').strip()
    if not answer.isascii() or not answer.isdecimal() or not 1 <= int(answer) <= len(items):
        raise RuntimeError('Selection cancelled or invalid.')
    return items[int(answer) - 1]


def select_gpu(devices, requested, interactive, targets):
    supported = [d for d in devices if d.get('arch', '').split(':', 1)[0] in targets]
    if requested is not None:
        selected = [d for d in supported if str(d['index']) == str(requested) or d['name'] == requested]
        if len(selected) == 1:
            return selected[0]
        raise RuntimeError('GPU missing, ambiguous or unsupported; use its exact HIP index or name with --gpu.')
    if len(supported) == 1:
        return supported[0]
    if not supported:
        raise RuntimeError('No compatible GPU for the bundled network. Targets: ' + ', '.join(sorted(targets)))
    if interactive:
        return choose(supported, 'GPU', lambda d: f"HIP {d['index']}: {d['name']} ({d['arch']})")
    raise RuntimeError('Multiple compatible GPUs; select one explicitly with --gpu INDEX or exact name.')


def bundled_targets():
    so = deploy.hip_paths()['so']
    if so.is_file():
        return kernels.bundled_targets(so)
    return kernels.FALLBACK_TARGETS


def _enter_game_path():
    entered = input('Game directory or .exe path (empty to cancel): ').strip()
    if not entered:
        raise RuntimeError('No game selected; nothing installed.')
    path = Path(entered).expanduser().absolute()
    if path.is_dir():
        return games.select_executable(path)
    if path.suffix.lower() == '.exe':
        return games.select_executable(path.parent, path)
    raise RuntimeError('Enter the game directory or the game .exe (non-Steam games are supported).')


def resolve_exe(args, interactive):
    if args.appid and (not args.appid.isascii() or not args.appid.isdecimal()):
        raise RuntimeError('--appid must be an exact numeric Steam ID.')
    if args.appid:
        if args.game_dir:
            raise RuntimeError('--game-dir and --appid cannot be combined.')
        entries = [g for g in games.discover_games(args.steam_root) if g['appid'] == args.appid]
        if len(entries) != 1:
            raise RuntimeError('AppID missing or ambiguous; use --steam-root or --exe without --appid.')
        return games.select_executable(entries[0]['path'], args.exe)
    if args.exe and not args.game_dir:
        path = args.exe.expanduser().absolute()
        if path.is_symlink():
            raise RuntimeError('The game executable must not be a symlink.')
        return games.select_executable(path.parent, path)
    root = args.game_dir.expanduser().absolute() if args.game_dir else PACKAGE_ROOT
    if not args.game_dir and root.name == 'linux':
        root = root.parent
    if not args.game_dir and not args.exe and interactive:
        entries = games.discover_games(args.steam_root)
        if entries:
            sentinel = object()
            game = choose(entries + [sentinel], 'Steam game',
                          lambda g: 'Not a Steam game (type a path)' if g is sentinel
                          else f"{g['name']} [{g['appid']}]")
            if game is sentinel:
                return _enter_game_path()
            return games.select_executable(game['path'])
    try:
        return games.select_executable(root, args.exe)
    except RuntimeError as exc:
        if not interactive or args.exe or not str(exc).startswith(('No PE x64 executable found', 'Multiple possible executables')):
            raise
        print(str(exc))
        candidates = games.find_executables(root)
        if candidates:
            selected = choose(candidates, 'Game executable', lambda p: str(p.relative_to(root)))
            return games.select_executable(root, selected)
        return _enter_game_path()
    raise RuntimeError('Specify --game-dir /path/game or --exe /path/game.exe; Steam lookup is optional with --appid ID.')


def resolve_proton(args, interactive):
    if args.proton:
        return games.validate_proton(args.proton, hip_interop=False)
    if interactive:
        print('Enter the Wine/Proton runner directory used by this game.\n\n'
              'Examples:\n'
              '  Steam:  ~/.local/share/Steam/compatibilitytools.d/GE-Proton...\n'
              '  Lutris: ~/.local/share/lutris/runners/wine/wine-ge-...\n\n'
              "Select the runner's folder, not its bin/wine executable.\n"
              'Type "steam" to list Steam runners, or leave empty to cancel.\n')
        entered = input('Runner directory: ').strip()
        if entered and entered.casefold() != 'steam':
            return games.validate_proton(Path(entered).expanduser(), hip_interop=False)
        if not entered:
            raise RuntimeError('No runner selected; specify --runner /path/to/runner.')
    elif not args.appid and not args.steam_root:
        raise RuntimeError('Specify --runner /path/to/runner (alias: --proton). Steam discovery is optional.')
    valid, errors = [], []
    for candidate in games.discover_protons(args.steam_root):
        try:
            valid.append(games.validate_proton(candidate, hip_interop=False))
        except RuntimeError as exc:
            errors.append(str(exc))
    if len(valid) == 1:
        return valid[0]
    if valid and interactive:
        return choose(valid, 'Runner (use the same one in your launcher)', lambda p: str(p['root']))
    raise RuntimeError('Specify --runner /path/to/runner; multiple candidates or none compatible.\n' + '\n'.join(errors))


def require(accepted, interactive, question, flag):
    if accepted:
        return
    if interactive and input(question + ' [y/N] ').strip().casefold() in ('y', 'yes'):
        return
    raise RuntimeError(f'Explicit confirmation required: {flag}. {question}')


def weight_progress(args):
    """Progress reporter for the NVIDIA DLL -> weights conversion (None silences it)."""
    if args.json:
        return None
    out = sys.stdout
    tty = out.isatty()

    def progress(done, total, label=None):
        if tty:
            pct = 100.0 * done / total
            filled = int(pct // 5)
            out.write('\r  extracting weights  [%s%s] %3d%% %d/%d %s'
                      % ('#' * filled, '.' * (20 - filled), pct, done, total, label or ''))
            out.flush()
            if done >= total:
                out.write('\n')
        elif done == 1 or done % 50 == 0 or done >= total:
            out.write('Extracting weights from nvngx_dlssnr.dll: %d/%d tables%s\n'
                      % (done, total, ' (' + label + ')' if label else ''))
            out.flush()

    return progress


def data_dir(args):
    base = args.data_dir or Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local/share') / 'dlssnr-linux'
    base = Path(base).expanduser()
    if not base.is_absolute():
        raise RuntimeError('--data-dir / XDG_DATA_HOME must be absolute.')
    return base


def check_host(manifest=None):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This package requires Linux x86_64.')
    name, version = platform.libc_ver()
    minimum = tuple(map(int, (manifest or {}).get('minimum_glibc', '2.34').split('.')))
    if name != 'glibc' or tuple(map(int, version.split('.'))) < minimum:
        raise RuntimeError('glibc >= ' + '.'.join(map(str, minimum)) + ' is required by the prebuilt bridge.')
    if os.geteuid() == 0:
        raise RuntimeError('Do not run this installer as sudo/root; use the game owner account.')
    return {'system': platform.system(), 'machine': platform.machine(), 'glibc': version}


def readonly_runtime(args):
    supplied = args.hip_library.expanduser().absolute() if args.hip_library else None
    if supplied and not supplied.is_dir():
        candidates = [supplied]
    else:
        candidates = runtime.discover_runtimes([supplied] if supplied else None,
                                               managed_root=data_dir(args) / 'rocm-venv')
        if supplied:
            candidates = [p for p in candidates if p.resolve().is_relative_to(supplied.resolve())]
    errors = []
    for candidate in candidates:
        try:
            return runtime.probe_runtime(candidate)
        except runtime.DriverUnavailable:
            raise
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError('HIP7 unavailable; doctor/--dry-run never download. '
                       'Use runtime --install-rocm or --hip-library.\n' + '\n'.join(errors))


def ensure_runtime(args, interactive):
    try:
        return runtime.ensure_runtime(data_dir(args), supplied=args.hip_library, allow_install=args.install_rocm)
    except runtime.DriverUnavailable:
        raise
    except RuntimeError as exc:
        if not interactive or args.hip_library or args.install_rocm or '--install-rocm' not in str(exc):
            raise
        print(str(exc))
        require(False, True, 'Download the pinned official AMD HIP7 wheel into your user directory (3 GiB free required)?', '--install-rocm')
        return runtime.ensure_runtime(data_dir(args), allow_install=True)


def resolve_weights(args, exe, interactive):
    if args.nvidia_dll is not None:
        return args.nvidia_dll.expanduser().absolute()
    if args.weights is not None:
        return args.weights.expanduser().absolute()
    local = package.local_weights(exe)
    if local is not None:
        if not args.json:
            print('Detected weights:', local)
        return local
    if interactive:
        entered = input('Path to your weights folder (native-game-tiled-assets) or nvngx_dlssnr.dll (empty to cancel): ').strip()
        if not entered:
            raise RuntimeError('No weights selected; nothing installed.')
        return Path(entered).expanduser().absolute()
    raise RuntimeError('Specify --weights / --nvidia-dll / --package, or place the weights beside the game executable.')


def build_warnings(info, weights_root, exe):
    warnings = [
        'Bundled ReShade add-on loader is copied as d3d12.dll (or dxgi.dll for Magpie).',
        'Experimental HIP network: per-frame cost is being optimized; no gameplay or NVIDIA-equivalence guarantee.',
    ]
    if weights_root.suffix.lower() == '.dll':
        warnings.append('Weights will be converted from the NVIDIA DLL into a private cache; the DLL is never bundled.')
    else:
        game_roots = [Path(exe).parent]
        if game_roots[0].name.casefold() == 'win64' and game_roots[0].parent.name.casefold() == 'binaries':
            game_roots.append(game_roots[0].parents[2])
        inside = any(weights_root == root or root in weights_root.parents or weights_root in root.parents
                     for root in game_roots)
        if not inside:
            warnings.append(f'Weights come from outside the game directory ({weights_root}): they must have been '
                            'generated from this game\'s own DLSS5 model, otherwise the output will not match this title.')
    return warnings


def emit(result, args):
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == 'runtime':
        print('HIP7 verified:', result['library'])
        for device in result['devices']:
            print(f"  HIP {device['index']}: {device['name']} ({device['arch']})")
        print('GPU copy test:', 'completed' if args.self_test else 'not requested')
        return
    if args.command == 'build-addon':
        print('Add-on:', result['addon'])
        print('SHA256:', result['sha256'])
        return
    if args.command == 'build-hip':
        print('HIP library:', result.get('so'))
        print('PE trampoline:', result.get('dll'))
        return
    if args.command == 'doctor':
        print('Static and HIP prerequisites checked. No files modified.')
        print('Game:', result['game']['exe'])
        print('HIP runtime:', result['runtime']['library'])
        for device in result['runtime']['devices']:
            print(f"  HIP {device['index']}: {device['name']} ({device['arch']})")
        print('GPU:', result['gpu']['name'])
        print('Runner:', result['proton']['root'])
        print('Weights:', result.get('weights'))
        print('Conversion required:', result.get('conversion_required', False))
        for warning in result.get('warnings', []):
            print('Warning:', warning)
        print('In-game rendering is NOT verified.')
        return
    if result.get('dry_run'):
        print('Dry run completed; nothing installed.')
    elif result.get('removed'):
        print('Uninstalled. Remove the wrapper from the launcher.')
    elif result.get('installed') and result.get('valid'):
        print('Installation verified on disk (not an in-game rendering validation).')
    elif result.get('installed') or result.get('pending'):
        print('Installation needs attention; read diagnostics before making changes.')
    else:
        print('No active managed installation.')
    if result.get('conversion_required') is not None:
        print('Conversion required:', result['conversion_required'])
    if result.get('exe'):
        print('Game:', result['exe'])
    if result.get('gpu'):
        print('GPU:', result['gpu']['name'])
    if result.get('hip_library'):
        print('HIP runtime:', result['hip_library'])
    if result.get('bridge_cache'):
        print('HIP bridge cache (LD_PRELOAD target):', result['bridge_cache'])
    if result.get('proton'):
        print('Use this Wine/Proton runner in your launcher:', result['proton'])
    if result.get('command_prefix'):
        print('Command prefix for Lutris / another launcher (no %command%):')
        print(result['command_prefix'])
        print('Keep your existing runner, Wine prefix and game arguments; do not launch the .exe directly from Linux.')
    if result.get('launch_options'):
        print('Steam launch options (only if using Steam; preserve unrelated existing options):')
        print(result['launch_options'])
        print('Enable FSR in the game as the injection hook; the native network runs on the selected GPU.')
    for warning in result.get('warnings', []):
        print('Warning:', warning)
    for note in result.get('notes', []):
        print('-', note)
    if args.command == 'uninstall':
        print('Remove the wrapper from your launcher command prefix or Steam launch options. Shared runtime caches are retained.')


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments and not sys.stdin.isatty():
        print('Usage: install --exe /path/game.exe; see --help.', file=sys.stderr)
        return 2
    args = parser().parse_args(arguments or ['install'])
    interactive = sys.stdin.isatty() and not getattr(args, 'json', False)
    try:
        if not args.json:
            print(f'dlss5-amd-hip {VERSION} — {TAGLINE}')
        if args.command in ('list-games', 'list-protons'):
            if args.command == 'list-games':
                rows = games.discover_games(args.steam_root)
            else:
                rows = []
                for candidate in games.discover_protons(args.steam_root):
                    try:
                        games.validate_proton(candidate, hip_interop=False)
                        rows.append({'path': candidate, 'compatible': True})
                    except RuntimeError as exc:
                        rows.append({'path': candidate, 'compatible': False, 'reason': str(exc)})
            if args.json:
                print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
            else:
                for row in rows:
                    print(f"{row.get('appid', 'OK' if row.get('compatible') else 'INCOMPATIBLE')}  {row.get('name', '')}  {row['path']}")
                    if row.get('reason'):
                        print('  ' + row['reason'])
                if not rows:
                    print('No entries found. Use --steam-root or an explicit --exe / --proton path.')
            return 0
        if args.command == 'build-addon':
            path = addon.build(args.output)
            emit({'addon': str(path), 'sha256': package.sha256(path)}, args)
            return 0
        if args.command == 'build-hip':
            import subprocess
            hip = REPO_ROOT / 'hip'
            result = subprocess.run(['make', '-C', str(hip), 'game'], check=False, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError((result.stdout + '\n' + result.stderr).strip() or 'make -C hip game failed')
            paths = deploy.hip_paths()
            emit({'so': str(paths['so']), 'dll': str(paths['dll'])}, args)
            return 0
        if args.command == 'runtime':
            check_host(None)
            rt = ensure_runtime(args, interactive)
            if args.self_test:
                rt = runtime.probe_runtime(rt['library'], self_test=True)
            emit(rt, args)
            return 0
        exe = resolve_exe(args, interactive)
        if args.command == 'status':
            status = deploy.status_game(exe)
            emit(status, args)
            return 1 if status['pending'] else 0
        if args.command == 'uninstall':
            require(args.yes, interactive, 'Restore original files and uninstall?', '--yes')
            try:
                result = deploy.uninstall_game(exe, yes=True, force=args.overwrite)
            except deploy.ChangedDeploymentError as exc:
                if not interactive:
                    raise
                print(str(exc))
                require(False, interactive, 'Overwrite the changed files and continue?', '--overwrite')
                result = deploy.uninstall_game(exe, yes=True, force=True)
            actual = deploy.status_game(exe)
            if actual['installed'] or actual['pending']:
                raise RuntimeError('Restore incomplete; retain all backups and the journal.')
            emit(result, args)
            return 0
        manifest = assets.verify_assets(PACKAGE_ROOT) if (PACKAGE_ROOT / 'manifest.json').is_file() else {}
        if manifest and not args.json:
            print('dlss5-amd-hip installer (release integrity verified):', manifest.get('version', 'unknown'))
            print()
        host = check_host(manifest)
        evidence = games.inspect_game(exe)
        for key, explanation in (('dx12', 'No static DirectX 12 evidence found'),
                                 ('fsr_evidence', 'No FSR/FidelityFX evidence found')):
            if not evidence[key] and not args.magpie:
                raise RuntimeError(explanation + '; use --magpie for the window-scaler edition.')
        if evidence['anti_cheat_evidence']:
            raise RuntimeError('Anti-cheat detected: refusing installation, no bypass. ' + ', '.join(evidence['anti_cheat_evidence']))
        proton = resolve_proton(args, interactive)
        weights_root = resolve_weights(args, exe, interactive)
        info = dict(package.inspect_weights(weights_root, allow_derived_layouts=args.allow_derived_layouts),
                    mode='magpie' if args.magpie else 'game', hip=True)
        warnings = build_warnings(info, weights_root, exe)
        foreign = deploy.foreign_deployment(exe)
        if foreign is not None:
            if args.command == 'doctor':
                warnings.append('Foreign dlssnr_on_amd deployment present; install requires '
                                '--replace-foreign (or interactive consent) to remove it first.')
            elif args.dry_run:
                warnings.append('Foreign dlssnr_on_amd deployment present; the real install requires '
                                '--replace-foreign to remove it via its own uninstaller first.')
                emit({'dry_run': True, 'installed': False, 'valid': False, 'foreign': True,
                      'warnings': warnings, 'notes': []}, args)
                return 0
            elif args.replace_foreign:
                deploy.remove_foreign_deployment(exe)
                if not args.json:
                    print('Removed the foreign dlssnr_on_amd deployment via its own uninstaller.')
            elif interactive:
                require(False, True,
                        'A foreign dlssnr_on_amd deployment owns this game directory. '
                        'Remove it via its own uninstaller and continue?', '--replace-foreign')
                deploy.remove_foreign_deployment(exe)
                print('Removed the foreign dlssnr_on_amd deployment via its own uninstaller.')
            else:
                raise RuntimeError('A foreign dlssnr_on_amd deployment owns this game directory; '
                                   're-run with --replace-foreign to remove it via its own uninstaller.')
        if args.command == 'doctor':
            rt = readonly_runtime(args)
            gpu = select_gpu(rt['devices'], args.gpu, interactive, bundled_targets())
            emit({'host': host, 'game': evidence, 'proton': proton, 'runtime': rt, 'gpu': gpu,
                  'weights': str(weights_root), 'conversion_required': bool(info.get('conversion_required')),
                  'package': info, 'warnings': warnings}, args)
            return 0
        require(args.confirm_proton, interactive,
                f"Does your launcher or Wine command use {proton['root']} for this game?",
                '--confirm-runner (alias: --confirm-proton)')
        require(args.accept_risk, interactive,
                'Experimental injection may crash, render incorrectly or trigger anti-cheat. Continue?',
                '--accept-risk')
        rt = readonly_runtime(args) if args.dry_run else ensure_runtime(args, interactive)
        gpu = select_gpu(rt['devices'], args.gpu, interactive, bundled_targets())
        if args.dry_run:
            result = deploy.install_hip(exe, weights_root, magpie=args.magpie,
                                        replace_existing=args.replace_existing,
                                        acknowledge_risk=True, dry_run=True, gpu=gpu,
                                        hip_library=rt.get('library'), allow_derived_layouts=True)
            result['conversion_required'] = bool(info.get('conversion_required'))
            result['warnings'] = warnings
            emit(result, args)
            return 0
        if info.get('conversion_required'):
            require(args.allow_derived_layouts, interactive,
                    'Accept experimental reconstructed weight layouts (not NVIDIA equivalence)?',
                    '--allow-derived-layouts')
            if not args.json:
                print('Extracting network weights from the NVIDIA DLL ...')

        def do_install(force_overwrite):
            return deploy.install_hip(exe, weights_root, magpie=args.magpie,
                                      replace_existing=args.replace_existing,
                                      acknowledge_risk=True, dry_run=False, gpu=gpu,
                                      hip_library=rt.get('library'), allow_derived_layouts=True,
                                      force=force_overwrite, progress=weight_progress(args))
        try:
            result = do_install(args.overwrite)
        except deploy.ChangedDeploymentError as exc:
            if not interactive:
                raise
            print(str(exc))
            require(False, interactive, 'Overwrite the previous installation?', '--overwrite')
            result = do_install(True)
        result['warnings'] = warnings
        result['gpu'] = gpu
        emit(result, args)
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print('Cancelled. Run status before your next operation.', file=sys.stderr)
        return 130
