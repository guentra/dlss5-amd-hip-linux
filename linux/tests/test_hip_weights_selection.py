"""Explicit converter consent and read-only HIP input inspection."""
import builtins
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlssnr import cli, convert_dll, deploy, package


class WeightSelectionTests(unittest.TestCase):
    def test_local_dll_discovery_stays_with_game_and_unreal_root(self):
        self.assertTrue(hasattr(package, 'local_weights'), 'missing game-relative weights discovery')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game = root / 'game'
            exe = game / 'Project/Binaries/Win64/Game.exe'
            exe.parent.mkdir(parents=True)
            exe.touch()
            unrelated = root / 'nvngx_dlssnr.dll'
            unrelated.touch()
            self.assertIsNone(package.local_weights(exe))
            local = game / 'nvngx_dlssnr.dll'
            local.touch()
            self.assertEqual(package.local_weights(exe), local)
            adjacent = exe.parent / 'nvngx_dlssnr.dll'
            adjacent.touch()
            self.assertEqual(package.local_weights(exe), adjacent)

    def test_conversion_requires_explicit_derived_layout_consent(self):
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as exit:
            cli.parser().parse_args(['install', '--help'])
        self.assertEqual(exit.exception.code, 0)
        self.assertIn('--allow-derived-layouts', out.getvalue())
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'nvngx_dlssnr.dll'
            source.write_bytes(b'fixture')
            with patch.object(convert_dll, 'convert_nvidia_dll', return_value=Path(tmp) / 'cache') as convert:
                with self.assertRaisesRegex(RuntimeError, 'allow-derived-layouts'):
                    package.find_weights(source)
                convert.assert_not_called()
                self.assertEqual(package.find_weights(source, allow_derived_layouts=True), Path(tmp) / 'cache')
                convert.assert_called_once_with(source, layout_mode='amd-consumer-derived', progress=None)

    def test_derived_folder_requires_consent_and_keeps_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'block0-ffn.f32').write_bytes(b'\0' * 4)
            (root / 'manifest.json').write_text(json.dumps({'layout_mode':'amd-consumer-derived'}))
            with self.assertRaisesRegex(RuntimeError, 'allow-derived-layouts'):
                package.find_weights(root)
            # This manifest is deliberately incomplete; consent cannot waive it.
            with self.assertRaisesRegex(RuntimeError, 'manifest|provenance'):
                package.find_weights(root, allow_derived_layouts=True)

    def test_dll_inspection_never_converts_and_requires_pinned_digest(self):
        self.assertTrue(hasattr(package, 'inspect_weights'), 'missing read-only weight inspection')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'nvngx_dlssnr.dll'
            source.write_bytes(b'not an accepted DLL')
            with patch.object(convert_dll, 'convert_nvidia_dll', side_effect=AssertionError('converted in inspection')):
                with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                    package.inspect_weights(source)
                with patch.object(package, 'sha256', return_value=convert_dll.KNOWN_NVIDIA_SHA):
                    info = package.inspect_weights(source)
            self.assertTrue(info['conversion_required'])
            self.assertFalse(info['runtime_ready'])
            self.assertEqual(sorted(Path(tmp).iterdir()), [source])

    class _Args:
        def __init__(self):
            self.allow_derived_layouts = False
            self.magpie = False

    def test_interactive_consent_offers_to_reuse_extracted_weights(self):
        args = self._Args()
        calls = []
        def fake_inspect(weights_root, *, allow_derived_layouts):
            calls.append(allow_derived_layouts)
            if not allow_derived_layouts:
                raise package.DerivedLayoutsConsentRequired('Reconstructed cache requires --allow-derived-layouts')
            return {'root': str(weights_root), 'conversion_required': False, 'runtime_ready': True,
                    'layout_mode': 'amd-consumer-derived'}
        with patch.object(package, 'inspect_weights', side_effect=fake_inspect), \
             patch.object(builtins, 'input', return_value='y'):
            info = cli.inspect_weights_consent(args, Path('/tmp/w'), interactive=True)
        self.assertEqual(calls, [False, True], 'consent must retry inspection with the flag')
        self.assertTrue(args.allow_derived_layouts, 'consent must be remembered for the install')
        self.assertEqual(info['layout_mode'], 'amd-consumer-derived')

    def test_interactive_consent_decline_aborts_with_flag_hint(self):
        def fake_inspect(weights_root, *, allow_derived_layouts):
            if not allow_derived_layouts:
                raise package.DerivedLayoutsConsentRequired('Reconstructed cache requires --allow-derived-layouts')
            return {}
        with patch.object(package, 'inspect_weights', side_effect=fake_inspect), \
             patch.object(builtins, 'input', return_value='n'), \
             self.assertRaisesRegex(RuntimeError, 'allow-derived-layouts'):
            cli.inspect_weights_consent(self._Args(), Path('/tmp/w'), interactive=True)

    def test_non_interactive_consent_still_requires_flag(self):
        def fake_inspect(weights_root, *, allow_derived_layouts):
            if not allow_derived_layouts:
                raise package.DerivedLayoutsConsentRequired('Reconstructed cache requires --allow-derived-layouts')
            return {}
        with patch.object(package, 'inspect_weights', side_effect=fake_inspect):
            with self.assertRaises(package.DerivedLayoutsConsentRequired):
                cli.inspect_weights_consent(self._Args(), Path('/tmp/w'), interactive=False)

    def test_cli_doctor_and_dry_run_are_read_only_for_dll(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / 'Game.exe'
            exe.write_bytes(b'fixture')
            source = root / 'nvngx_dlssnr.dll'
            source.write_bytes(b'fixture')
            for command in ('doctor', 'install'):
                args = [command, '--exe', str(exe), '--runner', str(root),
                        '--weights', str(source), '--confirm-runner', '--accept-risk']
                if command == 'install':
                    args.append('--dry-run')
                out = io.StringIO()
                with patch.object(cli, 'resolve_exe', return_value=exe), \
                     patch.object(cli, 'check_host', return_value={}), \
                     patch.object(cli.games, 'inspect_game', return_value={'exe':str(exe), 'dx12':True, 'fsr_evidence':['fixture'], 'anti_cheat_evidence':[]}), \
                     patch.object(cli, 'resolve_proton', return_value={'root':root}), \
                     patch.object(cli, 'readonly_runtime', return_value={
                         'library': str(root / 'rocm' / 'libamdhip64.so.7'), 'runtime_version': 70000000,
                         'devices': [{'index': 0, 'name': 'Test GPU', 'arch': 'gfx1201',
                                      'pci_bus_id': '0000:00:00.0', 'total_memory': 1}]}), \
                     patch.object(cli.kernels, 'bundled_targets', return_value=frozenset(('gfx1201',))), \
                     patch.object(package, 'sha256', return_value=convert_dll.KNOWN_NVIDIA_SHA), \
                     patch.object(convert_dll, 'convert_nvidia_dll', side_effect=AssertionError('conversion in readonly CLI')), \
                     redirect_stdout(out):
                    self.assertEqual(cli.main(args), 0)
                self.assertFalse((root / deploy.STORE).exists())
                self.assertEqual(set(p.name for p in root.iterdir()), {'Game.exe', 'nvngx_dlssnr.dll'})
                self.assertIn('conversion', out.getvalue().lower())


if __name__ == '__main__':
    unittest.main()
