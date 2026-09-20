"""Local HIP staging regressions; never install into an actual game."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlssnr import deploy, package


def fixture(root):
    exe = root / 'game/Game.exe'
    exe.parent.mkdir()
    exe.write_bytes(b'MZ fixture')
    weights = root / 'weights'
    weights.mkdir()
    (weights / 'block0.f16').write_bytes(b'\0\0')
    artifacts = {key: root / 'artifacts' / key for key in ('so', 'dll', 'addon', 'reshade', 'flags')}
    for key, path in artifacts.items():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'DLSS5_HIP=1\n' if key == 'flags' else ('fixture-' + key).encode())
    return exe, weights, artifacts


class HipStagingSafetyTests(unittest.TestCase):
    def setUp(self):
        # Keep the shared LD_PRELOAD bridge cache out of the real user home.
        self._xdg = tempfile.TemporaryDirectory()
        self._patcher = patch.dict(os.environ, {'XDG_DATA_HOME': self._xdg.name})
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._xdg.cleanup()

    def test_uninstall_checks_all_targets_before_restoring_any_and_keeps_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, weights, artifacts = fixture(root)
            flags = exe.parent / 'DLSS5-AMD/native-game-flags.txt'
            flags.parent.mkdir()
            flags.write_bytes(b'user flags')
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                deploy.install_hip(exe, weights, acknowledge_risk=True, replace_existing=True)
            notes = exe.parent / deploy.STORE / 'user-notes.txt'
            notes.write_bytes(b'keep notes')
            dll = exe.parent / 'dlss5_hip.dll'
            dll.write_bytes(b'user changed DLL')
            before = {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaises(RuntimeError):
                deploy.uninstall_game(exe, yes=True)
            self.assertEqual(before, {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}, 'partial uninstall before validation')
            dll.write_bytes(artifacts['dll'].read_bytes())
            deploy.uninstall_game(exe, yes=True)
            self.assertTrue(notes.exists(), 'unrelated store file removed')
            self.assertEqual(notes.read_bytes(), b'keep notes')
            self.assertEqual(flags.read_bytes(), b'user flags')
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                self.assertTrue(deploy.install_hip(exe, weights, acknowledge_risk=True,
                                                  replace_existing=True)['valid'])
            self.assertEqual(notes.read_bytes(), b'keep notes')

    def test_install_failure_leaves_recoverable_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, weights, artifacts = fixture(root)
            original_copy = deploy._atomic_copy
            def fail_addon(source, target, *args, **kwargs):
                if target.name == package.ADDON:
                    raise OSError('injected copy failure')
                return original_copy(source, target, *args, **kwargs)
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts), \
                 patch.object(deploy, '_atomic_copy', side_effect=fail_addon):
                with self.assertRaisesRegex(OSError, 'injected'):
                    deploy.install_hip(exe, weights, acknowledge_risk=True)
            journal = exe.parent / deploy.STORE / 'manifest.json'
            self.assertTrue(journal.is_file(), 'copy failure lost recovery journal')
            self.assertEqual(json.loads(journal.read_text())['state'], 'installing')
            self.assertFalse(deploy.status_game(exe)['valid'])
            deploy.uninstall_game(exe, yes=True)
            self.assertEqual([p for p in exe.parent.rglob('*') if p.is_file()], [exe])

    def test_status_rejects_incomplete_or_tampered_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, weights, artifacts = fixture(root)
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                deploy.install_hip(exe, weights, acknowledge_risk=True)
            journal = exe.parent / deploy.STORE / 'manifest.json'
            original = journal.read_bytes()
            data = json.loads(original)
            data['state'] = 'installing'
            journal.write_text(json.dumps(data))
            self.assertFalse(deploy.status_game(exe)['valid'], 'incomplete journal accepted')
            journal.write_bytes(original)
            (exe.parent / 'dlss5_hip.dll').write_bytes(b'tampered')
            self.assertFalse(deploy.status_game(exe)['valid'], 'changed deployed DLL accepted')

    def test_uninstall_rejects_journal_path_escape_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, weights, artifacts = fixture(root)
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                deploy.install_hip(exe, weights, acknowledge_risk=True)
            outside = root / 'outside.txt'
            outside.write_bytes(b'user outside')
            journal = exe.parent / deploy.STORE / 'manifest.json'
            data = json.loads(journal.read_bytes())
            data['files']['../outside.txt'] = {'sha256':package.sha256(outside), 'original':None}
            journal.write_text(json.dumps(data))
            before = {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaisesRegex(RuntimeError, 'journal path'):
                deploy.uninstall_game(exe, yes=True)
            self.assertEqual(before, {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_weight_provenance_is_preserved_in_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, weights, artifacts = fixture(root)
            (weights / 'manifest.json').write_bytes(b'validated fixture manifest')
            (weights / 'full-network.ok').write_bytes(b'validated fixture marker')
            # Only the converter validation is stubbed: copying/journaling is real.
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts), \
                 patch.object(package, 'find_weights', return_value=weights):
                deploy.install_hip(exe, weights, acknowledge_risk=True)
            assets = exe.parent / 'DLSS5-AMD/native-game-tiled-assets'
            for name in ('manifest.json', 'full-network.ok'):
                self.assertTrue((assets / name).is_file(), 'lost weight provenance')
                self.assertEqual((assets / name).read_bytes(), (weights / name).read_bytes())

    def test_update_and_uninstall_restore_first_install_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, weights, artifacts = fixture(root)
            flags = exe.parent / 'DLSS5-AMD/native-game-flags.txt'
            flags.parent.mkdir()
            flags.write_bytes(b'user flags')
            loader = exe.parent / 'd3d12.dll'
            loader.write_bytes(b'user loader')
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                deploy.install_hip(exe, weights, acknowledge_risk=True, replace_existing=True)
                artifacts['reshade'].write_bytes(b'updated loader')
                deploy.install_hip(exe, weights, acknowledge_risk=True)
            self.assertTrue(deploy.uninstall_game(exe, yes=True)['removed'])
            self.assertEqual(flags.read_bytes(), b'user flags')
            self.assertEqual(loader.read_bytes(), b'user loader')
            self.assertFalse((exe.parent / 'dlss5_hip.dll').exists())

    def test_install_refuses_modified_deployment_until_overwrite_is_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, weights, artifacts = fixture(root)
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                deploy.install_hip(exe, weights, acknowledge_risk=True)
            dll = exe.parent / 'dlss5_hip.dll'
            dll.write_bytes(b'user changed DLL')
            with self.assertRaises(deploy.ChangedDeploymentError):
                deploy.install_hip(exe, weights, acknowledge_risk=True)
            self.assertEqual(dll.read_bytes(), b'user changed DLL', 'refused install touched the file')
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                self.assertTrue(deploy.install_hip(exe, weights, acknowledge_risk=True, force=True)['valid'])
            self.assertEqual(dll.read_bytes(), artifacts['dll'].read_bytes())

    def test_uninstall_force_overrides_changed_file_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe, weights, artifacts = fixture(root)
            with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                deploy.install_hip(exe, weights, acknowledge_risk=True)
            dll = exe.parent / 'dlss5_hip.dll'
            dll.write_bytes(b'user changed DLL')
            with self.assertRaises(deploy.ChangedDeploymentError):
                deploy.uninstall_game(exe, yes=True)
            self.assertEqual(dll.read_bytes(), b'user changed DLL', 'refused uninstall touched the file')
            self.assertTrue(deploy.uninstall_game(exe, yes=True, force=True)['removed'])
            self.assertFalse(dll.exists())

    def test_existing_activation_markers_refuse_without_deleting_user_files(self):
        for name in ('continuous-every-frame.txt', 'continuous-reset-preview.txt',
                     'neural-frame-request.txt'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                exe, weights, artifacts = fixture(root)
                marker = exe.parent / 'DLSS5-AMD' / name
                marker.parent.mkdir()
                marker.write_bytes(b'user-request')
                before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
                with patch.object(deploy, 'ensure_hip_artifacts', return_value=artifacts):
                    with self.assertRaisesRegex(RuntimeError, 'activation marker'):
                        deploy.install_hip(exe, weights, acknowledge_risk=True)
                self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
