"""Local archive smoke contract, using labelled tiny build fixtures."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'build_release.py'


class ReleaseArchiveTests(unittest.TestCase):
    def test_archive_contains_offline_runtime_and_exact_member_manifest(self):
        spec = importlib.util.spec_from_file_location('build_release_test', SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                'LICENSE': 'MIT fixture',
                'hip/libdlss5_hip.so': 'synthetic ELF fixture',
                'hip/dlss5_hip.dll': 'synthetic PE fixture',
                'hip/hip-network70': 'synthetic bench fixture',
                'hip/infer_image.py': 'print("offline fixture")',
                'hip/requirements.txt': 'numpy\nPillow\n',
                'hip/README.md': 'offline contract fixture',
                'hip/include/dlss5_capi.h': '/* host ABI fixture */',
                'scripts/game-flags.txt': 'DLSS5_TILED_WEIGHTS=1\n',
                'third_party/minhook/LICENSE.txt': 'MinHook license fixture',
                'linux/ARCHIVE-README.md': 'local-only fixture',
                'linux/build/dlss5-amd.addon64': 'synthetic addon fixture',
                'linux/build/live/build-vkd3d/libs/d3d12/d3d12.dll': 'synthetic vkd3d fixture',
                'linux/build/live/build-vkd3d/libs/d3d12core/d3d12core.dll': 'synthetic core fixture',
                'linux/build/live/vkd3d-proton/LICENSE': 'LGPL fixture',
                'linux/build/live/vkd3d-proton/COPYING': 'vkd3d notice fixture',
                'linux/build/live/vkd3d-proton/AUTHORS': 'vkd3d authors fixture',
                'linux/vendor/reshade/ReShade64.dll': 'synthetic ReShade fixture',
                'linux/vendor/reshade/LICENSE.md': 'ReShade license fixture',
                'linux/vendor/reshade/PROVENANCE.txt': 'ReShade provenance fixture',
            }
            for name in ('installer.py', 'install.sh', 'THIRD-PARTY.md',
                         'dlssnr/__init__.py', 'dlssnr/cli.py', 'dlssnr/deploy.py',
                         'dlssnr/games.py', 'dlssnr/package.py', 'dlssnr/addon.py',
                         'dlssnr/convert_dll.py', 'dlssnr/runtime.py', 'dlssnr/assets.py',
                         'dlssnr/kernels.py'):
                files['linux/' + name] = '# synthetic package source fixture\n'
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            with patch.object(module, 'REPO', root), patch.object(module, 'HERE', root / 'linux'):
                with redirect_stdout(io.StringIO()):
                    module.main()
            archive = root / 'dist' / (module.ARCHIVE + '.tar.gz')
            with tarfile.open(archive) as tar:
                prefix = module.ARCHIVE + '/'
                members = {m.name.removeprefix(prefix): m for m in tar.getmembers()}
                for name in ('infer_image.py', 'requirements.txt', 'HIP.md', 'bin/hip-network70',
                             'include/dlss5_capi.h', 'manifest.json', 'licenses/MinHook-LICENSE.txt',
                             'bin/dlss5-d3d12.dll', 'bin/d3d12core.dll', 'licenses/vkd3d-LICENSE'):
                    self.assertIn(name, members)
                stream = tar.extractfile(members['manifest.json'])
                assert stream is not None
                manifest = json.load(stream)
                self.assertEqual(manifest['schema'], 1)
                self.assertFalse(manifest['gameplay_verified'])
                self.assertFalse(manifest['nvidia_equivalence_verified'])
                self.assertEqual(set(manifest['files']), set(members) - {'manifest.json'})
                for name, entry in manifest['files'].items():
                    stream = tar.extractfile(members[name])
                    assert stream is not None
                    data = stream.read()
                    self.assertEqual(entry['sha256'], hashlib.sha256(data).hexdigest())
                    self.assertEqual(entry['bytes'], len(data))
                self.assertEqual(members['bin/hip-network70'].mode, 0o755)
                self.assertEqual(members['install.sh'].mode, 0o755)
                self.assertFalse(any('tests/' in name or 'nvngx' in name for name in members))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(archive.with_suffix('.gz.sha256').read_text(), f'{digest}  {archive.name}\n')


if __name__ == '__main__':
    unittest.main()
