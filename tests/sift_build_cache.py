"""Persistent native compilation cache; no GPU or media is required."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from uav.sift_build import _cached_library


class BuildCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name)/'source'
        self.source.mkdir()
        for name in ('uav_bridge.cu', 'cudaSiftH.cu', 'cudaImage.cu', 'cudaSift.h'):
            (self.source/name).write_text('// synthetic compiler input\n')
        self.directory = Path(self.temp.name)/'cache'

    def build(self, architecture='sm_89', version='12.8'):
        return _cached_library(self.source, self.directory, architecture, version)

    def compiler(self, command, *, check):
        self.assertTrue(check)
        Path(command[-1]).write_bytes(b'synthetic library')

    def test_first_build_then_cross_process_reuse_without_compiler_discovery(self):
        with patch('uav.sift_build._find_nvcc', return_value='/image/bin/nvcc') as find, \
             patch('uav.sift_build.subprocess.run', side_effect=self.compiler) as compile:
            library = self.build()
            self.assertEqual(find.call_count, 1)
            self.assertEqual(compile.call_count, 1)
        # _cached_library has no process cache: this is the fresh-process path.
        with patch('uav.sift_build._find_nvcc', side_effect=AssertionError('cache hit')), \
             patch('uav.sift_build.subprocess.run', side_effect=AssertionError('cache hit')):
            self.assertEqual(self.build(), library)
        self.assertEqual(library.read_bytes(), b'synthetic library')

    def test_changes_and_missing_artifacts_trigger_rebuild(self):
        with patch('uav.sift_build._find_nvcc', return_value='/image/bin/nvcc'), \
             patch('uav.sift_build.subprocess.run', side_effect=self.compiler) as compile:
            library = self.build()
            (self.source/'cudaSift.h').write_text('// changed header\n')
            self.build()
            self.build('sm_86')
            self.build('sm_86', '12.9')
            library.unlink()
            self.build('sm_86', '12.9')
            (self.directory/'source.sha256').unlink()
            self.build('sm_86', '12.9')
            self.assertEqual(compile.call_count, 6)

    def test_failed_build_preserves_previous_library_and_stamp(self):
        with patch('uav.sift_build._find_nvcc', return_value='/image/bin/nvcc'), \
             patch('uav.sift_build.subprocess.run', side_effect=self.compiler):
            library = self.build()
        stamp = (self.directory/'source.sha256').read_text()
        (self.source/'uav_bridge.cu').write_text('// changed\n')
        def fail(command, **kwargs):
            Path(command[-1]).write_bytes(b'incomplete output')
            raise subprocess.CalledProcessError(1, command)
        with patch('uav.sift_build._find_nvcc', return_value='/image/bin/nvcc'), \
             patch('uav.sift_build.subprocess.run', side_effect=fail), \
             self.assertRaises(subprocess.CalledProcessError):
            self.build()
        self.assertEqual(library.read_bytes(), b'synthetic library')
        self.assertEqual((self.directory/'source.sha256').read_text(), stamp)
        self.assertFalse((self.directory/'libuav_sift.build.so').exists())

    def test_mount_path_does_not_invalidate_cache(self):
        import shutil
        with patch('uav.sift_build._find_nvcc', return_value='/image/bin/nvcc'), \
             patch('uav.sift_build.subprocess.run', side_effect=self.compiler):
            self.build()
        root = Path(self.temp.name)/'different-mount'
        shutil.copytree(self.source, root/'source')
        shutil.copytree(self.directory, root/'cache')
        with patch('uav.sift_build._find_nvcc', side_effect=AssertionError('cache hit')):
            _cached_library(root/'source', root/'cache', 'sm_89', '12.8')


if __name__ == '__main__':
    unittest.main()
