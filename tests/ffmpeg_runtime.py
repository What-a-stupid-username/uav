"""Runtime binary selection without opening any media."""
import unittest
from unittest.mock import patch
from uav.ffmpeg import executable

class FfmpegTests(unittest.TestCase):
    def test_explicit_override(self):
        with patch.dict('os.environ',{'IMAGEIO_FFMPEG_EXE':'/custom/ffmpeg'}),patch('shutil.which') as lookup:
            self.assertEqual(executable(),'/custom/ffmpeg')
            lookup.assert_not_called()

    def test_runtime_path_precedes_packaged_fallback(self):
        with patch.dict('os.environ',{},clear=True),patch('shutil.which',return_value='/usr/bin/ffmpeg'):
            self.assertEqual(executable(),'/usr/bin/ffmpeg')

    def test_package_fallback(self):
        with patch.dict('os.environ',{},clear=True),patch('shutil.which',return_value=None),patch('imageio_ffmpeg.get_ffmpeg_exe',return_value='/package/ffmpeg'):
            self.assertEqual(executable(),'/package/ffmpeg')
