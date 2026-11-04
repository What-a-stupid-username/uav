"""Audio mapping and muxing with in-memory synthetic FFmpeg sources only."""
import json
import shutil
import subprocess
import unittest
from uav.async_video_writer import encoder_command


class AudioOutputTests(unittest.TestCase):
    def test_audio_offset_optional_mapping_and_output_duration(self):
        command=encoder_command('output.mp4',32,24,24,audio_source='source.mp4',
                                audio_start=12.5,duration=2.)
        self.assertEqual(command[command.index('-ss')+1],'12.5')
        self.assertEqual(command[command.index('-t')+1],'2.0')
        self.assertIn('1:a:0?',command)
        self.assertNotIn('-shortest',command)
        self.assertNotIn('-an',command)
        self.assertIn('-an',encoder_command('silent.mp4',32,24,24))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg tools required')
    def test_synthetic_audio_and_silent_source_keep_all_video_frames(self):
        for source,has_audio in [('sine=frequency=440:sample_rate=48000:duration=0.25',True),
                                 ('color=size=32x24:rate=24:duration=1',False)]:
            command=encoder_command('pipe:1',32,24,24,audio_source=source,duration=1.)
            # Supply a generated source instead of opening any media file.
            index=command.index(source)
            command[index-1:index-1]=['-f','lavfi']
            index=command.index('-movflags')
            command[index+1]='frag_keyframe+empty_moov'
            command[-1:-1]=['-f','mp4']
            encoded=subprocess.run(command,input=bytes(32*24*3*24),
                                   capture_output=True,check=True).stdout
            probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams',
                '-of','json','pipe:0'],input=encoded,capture_output=True,check=True)
            streams=json.loads(probe.stdout)['streams']
            video=next(s for s in streams if s['codec_type']=='video')
            self.assertEqual(int(video['nb_read_frames']),24)
            self.assertEqual(any(s['codec_type']=='audio' for s in streams),has_audio)


if __name__=='__main__':unittest.main()
