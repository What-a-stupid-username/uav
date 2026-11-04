"""Prepared debug strokes preserve pixels on in-memory synthetic canvases."""
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from trajectory_reference import draw_trajectory
from uav.template_mask.review_trajectory import PreparedTrajectory
from uav.template_mask.review import MaskOverlay,review_layout
from uav.template_mask.review_video import encoder_command,review_writer


class ReviewVideoTests(unittest.TestCase):
    def test_one_input_and_one_combined_native_output_in_both_layouts(self):
        for shape in ((1080,1920),(1281,715)):
            layout=review_layout(shape)
            self.assertGreaterEqual(layout['header_height'],96)
            self.assertGreaterEqual(layout['panel_height'],shape[0]+layout['header_height'])
            with patch('uav.ffmpeg.executable',return_value='ffmpeg'):
                command=encoder_command(Path('/synthetic'),'mask-review.mp4',layout,24)
            self.assertEqual(command.count('-i'),1)
            self.assertNotIn('-filter_complex',command)
            self.assertIn(f"{layout['width']}x{layout['height']}",command)
            self.assertEqual(command.count('-crf'),1)
            self.assertIn('/synthetic/mask-review.mp4',command)
            self.assertNotIn('/synthetic/stage1.mp4',command)
            self.assertNotIn('/synthetic/stage2.mp4',command)

    def test_encoder_failure_is_reported_after_stdin_is_closed(self):
        process=Mock();process.wait.return_value=1
        with (patch('uav.template_mask.review_video.subprocess.Popen',return_value=process),
              patch('uav.ffmpeg.executable',return_value='ffmpeg')):
            with self.assertRaisesRegex(RuntimeError,'exit 1'):
                with review_writer(Path('/synthetic'),'review.mp4',review_layout((80,100)),24):pass
        process.stdin.close.assert_called_once()
        process.wait.assert_called_once()

    def test_sparse_lookup_overlay_matches_old_formula_for_every_uint8_value(self):
        for color,opacity,background,rounding in (([140,240,40],.35,.65,True),
                                                 ([30,150,255],.55,.45,False)):
            values=np.broadcast_to(np.arange(256,dtype=np.uint8)[None,:,None],(4,256,3)).copy()
            mask=np.ones((4,256),bool);mask[1::2,::3]=False
            expected=values.copy();v=values[mask]*background+np.array(color)*opacity
            expected[mask]=(np.rint(v) if rounding else v).astype(np.uint8)
            overlay=MaskOverlay(mask,color,opacity,background_weight=background,rounding=rounding)
            overlay.draw(values)
            np.testing.assert_array_equal(values,expected)

    def test_overlay_handles_empty_masks_and_noncontiguous_portrait_panels(self):
        rng=np.random.default_rng(77)
        original=rng.integers(0,256,(31,27,3),dtype=np.uint8)
        canvas=np.zeros((32,56,3),np.uint8)
        panel=canvas[:31,28:55]
        panel[:]=0  # Conservative color must come from the original frame.
        for mask in (np.zeros((31,27),bool),rng.random((31,27))>.7):
            panel[:]=0
            expected=panel.copy()
            expected[mask]=(original[mask]*.45+np.array([30,150,255])*.55).astype(np.uint8)
            overlay=MaskOverlay(mask,[30,150,255],.55,background_weight=.45,rounding=False)
            overlay.draw(panel,source=original)
            np.testing.assert_array_equal(panel,expected)
            self.assertFalse(canvas[:,:28].any())


class PreparedTrajectoryTests(unittest.TestCase):
    def assert_same(self,shape,samples,frames):
        rng=np.random.default_rng(78)
        source=rng.integers(0,256,(*shape,3),dtype=np.uint8)
        color=(40,200,100)
        prepared=PreparedTrajectory(shape,samples,color)
        for frame in frames:
            with self.subTest(frame=frame,samples=samples[:3]):
                old,new=source.copy(),source.copy()
                draw_trajectory(old,samples,-1,color,full_path_dashed=True)
                prepared.draw(new)
                np.testing.assert_array_equal(new,old)

    def test_dense_curve_crossings_and_sparse_paths_match_exactly(self):
        dense=[(f,70+40*np.sin(f/12),55+35*np.sin(f/7)) for f in range(120)]
        self.assert_same((110,140),dense,[-1,0,1,24,48,72,119,120,167,180])
        self.assert_same((110,140),dense[::17],[-1,0,1,24,48,72,119,120,167,180])

    def test_static_subpixel_jitter_and_repeated_timestamps_match_exactly(self):
        self.assert_same((100,120),[(f,40.1,50.4) for f in range(80)],[-1,0,48,79,130])
        self.assert_same((100,120),[(0,10,10),(0,90,70),(12,20.5,40.5),(40,100.2,80.9)],[-1,0,12,40,87])

    def test_clipping_and_empty_trajectories_match_exactly(self):
        for samples in ([],[(0,10,10)],[(0,-40,-20),(20,70,60),(80,180,150)],
                        [(0,-40,-20),(20,-30,-40)]):
            self.assert_same((80,100),samples,[-1,0,1,20,48,80,120])

    def test_prepared_strokes_do_not_retain_or_mutate_source_canvas(self):
        prepared=PreparedTrajectory((80,100),[(0,10,20),(100,90,60)],(255,0,0))
        original=np.full((80,100,3),100,np.uint8)
        first,second=original.copy(),original.copy()
        prepared.draw(first);prepared.draw(second)
        np.testing.assert_array_equal(first,second)
        self.assertFalse(np.array_equal(first,original))
        self.assertLess(prepared.alpha.size,80*100)


if __name__=='__main__':unittest.main()
