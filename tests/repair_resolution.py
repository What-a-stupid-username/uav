"""Scaled input support, global interpolation, and writer ownership; synthetic only."""
import unittest
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from unittest.mock import patch
from uav.config import UAVConfig,use_config
from uav.template_mask.repair import repair_frames
from uav.repair_resolution import RepairInputs,ResolutionWriter,repair_size,restore_rgb

class ResolutionTests(unittest.TestCase):
    def test_scaled_handoff_restores_before_external_writer(self):
        image=np.full((24,32,3),31,np.uint8)
        mask=np.zeros((24,32),bool);mask[4:12,8:16]=True
        class Frames:
            count=3;fps=24
            def __call__(self,index):return image
        outputs=[]
        writer=SimpleNamespace(save=outputs.append,close=lambda:None,profile=lambda:{})
        class Sink:
            def __init__(self,model,path,w,h,*args,writer,**kw):
                self.writer=writer
                self.size=(w,h)
            def save(self,index,mask,rgb,**kw):
                self.writer.save(torch.full_like(rgb.permute(1,2,0),99))
            def close(self):self.writer.close()
            def profile(self):return dict(complete=True)
        real_to=torch.Tensor.to
        def to(tensor,*args,**kwargs):
            return tensor if args and args[0]=='cuda' else real_to(tensor,*args,**kwargs)
        with (use_config(UAVConfig(repair_scale=.7)),
              patch('uav.template_mask.repair.RepairSink',Sink),
              patch('torch.Tensor.to',to),patch('torch.cuda.synchronize')):
            report=repair_frames(Frames(),lambda i:mask,'/synthetic',object(),writer=writer)
        self.assertEqual(len(outputs),3)
        self.assertEqual(report['output_sink'],'segment_blend')
        for value in outputs:
            self.assertEqual(tuple(value.shape),(24,32,3))
            self.assertTrue(bool((value[mask]==99).all()))
            self.assertTrue(bool((value[~mask]==31).all()))

    def test_sizes_and_validation(self):
        self.assertEqual(repair_size(720,1316,.7),(504,920))
        self.assertEqual(repair_size(7,5,.7),(7,5))
        self.assertEqual(repair_size(719,1315,1),(719,1315))
        for scale in (0,-1,2,float('nan')):
            with self.assertRaises(ValueError):repair_size(20,30,scale)

    def test_preparation_order_support_and_failure(self):
        image=np.arange(24*32*3,dtype=np.uint8).reshape(24,32,3)
        class Frames:
            count=5
            def __call__(self,index):return image
        mask=np.zeros((24,32),bool);mask[3:20,7]=True
        calls=[]
        def mask_at(index):calls.append(index);return mask
        inputs=RepairInputs(Frames(),mask_at,(24,16),capacity=2)
        rows=list(inputs)
        self.assertEqual(calls,list(range(5)))
        for index,rgb,local,boxes,original,full in rows:
            expected=cv2.cvtColor(cv2.resize(image,(24,16),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(rgb,expected)
            np.testing.assert_array_equal(local,cv2.resize(mask.astype(np.float32),(24,16),interpolation=cv2.INTER_AREA)>0)
            np.testing.assert_array_equal(original,cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            self.assertIs(full,mask)
        inputs=RepairInputs(Frames(),lambda i:mask.astype(np.uint8),(24,16))
        with self.assertRaises(ValueError):list(inputs)

    def test_writer_uses_original_mask_and_global_coordinates(self):
        torch.manual_seed(4)
        rgb=torch.randint(256,(8,16,3),dtype=torch.uint8)
        image=np.arange(24*32*3,dtype=np.uint8).reshape(24,32,3)
        mask=np.zeros((24,32),bool);mask[0:4,0:3]=True;mask[10:14,19:22]=True
        frames=[];closed=[]
        writer=ResolutionWriter(SimpleNamespace(save=frames.append,close=lambda:closed.append(True),profile=lambda:{}))
        for index,support in enumerate((mask,np.zeros_like(mask))):
            writer.remember(index,image,support);writer.save(rgb)
        expected=F.interpolate(rgb.permute(2,0,1)[None].float(),size=(24,32),mode='bilinear',align_corners=False)[0].permute(1,2,0).round().byte()
        self.assertTrue(torch.equal(frames[0][mask],expected[mask]))
        self.assertTrue(torch.equal(frames[0][~mask],torch.from_numpy(image)[~mask]))
        self.assertTrue(torch.equal(frames[1],torch.from_numpy(image)))
        self.assertFalse(writer.originals);writer.close();self.assertEqual(closed,[True])

if __name__=='__main__':unittest.main()
