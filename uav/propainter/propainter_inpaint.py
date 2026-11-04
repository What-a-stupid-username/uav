from uav.config import get_config
# -*- coding: utf-8 -*-
import os
import gc
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image
from typing import List

import torch
import torchvision

from uav.propainter.video.model.modules.flow_comp_raft import RAFT_bi
from uav.propainter.video.model.recurrent_flow_completion import RecurrentFlowCompleteNet
from uav.propainter.video.model.propainter import InpaintGenerator
from uav.propainter.video.core.utils import to_tensors
from uav.propainter.video.model.misc import get_device

import warnings

warnings.filterwarnings("ignore")

def binary_mask(mask, th=0.1):
    mask[mask > th] = 1
    mask[mask <= th] = 0
    return mask


# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    # 如果传入的直接为numpy array
    if isinstance(mpath, np.ndarray):
        if mpath.ndim == 3 and mpath.shape[2] == 1:
            mpath = mpath.squeeze(2)  # 从 (H,W,1) 转为 (H,W)
        elif mpath.ndim == 3 and mpath.shape[2] == 3:
            # 如果是彩色图像，转为灰度
            mpath = cv2.cvtColor(mpath, cv2.COLOR_BGR2GRAY)
        masks_img = [Image.fromarray(mpath)]
    # input single img path
    else:
        if isinstance(mpath, str):
            if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
                masks_img = [Image.open(mpath)]
        elif isinstance(mpath, (list, tuple)):
            for mask_img in mpath:
                if isinstance(mask_img, Image.Image):
                    masks_img.append(mask_img)
                else:
                    masks_img.append(Image.fromarray(np.asarray(mask_img)))
        else:
            mnames = sorted(os.listdir(mpath))
            for mp in mnames:
                masks_img.append(Image.open(os.path.join(mpath, mp)))

    for mask_img in masks_img:
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))

        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))

    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


class PropainterInpaint:
    def __init__(self, device, model_dir, sub_video_length=80, use_fp16=True,
                 flow_mask_dilation=4, mask_dilation=4):
        self.device = device
        self.model_dir = model_dir
        self.use_fp16 = use_fp16
        self.use_half = True if self.use_fp16 else False
        if self.device == torch.device('cpu'):
            self.use_half = False
        # Length of sub-video for long video inference.
        self.sub_video_length = sub_video_length
        # Length of local neighboring frames.'
        self.neighbor_length = 10
        # Distance between transformer window centers. The original model uses
        # 50% overlap (5); larger values reduce repeated transformer work.
        self.neighbor_stride = 10
        # Flow estimation needs a wider safety margin than final compositing.
        # Keeping these separate prevents clean pixels around a precise mask
        # from being needlessly synthesized.
        self.flow_mask_dilation = flow_mask_dilation
        self.mask_dilation = mask_dilation
        # Stride of global reference frames
        self.ref_stride = 10
        # Iterations for RAFT inference
        self.raft_iter = 20
        # Stride of global reference frames
        self.ref_stride = 10
        # 设置raft模型
        self.fix_raft = self.init_raft_model()
        # 设置fix_flow模型
        self.fix_flow_complete = self.init_fix_flow_model()
        # 设置inpaint模型
        self.model = self.init_inpaint_model()

    def init_raft_model(self):
        # set up RAFT and flow competition model
        return RAFT_bi(os.path.join(self.model_dir, 'raft-things.pth'), self.device)

    def init_fix_flow_model(self):
        fix_flow_complete_model = RecurrentFlowCompleteNet(
            os.path.join(self.model_dir, 'recurrent_flow_completion.pth'))
        for p in fix_flow_complete_model.parameters():
            p.requires_grad = False

        if self.use_half:
            fix_flow_complete_model = fix_flow_complete_model.half()
        fix_flow_complete_model.to(self.device)
        fix_flow_complete_model.eval()
        return fix_flow_complete_model

    def init_inpaint_model(self):
        # set up ProPainter model
        model = InpaintGenerator(model_path=os.path.join(self.model_dir, 'ProPainter.pth'))
        if self.use_half:
            model = model.half()
        model = model.to(self.device).eval()
        if (self.device.type=='cuda' and self.use_half and
                get_config().propainter_graph_propagation and
                get_config().propainter_stream_deform):
            from uav.cuda_graph import graph_propagation
            model.feat_prop_module.forward=graph_propagation(model.feat_prop_module)
        return model

    @torch.inference_mode()
    def inpaint(self, frames, mask, output_range=None, output_presence=None, *, stack_output=True, return_features=False, window_length=None, window_overlap=5):
        gpu_input = isinstance(frames, torch.Tensor)
        if output_presence is not None:
            if output_range is None or len(output_presence) != len(frames):
                raise ValueError('output_presence requires output_range and one flag per frame')
        if output_range is not None:
            if not gpu_input or not 0 <= output_range[0] < output_range[1] <= len(frames):
                raise ValueError('output_range requires GPU frames and valid local frame bounds')
        if gpu_input:
            if frames.device.type != 'cuda' or frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
                raise ValueError('GPU frames must be CUDA uint8 TCHW RGB')
            if mask.device != frames.device or mask.shape != (len(frames), *frames.shape[-2:]):
                raise ValueError('GPU masks must match frames on the same device')
            from uav.gpu_morphology import dilate_cross
            original_gpu = frames
            h,w = frames.shape[-2:]
            binary = mask.bool()[:,None]
            flow_masks = dilate_cross(binary, self.flow_mask_dilation).float()[None]
            masks_dilated = dilate_cross(binary, self.mask_dilation).float()[None]
            frames = frames.float().div(255).unsqueeze(0) * 2 - 1
        else:
            if isinstance(frames[0], np.ndarray):
                frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
            size = frames[0].size
            frames_len = len(frames)
            flow_masks, masks_dilated = read_mask(mask, frames_len, size,
                                                  flow_mask_dilates=self.flow_mask_dilation,
                                                  mask_dilates=self.mask_dilation)
            w, h = size
            frames_inp = [np.array(f).astype(np.uint8) for f in frames]
            frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
            flow_masks = to_tensors()(flow_masks).unsqueeze(0)
            masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
            frames, flow_masks, masks_dilated = frames.to(self.device), flow_masks.to(self.device), masks_dilated.to(
                self.device)
        video_length = frames.size(1)
        stage_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        stage_events[0].record()
        from uav.raft_resolution import estimate_flow,flow_size
        raft_scale=get_config().raft_scale
        flow_size(h,w,raft_scale)
        def estimate(part,offset=0):
            cache=getattr(self,'flow_cache',None)
            if cache is None:return estimate_flow(self.fix_raft,part,self.raft_iter,raft_scale)
            return cache.estimate(self.fix_raft,part,self.flow_start+offset,self.flow_crop,self.raft_iter,raft_scale)
        with torch.no_grad():
            transformer_only=get_config().propainter_transformer_only
            if transformer_only:
                if self.use_half:frames,masks_dilated=frames.half(),masks_dilated.half()
                stage_events[1].record();stage_events[2].record()
                updated_frames=frames*(1-masks_dilated)
                updated_masks=masks_dilated
                pred_flows_bi=None
                del frames,flow_masks
            else:
                # ---- compute flow ----
                if frames.size(-1) <= 640:
                    short_clip_len = 12
                elif frames.size(-1) <= 720:
                    short_clip_len = 8
                elif frames.size(-1) <= 1280:
                    short_clip_len = 4
                else:
                    short_clip_len = 2

                # use fp32 for RAFT
                if frames.size(1) > short_clip_len:
                    gt_flows_f_list, gt_flows_b_list = [], []
                    for f in range(0, video_length, short_clip_len):
                        end_f = min(video_length, f + short_clip_len)
                        if f == 0:
                            flows_f, flows_b = estimate(frames[:, f:end_f],f)
                        else:
                            flows_f, flows_b = estimate(frames[:, f - 1:end_f],f-1)
                        gt_flows_f_list.append(flows_f)
                        gt_flows_b_list.append(flows_b)
                    gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                    gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                    gt_flows_bi = (gt_flows_f, gt_flows_b)
                    del gt_flows_f_list, gt_flows_b_list, flows_f, flows_b, gt_flows_f, gt_flows_b
                else:
                    gt_flows_bi = estimate(frames)

                stage_events[1].record()

                if self.use_half:
                    frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                    gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())

                # ---- complete flow ----
                from uav.flow_completion_resolution import complete_flow
                completion_scale=get_config().flow_completion_scale
                flow_length = gt_flows_bi[0].size(1)
                if flow_length > self.sub_video_length:
                    pred_flows_f, pred_flows_b = [], []
                    pad_len = 5
                    for f in range(0, flow_length, self.sub_video_length):
                        s_f = max(0, f - pad_len)
                        e_f = min(flow_length, f + self.sub_video_length + pad_len)
                        pad_len_s = max(0, f) - s_f
                        pad_len_e = e_f - min(flow_length, f + self.sub_video_length)
                        pred_flows_bi_sub = complete_flow(self.fix_flow_complete,
                            (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                            flow_masks[:, s_f:e_f + 1],completion_scale)

                        pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                        pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])

                    pred_flows_f = torch.cat(pred_flows_f, dim=1)
                    pred_flows_b = torch.cat(pred_flows_b, dim=1)
                    pred_flows_bi = (pred_flows_f, pred_flows_b)
                else:
                    pred_flows_bi = complete_flow(self.fix_flow_complete,gt_flows_bi,flow_masks,completion_scale)

                del gt_flows_bi, flow_masks
                stage_events[2].record()

                # ---- image propagation ----
                masked_frames = frames * (1 - masks_dilated)
                # ensure a minimum of 100 frames for image propagation
                subvideo_length_img_prop = min(100, self.sub_video_length)
                if video_length > subvideo_length_img_prop:
                    updated_frames, updated_masks = [], []
                    pad_len = 10
                    for f in range(0, video_length, subvideo_length_img_prop):
                        s_f = max(0, f - pad_len)
                        e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                        pad_len_s = max(0, f) - s_f
                        pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                        b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                        pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                        prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(masked_frames[:, s_f:e_f],
                                                                                            pred_flows_bi_sub,
                                                                                            masks_dilated[:, s_f:e_f],
                                                                                            'nearest')
                        updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                        updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                        updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                        updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])

                    updated_frames = torch.cat(updated_frames, dim=1)
                    updated_masks = torch.cat(updated_masks, dim=1)
                else:
                    b, t, _, _, _ = masks_dilated.size()
                    prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, pred_flows_bi, masks_dilated,
                                                                           'nearest')
                    updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                    updated_masks = updated_local_masks.view(b, t, 1, h, w)
                    del prop_imgs, updated_local_masks

                del masked_frames, frames
            stage_events[3].record()
            if gpu_input and getattr(self,'debug_propagation',False):
                # Debug only. Snapshot before learned feature propagation and Transformer.
                self.last_propagated_rgb=((updated_frames[0].float()+1)*127.5).clamp(0,255).to(torch.uint8)
                self.last_unresolved_masks=updated_masks[0,:,0].bool()
            elif gpu_input and getattr(self,'measure_propagation',False):
                self.last_unresolved_masks=updated_masks[0,:,0].bool()

        ori_frames = original_gpu if gpu_input else frames_inp
        comp_frames = [None] * video_length
        blend_weights = [0.] * video_length
        feature_blend=gpu_input and (return_features or get_config().propainter_feature_blend)
        from uav.temporal_blend import window_weight

        neighbor_stride = self.neighbor_stride
        neighbor_radius = self.neighbor_length // 2
        if get_config().propainter_full_reference_range:
            ref_num=-1
        elif video_length > self.sub_video_length:
            ref_num = self.sub_video_length // self.ref_stride
        else:
            ref_num = -1

        # ---- feature propagation + transformer ----
        # Reuse only within this propagated clip; other windows have different pixels.
        encoded_features=torch.cat([self.model.encode_frames(
            updated_frames[:,i:i+4],masks_dilated[:,i:i+4],updated_masks[:,i:i+4])
            for i in range(0,video_length,4)],dim=1)
        reference_bank=getattr(self,'reference_bank',None)
        window_centers = list(range(0, video_length, neighbor_stride))
        # A stride wider than the window radius can otherwise leave the final
        # few frames uncovered. Add an explicit tail-centered window only when
        # it is needed.
        if window_centers[-1] + neighbor_radius < video_length - 1:
            window_centers.append(video_length - 1)
        windows=None
        if window_length is not None:
            from uav.temporal_blend import window_ranges
            windows=[list(range(start,end)) for start,end in window_ranges(video_length,window_length,window_overlap)]
            window_centers=[(ids[0]+ids[-1])/2 for ids in windows]
        finite_checks=[]
        transformer_calls=0
        decoded_frame_predictions=0
        for window_index,f in enumerate(window_centers):
            neighbor_ids = windows[window_index] if windows is not None else [
                i for i in range(max(0, f - neighbor_radius),
                                 min(video_length, f + neighbor_radius + 1))
            ]
            output_indices=None if output_range is None else [i for i,idx in enumerate(neighbor_ids)
                if output_range[0]<=idx<output_range[1] and (output_presence is None or output_presence[idx])]
            if output_indices==[]:continue
            external=None if reference_bank is None else reference_bank.get(
                self,self.flow_start+neighbor_ids[0],self.flow_start+neighbor_ids[-1]+1,self.reference_total)
            ref_ids = [] if external is not None else get_ref_index(f, neighbor_ids, video_length, self.ref_stride, ref_num)
            if not ref_ids:
                # Contiguous local windows can be views, avoiding full-resolution
                # advanced-index copies. These inputs are read-only downstream.
                lo,hi=neighbor_ids[0],neighbor_ids[-1]+1
                selected_imgs=updated_frames[:,lo:hi]
                selected_masks=masks_dilated[:,lo:hi]
                selected_update_masks=updated_masks[:,lo:hi]
                selected_pred_flows_bi=None if pred_flows_bi is None else tuple(flow[:,lo:hi-1] for flow in pred_flows_bi)
                selected_encoded=encoded_features[:,lo:hi]
            else:
                selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
                selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
                selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
                selected_pred_flows_bi = None if pred_flows_bi is None else (
                    pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
                selected_encoded=encoded_features[:,neighbor_ids+ref_ids]
            if external is not None:
                selected_encoded=torch.cat((selected_encoded,external),dim=1)

            # 1.0 indicates mask
            l_t = len(neighbor_ids)
            pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t,
                encoded_features=selected_encoded,
                output_indices=output_indices,skip_feature_propagation=transformer_only,return_decoder_features=feature_blend)
            transformer_calls+=1
            if not feature_blend:decoded_frame_predictions+=l_t if output_indices is None else len(output_indices)
            finite=torch.isfinite(pred_img).all()
            finite_checks.append(finite)
            if not feature_blend:
                pred_img = pred_img.view(-1, 3, h, w)
                pred_img = (pred_img + 1) / 2
            if gpu_input:
                output_ids=neighbor_ids if output_indices is None else [neighbor_ids[i] for i in output_indices]
                for i,idx in enumerate(output_ids):
                    weight=window_weight(idx,f,(len(neighbor_ids)-1)/2 if windows is not None else neighbor_radius)
                    value=pred_img[0,i].float() if feature_blend else pred_img[i].float().clamp(0,1)*255
                    comp_frames[idx]=value*weight if comp_frames[idx] is None else comp_frames[idx]+value*weight
                    blend_weights[idx]+=weight
            else:
                pred_img = pred_img.float().clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                          + ori_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)
            # Output images have been copied/merged; do not keep the previous
            # window's predictions and concatenated references during the next call.
            del pred_img, selected_imgs, selected_masks, selected_update_masks
            del selected_pred_flows_bi, selected_encoded
            del external
        if finite_checks and not torch.stack(finite_checks).all():
            raise RuntimeError('ProPainter produced non-finite pixels; refusing to encode a black repair patch')
        if gpu_input:
            for idx,value in enumerate(comp_frames):
                if value is not None:
                    value=value/blend_weights[idx]
                    if return_features:
                        comp_frames[idx]=value.to(encoded_features.dtype)
                        continue
                    if feature_blend:
                        value=self.model.decoder(value.to(encoded_features.dtype)[None]).tanh()[0].float().add(1).mul(127.5)
                        finite_checks.append(torch.isfinite(value).all())
                        decoded_frame_predictions+=1
                    predicted=value.round().clamp_(0,255).to(torch.uint8)
                    comp_frames[idx]=torch.where(masks_dilated[0,idx].bool(),predicted,ori_frames[idx])
            # Feature returns were checked above; only decoding adds new values
            # that need another host-visible finite check.
            if feature_blend and not return_features and finite_checks and not torch.stack(finite_checks).all():
                raise RuntimeError('ProPainter decoder produced non-finite pixels')
            stage_events[4].record()
            requested=range(video_length) if output_range is None else range(*output_range)
            if any(comp_frames[i] is None for i in requested if output_presence is None or output_presence[i]):
                raise RuntimeError('repair windows left requested frames uncovered')
            self.last_work_counts=dict(transformer_calls=transformer_calls,
                decoded_frame_predictions=decoded_frame_predictions)
            self.last_stage_events = stage_events
            result=comp_frames if return_features else [ori_frames[i] if frame is None else frame for i,frame in enumerate(comp_frames)]
            return torch.stack(result) if stack_output else result
        stage_events[4].record()
        stage_events[4].synchronize()
        self.last_timings = {
            "raft_flow_estimation": stage_events[0].elapsed_time(stage_events[1]) / 1000,
            "flow_completion": stage_events[1].elapsed_time(stage_events[2]) / 1000,
            "image_propagation": stage_events[2].elapsed_time(stage_events[3]) / 1000,
            "transformer_inpainting": stage_events[3].elapsed_time(stage_events[4]) / 1000,
        }
        # save videos frame
        comp_frames = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in comp_frames]
        return comp_frames
