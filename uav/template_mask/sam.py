"""Image-only SAM adapter: candidate probabilities and cached original-crop tasks."""
import cv2
import math
from dataclasses import replace
from threading import RLock
import time
import numpy as np
from .geometry import clip_box, support_bounds
from .records import Candidate


def context_crop(box, shape, config):
    """Retain scene context for text-only SAM, including at frame edges.

    SAM needs the scene behind and around a watermark to recognize the concept.
    This crop is independent of SIFT's search region and is never a box prompt.
    Expand first, then translate into the frame instead of clipping off context.
    """
    h, w = shape
    if box is None:
        return (0, 0, w, h)
    l, t, r, b = clip_box(box, shape)
    if r <= l or b <= t:
        return (l, t, r, b)
    side = math.ceil(max(config.sam_context_min_side,
                        max(box[2]-box[0], box[3]-box[1])*config.sam_context_scale))
    width, height = min(w, side), min(h, side)
    x = max(0, min(w-width, math.floor((l+r-width)/2)))
    y = max(0, min(h-height, math.floor((t+b-height)/2)))
    return (x, y, x+width, y+height)


def refinement_config(config, shape):
    """Stage-two native square: 1.7x target, at least 35% of frame short side."""
    return replace(config, sam_context_scale=1.7,
                   sam_context_min_side=math.ceil(.35*min(shape)))


def target_probability(probability, origin, box):
    """Keep target-box probabilities, softly suppress neighbouring watermarks.

    Allow a free margin of 15% of the longer box side (at least two native
    pixels) on EVERY edge. A discovery may contain only one line of a watermark;
    using its short side for this allowance would erase the missing line.
    Beyond that allowance, decay with sigma half the original width/height.
    Apply once to raw cached probabilities, without object-score multiplication.
    """
    l, t, r, b = box
    margin = max(2., .15*max(r-l, b-t))
    x = np.arange(probability.shape[1], dtype=np.float32)+origin[0]
    y = np.arange(probability.shape[0], dtype=np.float32)+origin[1]
    dx = np.maximum(np.maximum(l-margin-x, x-r-margin), 0.)/max(1., (r-l)*.5)
    dy = np.maximum(np.maximum(t-margin-y, y-b-margin), 0.)/max(1., (b-t)*.5)
    weight = np.exp(-.5*(dy[:, None]**2+dx[None, :]**2))
    return np.asarray(probability, np.float32)*weight


def probability_bounds(probability, box, crop, threshold):
    """Keep ALL candidate support, including strokes outside the box head.

    Box prediction is a proposal, not a hard segmentation boundary. Keep a
    small low-probability fringe for interpolation, without storing full frames.
    """
    l, t, r, b = crop
    x, y, z, v = clip_box(box, (b, r), padding=3)
    support = support_bounds(probability > threshold*.5)
    if support is not None:
        x, y = min(x, l+support[0]-2), min(y, t+support[1]-2)
        z, v = max(z, l+support[2]+2), max(v, t+support[3]+2)
    return max(l, x), max(t, y), min(r, z), min(b, v)

class SamDetector:
    def __init__(self, model, cache, config):
        self.model, self.cache, self.config = model, cache, config
        from uav.sam3_inference import prepare_image_inference
        prepare_image_inference(model, real_rope=config.sam3_real_rope,
            compile_trunk=config.sam3_compile, grounding_graph=config.sam3_graph_grounding,
            trt_engine=config.sam3_trt_engine)
        from uav.sam3_resolution import use_configured_rpb_size
        use_configured_rpb_size(model)
        self.processors = {}
        self.inference_lock = RLock()
        self.calls = self.cache_hits = 0
        self.task_calls = dict(global_coarse=0, local=0, refinement=0, boundary_full=0)
        import torch
        self.stream = torch.cuda.Stream(priority=0)
        self.stream.wait_stream(torch.cuda.current_stream())
        self.lock_wait_seconds = 0.
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            self.text = model.backbone.forward_text(['watermark'], device='cuda')
        self.stream.wait_stream(torch.cuda.current_stream())

    def _infer(self, image, resolution, *, boxes_only=False):
        # Only SAM requests share mutable processor/graph state. Tracking uses
        # independent high-priority streams and never acquires this mutex.
        import torch
        tick = time.monotonic()
        with self.inference_lock:
            self.lock_wait_seconds += time.monotonic()-tick
            with torch.cuda.stream(self.stream):
                return self._infer_locked(image,resolution,boxes_only=boxes_only)

    def _infer_locked(self, image, resolution, *, boxes_only=False):
        """One text-only inference, shared by discovery, refinement and presence checks."""
        import torch
        from uav.sam3_gpu_image import GpuImage
        from sam3.model.sam3_image_processor import Sam3Processor
        from uav.sam3_resolution import set_resolution
        from uav.sam3_crop_groups import set_detection_images, segment_group
        h,w = image.shape[:2]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            set_resolution(self.model, resolution)
            if resolution not in self.processors:
                self.processors[resolution] = Sam3Processor(self.model, resolution=resolution,
                    confidence_threshold=self.config.object_threshold)
            processor = self.processors[resolution]
            processor.generate_binary_masks = False
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original = GpuImage(torch.from_numpy(rgb).permute(2,0,1).to(processor.device))
            state = dict(original_height=h,original_width=w,boxes_only=boxes_only,
                         backbone_out=set_detection_images(processor,[original]))
            out = segment_group(processor,original,dict(crop=(0,0,w,h)),self.text,state)
            return (out['scores'].float().cpu().numpy(),
                    out['boxes'].float().cpu().numpy(),
                    None if boxes_only else out['masks_logits'][:,0].float().cpu().numpy())

    def coverage(self, image):
        """Startup box-only measurement; never becomes template evidence."""
        from uav.coverage_preflight import box_coverage_statistics
        _, boxes, _ = self._infer(image, self.config.global_resolution, boxes_only=True)
        self.calls += 1
        self.task_calls['coverage_preflight'] = self.task_calls.get('coverage_preflight', 0)+1
        h, w = image.shape[:2]
        return box_coverage_statistics(boxes, w, h)

    def _probability_arrays(self,image,crop,resolution,local):
        l,t,r,b = crop
        scores,boxes,probabilities = self._infer(image[t:b,l:r],resolution)
        boxes = boxes+[l,t,l,t]
        arrays = dict(scores=scores,boxes=boxes)
        for index,probability in enumerate(probabilities):
            x,y,z,v = (crop if local else
                probability_bounds(probability,boxes[index],crop,self.config.pixel_threshold))
            arrays['p'+str(index)] = probability[y-t:v-t,x-l:z-l].copy()
            arrays['o'+str(index)] = np.array([x,y])
        return arrays

    def prepare_global(self,image,source_id):
        """Producer computes only raw data; no shared cache/counters are changed."""
        h,w=image.shape[:2]
        crop=(0,0,w,h)
        resolution=self.config.global_resolution
        key=self.cache.key('sam-probability-v4',int(source_id),crop,resolution,
                           False,'watermark',self.config.object_threshold)
        return key,self._probability_arrays(image,crop,resolution,False)

    def accept_global(self,prepared):
        """Consumer publishes one reached keyframe into ordinary evidence."""
        key,arrays=prepared
        if key not in self.cache:
            self.cache.put(key,**arrays)
            self.calls+=1
            self.task_calls['global_coarse']+=1

    def detect_presence(self, image, source_id):
        """Full-frame recheck returning boxes, never cached mask evidence."""
        h,w = image.shape[:2]
        global_key = self.cache.key('sam-probability-v4',int(source_id),(0,0,w,h),
            self.config.global_resolution,False,'watermark',self.config.object_threshold)
        if global_key in self.cache:
            self.cache_hits += 1
            data=self.cache.get(global_key)
            return [tuple(b) for b,score in zip(data['boxes'],data['scores'])
                    if score>=self.config.object_threshold]
        key = self.cache.key('sam-presence-v1',int(source_id),self.config.global_resolution)
        if key not in self.cache:
            scores,boxes,_ = self._infer(image,self.config.global_resolution,boxes_only=True)
            self.cache.put(key,boxes=boxes[scores >= self.config.object_threshold])
            self.calls += 1
            self.task_calls.setdefault('boundary_full',0)
            self.task_calls['boundary_full'] += 1
        else:
            self.cache_hits += 1
        return [tuple(b) for b in self.cache.get(key)['boxes']]

    def detect(self, image, source_id, box, resolution, *, refinement=False):
        h, w = image.shape[:2]
        crop = context_crop(box, (h, w),
                            refinement_config(self.config, (h, w)) if refinement else self.config)
        # Global tasks store only support patches; local means need the whole
        # measured crop. Even an identical full-frame crop must not alias them.
        key = self.cache.key('sam-probability-v4', int(source_id), crop, resolution,
                             box is not None, 'watermark', self.config.object_threshold)
        if key not in self.cache:
            l, t, r, b = crop
            if r <= l or b <= t:
                return []
            arrays = self._probability_arrays(image,crop,resolution,box is not None)
            self.cache.put(key, **arrays)
            self.calls += 1
            self.task_calls['refinement' if refinement else 'local' if box is not None else 'global_coarse'] += 1
        else:
            self.cache_hits += 1
        data = self.cache.get(key)
        candidates = []
        for i, (candidate_box, score) in enumerate(zip(data['boxes'], data['scores'])):
            p, origin = data['p'+str(i)], tuple(data['o'+str(i)])
            if score < self.config.object_threshold or not p.size:
                continue
            if box is not None:
                p = target_probability(p, origin, box)
                if not (p > self.config.pixel_threshold).any():
                    continue
            candidates.append(Candidate(p, origin, tuple(candidate_box), float(score), str(i)))
        return candidates
