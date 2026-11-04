"""Rolling final-mask and original-frame handoff to ProPainter."""
from uav.config import get_config
import json,time
import torch
import torch.nn.functional as F
from .gpu_morphology import dilate_review
from .async_video_writer import AsyncVideoWriter
from .sam3_mask_writer import MaskWriter
from .repair_regions import plan_regions
from .repair_blend import RepairBlend,edge_weights


class _RepairSinkState:
    def __init__(self,model,dest,width,height,fps,source_frames,output_frames,writer=None):
        self.model=model;self.dest=dest;self.n=source_frames;self.end=min(source_frames,output_frames)
        self.work_counts=[]
        from .propainter.video.model.modules.sparse_transformer import ffn_compile_profile
        self.ffn_profile_start=ffn_compile_profile()
        from .propainter_references import ReferenceBank
        self.references=ReferenceBank(fps,count=get_config().reference_count,
            seconds=get_config().reference_seconds,
            capacity_bytes=get_config().propainter_reference_cache_mib*2**20,
            host_capacity_bytes=get_config().propainter_reference_host_cache_mib*2**20)
        self.lookahead=self.references.radius if self.references.count else 0
        self.model.reference_bank=self.references if self.references.count else None
        self.model.reference_total=self.n
        self.closed=False
        self.context=0
        self.frames={};self.masks={};self.first=0;self.reports=[];self.events=[];self.max_cached=0
        self.region_mode=True
        self.boxes={};self.previous_crops=[];self.region_reports=[]
        self.writer=writer if writer is not None else AsyncVideoWriter(dest/'repaired.mp4',width,height,fps)
        self.model.debug_propagation=get_config().propainter_debug_stages
        self.model.measure_propagation=get_config().propainter_measure_propagation
        self.propagation_counts=[]
        self.propagation_writer=AsyncVideoWriter(dest/'propagated.mp4',width,height,fps) if self.model.debug_propagation else None
        self.stage_counts=[]
        self.debug=MaskWriter(dest) if get_config().debug else None
        self.seconds=0.;self.wait_seconds=0.;self.changed_outside=torch.zeros((),dtype=torch.long,device='cuda') if self.debug else None
        from .propainter_flow_cache import FlowCache
        self.flow_cache=FlowCache()
        self.model.flow_cache=self.flow_cache
        self.model.fix_raft.reuse_features=True
        self.model.fix_raft.fix_raft.skip_unused_mask=True
        self.model.fix_raft.fix_raft.cache_corr_grid=True

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.first!=self.end:raise RuntimeError('missing masks/context for final repair window')
        finally:
            try:
                self.writer.close()
            finally:
                try:
                    if self.propagation_writer:self.propagation_writer.close()
                finally:
                    try:
                        accumulator=getattr(self,'accumulator',None)
                        if accumulator is not None:accumulator.close()
                    finally:
                        if self.debug:self.debug.close()
                        self.references.clear()
                        self.frames.clear();self.masks.clear()
                        self.model.reference_bank=None
                        self.model.flow_cache=None

    def profile(self):
        from .propainter.video.model.modules.sparse_transformer import ffn_compile_profile
        ffn_profile=ffn_compile_profile()
        for key in ('compiled','fallback'):
            ffn_profile[key]-=self.ffn_profile_start[key]
        replay=getattr(self.model.model.feat_prop_module.forward,'replay',None)
        propagation_graph=None if replay is None else dict(
            captures=replay.captures,replays=replay.replays,fallbacks=replay.fallbacks,signatures=len(replay.cache))
        completion_replay=getattr(self.model.fix_flow_complete.forward,'replay',None)
        completion_graph=None if completion_replay is None else dict(
            captures=completion_replay.captures,replays=completion_replay.replays,
            fallbacks=completion_replay.fallbacks,signatures=len(completion_replay.cache))
        self.result=dict(complete=True,output_frames=self.end,max_cached_frames=self.max_cached,
            repair_host_seconds=self.seconds,windows=self.reports,output=self.writer.profile(),
            changed_pixels_outside_mask_before_encoding=int(self.changed_outside) if self.changed_outside is not None else None,mask_debug=self.debug is not None,
            mask_device='cuda',flow_cache=self.flow_cache.profile() if self.flow_cache else None,
            box_regions=self.region_mode,region_plans=self.region_reports,
            repair_context_ratio=get_config().repair_context_ratio,
            transformer_only=get_config().propainter_transformer_only,
            raft_iterations=self.model.raft_iter,reference_bank=self.references.profile(),
            raft_backend='tensorrt' if hasattr(self.model.fix_raft.fix_raft,'trt_backends') else 'torch',
            raft_trt_calls={name: backend.calls for name, backend in
                            getattr(self.model.fix_raft.fix_raft,'trt_backends',{}).items()},
            raft_scale=get_config().raft_scale,
            flow_completion_scale=get_config().flow_completion_scale,
            debug_stage_counts=self.stage_counts,
            propagation_pixel_counts=self.propagation_counts,
            neighbor_stride=10-self.overlap,temporal_context=0,
            transformer_calls=sum(x['transformer_calls'] for x in self.work_counts),
            decoded_frame_predictions=sum(x['decoded_frame_predictions'] for x in self.work_counts),
            encoder_cache=True,
            encoder_backend='tensorrt' if hasattr(self.model.model.encoder,'trt_backend') else 'torch',
            propagation_backend='tensorrt' if hasattr(self.model.model,'propagation_trt_backends') else 'torch',
            feature_blend=True,
            skip_unused_raft_mask=self.model.fix_raft.fix_raft.skip_unused_mask,
            core_decode=True,
            raft_channels_last=get_config().propainter_raft_channels_last,
            reuse_correlation=get_config().propainter_reuse_correlation,
            masked_decode=True,
            sparse_context=get_config().propainter_sparse_context,
            sdpa=get_config().propainter_sdpa,
            compile_ffn=get_config().propainter_compile_ffn,
            pack_qkv=get_config().propainter_pack_qkv,
            stream_deform=get_config().propainter_stream_deform,
            propagation_graph=propagation_graph,
            completion_graph=completion_graph,
            ffn_compile_profile=ffn_profile,
            cache_warp_grid=True,
            cache_normalizer=True,
            batch_consistency=True,
            compile_raft=get_config().propainter_compile_raft,
            compile_precision_casts=get_config().propainter_compile_precise,
            lean_inference=True,
            reuse_raft_features=self.model.fix_raft.reuse_features,stages={key:sum(events[i].elapsed_time(events[i+1]) for events in self.events)/1000
                for i,key in enumerate(('raft_flow_estimation','flow_completion','image_propagation','transformer_inpainting'))})
        (self.dest/'repair-report.json').write_text(json.dumps(self.result,indent=2))
        return self.result


class FeatureCanvas:
    def __init__(self,device=None):
        self.device=device
        self.bounds=None;self.sums=None;self.weights=None;self.crops=[]
        self.reserved=None

    def reserve(self,crop):
        """Plan allocation bounds before accumulation without allocating tensors."""
        l,t,r,b=crop;box=(l//4,t//4,(r+3)//4,(b+3)//4)
        a=self.reserved
        self.reserved=box if a is None else (min(a[0],box[0]),min(a[1],box[1]),max(a[2],box[2]),max(a[3],box[3]))

    def to(self,device):
        self.device=device
        if self.sums is not None:
            self.sums=self.sums.to(device);self.weights=self.weights.to(device)

    def add(self,feature,crop,temporal):
        if self.device is not None:feature=feature.to(self.device)
        l,t,r,b=crop;box=(l//4,t//4,(r+3)//4,(b+3)//4)
        if self.bounds is None:new=box
        else:
            a=self.bounds;new=(min(a[0],box[0]),min(a[1],box[1]),max(a[2],box[2]),max(a[3],box[3]))
        if self.reserved is not None:
            a=self.reserved;new=(min(a[0],new[0]),min(a[1],new[1]),max(a[2],new[2]),max(a[3],new[3]))
        if new!=self.bounds:
            x,y,z,v=new
            sums=feature.new_zeros((feature.shape[0],v-y,z-x),dtype=torch.float32)
            weights=feature.new_zeros((1,v-y,z-x),dtype=torch.float32)
            if self.bounds is not None:
                a,b_,c,d=self.bounds
                sums[:,b_-y:d-y,a-x:c-x]=self.sums
                weights[:,b_-y:d-y,a-x:c-x]=self.weights
            self.sums=sums;self.weights=weights;self.bounds=new
        x,y,_,_=self.bounds;a,b_,c,d=box
        weight=edge_weights(d-b_,c-a,feature.device)*temporal
        self.sums[:,b_-y:d-y,a-x:c-x].add_(feature[:,:d-b_,:c-a].float()*weight)
        self.weights[:,b_-y:d-y,a-x:c-x].add_(weight)
        if crop not in self.crops:self.crops.append(crop)

    def decode(self,model,original,mask):
        bounds=(min(c[0] for c in self.crops),min(c[1] for c in self.crops),
                max(c[2] for c in self.crops),max(c[3] for c in self.crops))
        blend=RepairBlend({0:original},bounds)
        x,y,_,_=self.bounds
        # Decode existing crop shapes, preserving bounded activations/graph shapes.
        for l,t,r,b in self.crops:
            h,w=b-t,r-l;fh,fw=(h+3)//4,(w+3)//4
            weights=self.weights[:,t//4-y:t//4-y+fh,l//4-x:l//4-x+fw]
            feat=self.sums[:,t//4-y:t//4-y+fh,l//4-x:l//4-x+fw]/weights.clamp_min(1e-6)
            ph=(-(b-t))%8;pw=(-(r-l))%8
            if ph or pw:feat=F.pad(feat,(0,pw//4,0,ph//4),mode='replicate')
            patch=model.decoder(feat.to(device=original.device,dtype=torch.float16)[None]).tanh()[0,:,:h,:w].float().add(1).mul(127.5)
            model._overlap_finite.append(torch.isfinite(patch).all())
            blend.add(0,patch,mask[t:b,l:r][None],(l,t,r,b),edge_weights(h,w,original.device))
        return blend.finish(0,original),len(self.crops)


class RepairSink(_RepairSinkState):
    def __init__(self,*args,feature_device=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.feature_device=feature_device
        self.accumulator=None
        self.context=0;self.pending={};self.emitted=0;self.pending_peak_bytes=0
        self.overlap=get_config().block_overlap
        if self.overlap not in (3,4,5):raise ValueError('overlap must be 3, 4 or 5')
        self.block=30-2*self.overlap;self.step=self.block-self.overlap

    def save(self,frame,pred,rgb,*,mask_is_final=False,boxes=None):
        from .gpu_morphology import dilate_review
        if boxes is None:raise ValueError('overlap repair requires boxes')
        self.boxes[frame]=boxes;self.frames[frame]=rgb
        self.masks[frame]=pred if mask_is_final else dilate_review(pred)
        self.references.add(frame,rgb,self.masks[frame])
        if self.debug:self.debug.save(frame,self.masks[frame])
        self.max_cached=max(self.max_cached,len(self.frames))
        while self.first<self.end:
            last=min(self.end,self.first+self.block)
            if not all(i in self.frames for i in range(self.first,min(self.n,last+self.lookahead))):break
            self._overlap(last)

    @torch.inference_mode()
    def _overlap(self,last):
        with torch.autocast('cuda',enabled=False):
            start=time.monotonic();first=self.first;final=last==self.end
            flush=last if final else first+self.step
            # Current-window hits remain readable; only cache newly computed
            # pairs that the next window can request. Earlier pairs are final.
            self.flow_cache.keep_from=flush
            h,w=self.frames[first].shape[-2:]
            regions=plan_regions([b for i in range(first,last) for b in self.boxes[i]],w,h,self.previous_crops)
            # Align to a global feature lattice, without resampling.
            crops=list(dict.fromkeys((max(0,(a['crop'][0]//8)*8),max(0,(a['crop'][1]//8)*8),min(w,((a['crop'][2]+7)//8)*8),min(h,((a['crop'][3]+7)//8)*8)) for a in regions))
            if self.feature_device=='auto' and crops:
                # Each frame owns a dense FP32 canvas with 128 channels and
                # one weight channel. Decide before allocating model activations.
                bounds=[(l//4,t//4,(r+3)//4,(b+3)//4) for l,t,r,b in crops]
                bounds.extend(v.bounds for v in self.pending.values() if v.bounds is not None)
                area=(max(b[2] for b in bounds)-min(b[0] for b in bounds))*(max(b[3] for b in bounds)-min(b[1] for b in bounds))
                if area*129*4*(last-first)>256*2**20:
                    self.feature_device='cpu'
                    for canvas in self.pending.values():canvas.to('cpu')
            if self.feature_device=='cpu' and self.accumulator is None:
                from .repair_accumulator import FeatureAccumulator
                self.accumulator=FeatureAccumulator()
            self.previous_crops=list(crops)
            masks=torch.stack([self.masks[i] for i in range(first,last)])
            present=torch.stack([masks[:,t:b,l:r].flatten(1).any(1) for l,t,r,b in crops]).cpu().tolist() if crops else []
            device=None if self.feature_device=='auto' else self.feature_device
            for crop,active in zip(crops,present):
                for j,exists in enumerate(active):
                    if exists:self.pending.setdefault(first+j,FeatureCanvas(device)).reserve(crop)
            calls=0
            for crop,active in zip(crops,present):
                if not any(active):continue
                l,t,r,b=crop;ph=(-(b-t))%8;pw=(-(r-l))%8
                clips=torch.stack([self.frames[i][:,t:b,l:r] for i in range(first,last)])
                local=masks[:,t:b,l:r]
                if ph or pw:
                    clips=F.pad(clips,(0,pw,0,ph),mode='replicate')
                    local=F.pad(local,(0,pw,0,ph))
                self.model.flow_start=first;self.model.flow_crop=(*crop,ph,pw)
                features=self.model.inpaint(clips,local,output_range=(0,last-first),output_presence=active,stack_output=False,return_features=True,window_length=10,window_overlap=self.overlap)
                self.events.append(self.model.last_stage_events);self.work_counts.append(dict(self.model.last_work_counts));calls+=1
                for j,feat in enumerate(features):
                    if feat is None:continue
                    i=first+j;weight=1.
                    if first and j<self.overlap:weight=(j+1)/(self.overlap+1)
                    if not final and i>=flush:weight=(last-i)/(self.overlap+1)
                    device=None if self.feature_device=='auto' else self.feature_device
                    canvas=self.pending.setdefault(i,FeatureCanvas(device))
                    if self.accumulator is None:canvas.add(feat,crop,weight)
                    else:self.accumulator.add(canvas,feat,crop,weight)
                del features,clips,local
            if self.accumulator is not None:self.accumulator.finish()
            self.pending_peak_bytes=max(self.pending_peak_bytes,sum(v.sums.numel()*4+v.weights.numel()*4 for v in self.pending.values()))
            del masks
            decoded=0
            self.model.model._overlap_finite=[]
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            for i in range(self.emitted,flush):
                canvas=self.pending.pop(i,None)
                if canvas is None:result=self.frames[i]
                else:
                    result,count=canvas.decode(self.model.model,self.frames[i],self.masks[i]);decoded+=count
                if self.changed_outside is not None:self.changed_outside+=((result!=self.frames[i]).any(0)&~self.masks[i]).sum()
                self.writer.save(result.permute(1,2,0).contiguous())
                del canvas,result
            if self.model.model._overlap_finite and not torch.stack(self.model.model._overlap_finite).all():raise RuntimeError('non-finite overlap decoder output')
            self.model.model._overlap_finite=[]
            end.record();self.decode_events=getattr(self,'decode_events',[])+[(begin,end)]
            self.work_counts.append(dict(transformer_calls=0,decoded_frame_predictions=decoded))
            self.emitted=flush;self.first=self.end if final else flush
            keep=self.first
            # Retain only references the upcoming rolling schedule may request.
            required=set()
            from .temporal_blend import window_ranges
            for position in range(keep,min(self.end,keep+self.references.radius+1),self.step):
                for a,b in window_ranges(min(self.block,self.end-position),10,self.overlap):
                    required.update(i for i in self.references.select(position+a,position+b,self.n) if i<keep)
            self.references.prune(keep,required)
            if self.flow_cache:self.flow_cache.keep_from=keep;self.flow_cache.prune(keep)
            self.frames={i:f for i,f in self.frames.items() if i>=keep};self.masks={i:m for i,m in self.masks.items() if i>=keep};self.boxes={i:b for i,b in self.boxes.items() if i>=keep}
            self.reports.append(dict(first=first,last_exclusive=last,flushed_until=flush,calls=calls,host_seconds=time.monotonic()-start))
            self.seconds+=time.monotonic()-start
            print(__import__('json').dumps(dict(repair_window=self.reports[-1])),flush=True)

    def profile(self):
        result=super().profile()
        result.update(outer_block=self.block,outer_stride=self.step,inner_window=10,inner_stride=10-self.overlap,pending_feature_peak_bytes=self.pending_peak_bytes,pending_features='cpu_memory' if self.feature_device=='cpu' else 'gpu_memory',final_decode_seconds=sum(a.elapsed_time(b) for a,b in getattr(self,'decode_events',[]))/1000)
        (self.dest/'repair-report.json').write_text(__import__('json').dumps(result,indent=2))
        return result
