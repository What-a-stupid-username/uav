"""Shared-template mask pipeline with shared-frame ProPainter repair."""
from contextlib import ExitStack
from pathlib import Path
import json
import resource
import time
import os
import cv2
import numpy as np
from .cache import EvidenceCache
from .config import TemplateConfig
from .engine import TemplateEngine
from .policy import SkipVideo, WarningLog
from .schedule import is_discovery_frame
from .frames import VideoFrames


def run(source, output, config=None, *, max_seconds=None, start_source_frame=0,
        model=None, review=True, stage1_checkpoint=None, refresh_stage1=False,
        repair=False, repair_model=None, write_masks=True, random_start=False, seed=20261019,
        max_watermark_coverage=None, max_single_watermark_coverage=None,
        frames=None, session=None, writer=None, mask_consumer=None):
    """Generate final masks, optionally export/debug, and hand memory to repair."""
    from .sam import SamDetector
    if refresh_stage1 and stage1_checkpoint is None:
        raise ValueError('refresh_stage1 requires an explicit stage1_checkpoint path')
    config = config or TemplateConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.json').exists():
        raise ValueError('use a fresh output directory for each evaluation')
    warning_log = WarningLog(output/'warnings.jsonl')
    cache = EvidenceCache()
    resources = ExitStack()
    resources.callback(cache.close)
    supplied_frames = frames
    frames = None
    engine = None
    detector = None
    start = time.monotonic()
    previous_threads = cv2.getNumThreads()
    resources.callback(cv2.setNumThreads, previous_threads)
    cv2.setNumThreads(min(8,len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1))
    report = dict(engine='template', complete=False, skipped=False)
    stage = 'initialization'
    stage_start = start
    stages = {}
    def enter(name):
        nonlocal stage, stage_start
        now = time.monotonic()
        stages[stage] = stages.get(stage, 0.)+now-stage_start
        stage, stage_start = name, now
        print(json.dumps(dict(event='template_stage', stage=name)), flush=True)
    try:
        if session is not None:
            model = session.sam()
        if model is None:
            import torch
            from uav.sam3_model import build
            from uav.inference_weights import compact_weights
            torch.set_num_threads(4)
            with torch.inference_mode():
                model = build()
                compact_weights(model.backbone)
        detector = (session.detection(cache,config) if session is not None
                    else SamDetector(model, cache, config))
        if any(limit is not None and limit < 1 for limit in
               (max_watermark_coverage, max_single_watermark_coverage)):
            enter('coverage_preflight')
            from uav.coverage_preflight import check_coverage
            preflight = check_coverage(source, max_watermark_coverage, seed, detector.coverage,
                                      single_threshold=max_single_watermark_coverage)
            report['coverage_preflight'] = preflight
            print(json.dumps(dict(event='coverage_preflight', **preflight)), flush=True)
            if preflight['skipped']:
                for reason in preflight['skip_reasons']:
                    warning_log.add(-1, -1, -1, reason, 'skip_video')
                raise SkipVideo(preflight['skip_reasons'][0])
        if supplied_frames is None:
            frames = VideoFrames(source, config.fps, start_source_frame, max_seconds,
                                 random_start=random_start, seed=seed)
            resources.callback(frames.close)
        else:
            frames = supplied_frames
        start_source_frame = frames.start
        image = frames(0)
        engine = TemplateEngine(config, cache, detector, warning_log, image.shape[:2], debug=review)
        engine.frame_count = frames.count
        resources.callback(engine.matcher.close)
        checkpoint_signature = None
        if stage1_checkpoint is not None:
            from . import checkpoint
            stage1_checkpoint = Path(stage1_checkpoint)
            checkpoint_signature = checkpoint.signature(source,frames,config,engine)
        if stage1_checkpoint is not None and stage1_checkpoint.exists() and not refresh_stage1:
            enter('stage1_checkpoint_load')
            report['stage1_checkpoint'] = checkpoint.load(stage1_checkpoint,engine,checkpoint_signature)
            report['evidence_reuse'] = True
            print(json.dumps(dict(event='template_stage1_checkpoint',**report['stage1_checkpoint'])),flush=True)
        else:
            enter('first_pass')
            from .prefetch import prefetch_discoveries
            with prefetch_discoveries(engine,frames):
                for frame in range(frames.count):
                    engine.first_frame(frame, frames.source_id(frame), frames(frame), image_at=frames)
                    if is_discovery_frame(frame,config.global_stride,frames.count):
                        print(json.dumps(dict(event='template_first_pass', frame=frame, frames=frames.count,
                            templates=len(engine.bank.templates), instances=len(engine.instances),
                            hypotheses=[dict(template=t.id, reason=t.investigation_reason,
                                observations=len(t.observations), provisional=t.provisional)
                                for t in engine.bank.templates.values()],
                            links=[dict(a=l.a, b=l.b, active=l.active, sources=len(l.evidence)) for l in engine.bank.links.values()],
                            aliases={i.id:i.alias_of for i in engine.instances.values() if i.alias_of is not None})), flush=True)
            if stage1_checkpoint is not None:
                enter('stage1_checkpoint_save')
                report['stage1_checkpoint'] = checkpoint.save(stage1_checkpoint,engine,checkpoint_signature)
                print(json.dumps(dict(event='template_stage1_checkpoint',**report['stage1_checkpoint'])),flush=True)
        enter('second_pass')
        engine.second_pass(frames)
        enter('final_validation')
        rendered = 0
        # Complete the policy gate before handing anything to an inpainting consumer.
        for frame in range(frames.count):
            mask = engine.render(frame)
            cache.put(cache.key('final_mask', frame), packed=np.packbits(mask))
            rendered += 1
        if mask_consumer is not None:
            for frame in range(frames.count):
                packed = cache.get(cache.key('final_mask', frame))['packed']
                mask_consumer(frame,np.unpackbits(packed,count=np.prod(engine.shape)).reshape(engine.shape).astype(bool))
        if write_masks:
            enter('mask_output')
            mask_dir = output/'masks'; mask_dir.mkdir(exist_ok=True)
            for frame in range(frames.count):
                mask = np.unpackbits(cache.get(cache.key('final_mask', frame))['packed'],
                                     count=np.prod(engine.shape)).reshape(engine.shape).astype(np.uint8)*255
                if not cv2.imwrite(str(mask_dir/f'{frame:06d}.png'), mask):
                    raise RuntimeError('mask write failed')
        report.update(rendered_frames=rendered)
        if review:
            enter('review')
            from .review import write_review
            report['review'] = write_review(engine, frames, output)
        if repair:
            # Keep original decoded frames alive through the repair handoff.
            # Release matching buffers before loading the inpainting models.
            engine.matcher.close()
            enter('repair_initialization')
            if session is not None:
                repair_model = session.repair()
            if repair_model is None:
                import torch
                replay = getattr(model.forward_grounding,'replay',None)
                if replay is not None:
                    replay.cache.clear()
                model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                from uav.repair_model import build_repair_model
                repair_model = build_repair_model()
            enter('repair')
            from .repair import repair_frames
            def final_mask(frame):
                packed = cache.get(cache.key('final_mask',frame))['packed']
                return np.unpackbits(packed,count=np.prod(engine.shape)).reshape(engine.shape).astype(bool)
            report['repair'] = repair_frames(frames,final_mask,output,repair_model,
                                               **({} if writer is None else dict(writer=writer)))
        report['complete'] = True
    except SkipVideo as error:
        report.update(complete=True, skipped=True, reason=str(error))
        # A diagnostic preview remains useful for user assessment of a skipped run.
        if review and engine is not None:
            from .review import write_review
            enter('skipped_review')
            try:
                report['review'] = write_review(engine, frames, output, skipped=True)
            except Exception as review_error:
                report.update(review_error=type(review_error).__name__+': '+str(review_error))
                print(json.dumps(dict(event='template_review_error', error=report['review_error'])), flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(complete=False, error=type(error).__name__+': '+str(error))
        raise
    finally:
        try:
            stages[stage] = stages.get(stage, 0.)+time.monotonic()-stage_start
            if engine is not None:
                report.update(engine.report())
            if detector is not None:
                report['sam'] = dict(calls=detector.calls, cache_hits=detector.cache_hits,
                                     task_calls=getattr(detector, 'task_calls', {}))
            report.update(source=str(source), start_source_frame=start_source_frame,
                          stage_seconds=stages,
                          total_seconds=time.monotonic()-start,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                          warnings=warning_log.rows)
            if frames is not None:
                report.update(frames=frames.count, fps=frames.fps, source_fps=frames.source_fps)
                if isinstance(getattr(frames,'decode_seconds',None),(float,int)):
                    report['frame_memory'] = frames.report()
            warning_log.flush()
            (output/'report.json').write_text(json.dumps(report, indent=2))
        finally:
            resources.close()
    return report
