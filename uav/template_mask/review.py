"""User-facing review assets. Generation does not require agent visual inspection."""
from pathlib import Path
import colorsys
import html
import json
import cv2
import numpy as np
from .geometry import bounds, points, translation, warp
from .sam import context_crop, refinement_config
from .output_geometry import selected_matrix
from .trajectory_quality import observed_only
from .schedule import is_discovery_frame


def capture_first_frame(engine, frame):
    """Freeze stage1 debug labels; stage2 must not rewrite this snapshot."""
    if not engine.debug:
        return
    engine.first_pass_observations[frame] = tuple(engine.frame_observations.get(frame, ()))
    marks = []
    for instance in engine.instances.values():
        if not engine._renders_at(instance, frame):
            continue
        pose = instance.poses.get(frame)
        if pose is None or pose.status == 'absent':
            continue
        template = engine.bank.templates[instance.template_id]
        marks.append(dict(instance=instance.id, template=template.id,
            box=list(bounds(pose.matrix, template.box)),
            anchor=points(pose.matrix, [template.anchor])[0].tolist(),
            status=pose.status, presence=pose.presence, reason=pose.reason,
            output_margin=engine.presence.output_margin(instance, frame)))
    engine.first_pass_marks[frame] = marks


def final_marks(engine, frame):
    """Repaired occurrence IDs and canonical IDs of coordinate-merged templates."""
    marks = []
    plan = getattr(engine,'output_plan',None)
    owners = (engine.instances[i] for i in plan.frame_owners.get(frame,())) if plan is not None else engine.instances.values()
    canonical = {i:min(g) for g in plan.occurrences for i in g} if plan is not None else {}
    for instance in owners:
        if plan is None and not engine._renders_at(instance, frame):
            continue
        pose = instance.poses.get(frame)
        matrix = selected_matrix(engine,instance,frame)
        if matrix is None:
            continue
        if observed_only(instance,frame) and not any(
                engine.bank.observations[oid].instance_id == instance.id
                for oid in engine.frame_observations.get(frame, ())):
            continue
        owner = engine.instances.get(instance.alias_of, instance)
        iid = owner.continuation_of if owner.continuation_of is not None else owner.id
        iid = canonical.get(instance.id,iid)
        members = engine.bank.members(instance.template_id)
        boxes = [bounds(matrix @ transform, engine.bank.templates[tid].box)
                 for tid, transform in members.items()]
        box = np.r_[np.min([b[:2] for b in boxes], axis=0),
                    np.max([b[2:] for b in boxes], axis=0)]
        marks.append(dict(instance=iid, template=min(members),
            status=pose.status if pose is not None else 'measured', box=box,
            anchor=points(matrix,[display_anchor(engine,instance.template_id)])[0].tolist()))
    return marks

def display_anchor(engine, template_id):
    """One final review point shared by all gauges of a merged template.

    Tracking anchors remain immutable. Review uses the final template extent
    in canonical coordinates, so refinement can shrink a seed without leaving
    its visible trajectory attached to the discarded seed-box centre.
    """
    members = engine.bank.members(template_id)
    root = min(members)
    canonical = engine.bank.members(root)
    boxes = [bounds(transform,engine.bank.templates[tid].box)
             for tid,transform in canonical.items()]
    low = np.min([box[:2] for box in boxes],axis=0)
    high = np.max([box[2:] for box in boxes],axis=0)
    return points(members[root],[(low+high)*.5])[0]


def stage1_mask(engine, frame):
    """Hold a global SAM result until the next scheduled sample, debug only."""
    sample = frame-frame % engine.config.global_stride
    if is_discovery_frame(frame,engine.config.global_stride,getattr(engine,'frame_count',None)):
        sample = frame
    source = engine.source_ids.get(sample)
    mask = np.zeros(engine.shape, bool)
    if source in engine.discovery_groups:
        for group in engine.discovery_groups[source]['groups']:
            for candidate in group['sources']:
                key = engine.cache.key('raw_discovery', source, candidate)
                if key in engine.cache:
                    data = engine.cache.get(key)
                    mask |= warp((data['probability'] > engine.config.pixel_threshold) & data['valid'],
                        translation(*data['origin']), engine.shape, nearest=True) > 0
        return mask
    for oid in engine.first_pass_observations.get(sample, ()):
        obs = engine.bank.observations[oid]
        if obs.mask_role == 'discovery':
            mask |= engine._observed_candidate(obs)
    return mask


def trajectory_segments(engine):
    """Final display anchors; no path across absent/non-rendering intervals."""
    paths = {}
    plan = getattr(engine,'output_plan',None)
    for instance in engine.instances.values():
        anchor = display_anchor(engine,instance.template_id)
        segments, current = [], []
        frames = (sorted(f for f,ids in plan.frame_owners.items() if instance.id in ids)
                  if plan is not None else sorted(instance.poses))
        for frame in frames:
            matrix = selected_matrix(engine,instance,frame)
            visible = matrix is not None and (plan is not None or engine._renders_at(instance,frame))
            if observed_only(instance,frame):
                visible = visible and any(engine.bank.observations[oid].instance_id == instance.id
                    for oid in engine.frame_observations.get(frame, ()))
            if not visible or current and frame != current[-1][0]+1:
                if current:
                    segments.append(current)
                    current = []
            if visible:
                center = points(matrix, [anchor])[0]
                if np.isfinite(center).all():
                    current.append((frame, float(center[0]), float(center[1])))
                elif current:
                    segments.append(current)
                    current = []
        if current:
            segments.append(current)
        paths[instance.id] = segments
    return paths


def first_pass_trajectories(engine):
    paths, current = {}, {}
    for frame, marks in sorted(engine.first_pass_marks.items()):
        for mark in marks:
            iid = mark['instance']
            samples = current.get(iid)
            if samples is None or samples[-1][0]+1 != frame:
                samples = []
                paths.setdefault(iid, []).append(samples)
                current[iid] = samples
            samples.append((frame, *mark['anchor']))
    return paths


def color(identity):
    return tuple(int(x*255) for x in colorsys.hsv_to_rgb((identity*.61803398875)%1, .65, 1))[::-1]


def uncertainty_banner(metrics):
    rows = metrics.get('uncertainty', [])
    restrictions = (['SUSPECTED SCENE / OBSERVED PIXELS ONLY']
                    if metrics.get('trajectory_restrictions') else [])
    if metrics.get('coarse_fallback'):
        return [*restrictions, 'COARSE MASK FALLBACK / 2x DILATION', *dict.fromkeys(r['reason'] for r in rows)]
    if not rows:
        return restrictions
    radii = [r['extra_radius'] for r in rows if r.get('extra_radius') is not None]
    margin = f' extra={min(radii)}-{max(radii)}px' if radii else ''
    return [*restrictions, 'CONSERVATIVE COVER'+margin, *dict.fromkeys(r['reason'] for r in rows)]


def review_layout(shape):
    """Preserve native pixels; stack landscape vertically, portrait horizontally."""
    h, w = shape
    vertical = w >= h
    # H.264 yuv420 requires even dimensions; pad at most one row/column.
    pw, ph = w+w%2, h+h%2
    return dict(axis=0 if vertical else 1, width=pw if vertical else pw*2,
                height=ph*2 if vertical else ph, panel_width=pw, panel_height=ph,
                first='上方' if vertical else '左侧', final='下方' if vertical else '右侧',
                native_width=w, native_height=h, box_thickness=1, crf=18)


class MaskOverlay:
    """Blend a mask's occupied region with OpenCV, without pixel index lists."""
    def __init__(self,mask,color,opacity,*,background_weight=None,rounding=True):
        x,y,w,h=cv2.boundingRect(mask.view(np.uint8))
        self.box=x,y,w,h
        self.mask=mask[y:y+h,x:x+w].view(np.uint8)
        background_weight=1-opacity if background_weight is None else background_weight
        value=np.arange(256)[:,None]*background_weight+np.array(color)*opacity
        self.lookup=(np.rint(value) if rounding else value).astype(np.uint8).reshape(256,1,3)

    def draw(self,panel,*,source=None):
        x,y,w,h=self.box
        if not w:
            return
        source=panel if source is None else source
        cv2.copyTo(cv2.LUT(source[y:y+h,x:x+w],self.lookup),
                   self.mask,panel[y:y+h,x:x+w])


def write_review(engine, frames, output, skipped=False, *, filename='mask-review.mp4',
                 write_templates=True):
    h, w = engine.shape
    from .review_trajectory import PreparedTrajectory
    from .review_video import review_writer
    paths = trajectory_segments(engine)
    first_paths = first_pass_trajectories(engine)
    prepared = []
    for index,panel_paths in enumerate((first_paths,paths)):
        strokes = []
        for iid,segments in panel_paths.items():
            owner = engine.instances[iid]
            canonical = engine.instances.get(owner.alias_of,owner) if index == 1 else owner
            path_id = (canonical.continuation_of if index == 1 and canonical.continuation_of is not None
                       else canonical.id)
            strokes.extend(PreparedTrajectory(engine.shape,samples,color(path_id)) for samples in segments)
        prepared.append(strokes)
    layout = review_layout(engine.shape)
    pw, ph = layout['panel_width'], layout['panel_height']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    canvas=np.zeros((layout['height'],layout['width'],3),np.uint8)
    panels=([canvas[:ph],canvas[ph:]] if layout['axis']==0
            else [canvas[:,:pw],canvas[:,pw:]])
    sample=None
    with review_writer(output,filename,layout,frames.fps) as combined:
        for frame in range(frames.count):
            current_sample=frame-frame % engine.config.global_stride
            if is_discovery_frame(frame,engine.config.global_stride,getattr(engine,'frame_count',None)):
                current_sample=frame
            if current_sample != sample:
                first = stage1_mask(engine, frame)
                first_overlay=MaskOverlay(first,[140,240,40],.35)
                sample=current_sample
            final_key = engine.cache.key('final_mask', frame)
            if final_key in engine.cache:
                final = np.unpackbits(engine.cache.get(final_key)['packed'], count=h*w).reshape(h, w).astype(bool)
            else:
                # Diagnostic rendering keeps failure labels visible; it never
                # changes tracking, ownership or template evidence.
                final = engine.render(frame, diagnostic=True)
            radii = engine.render_metrics.get(frame, {}).get('dilation_radii', [])
            dilation_label = f' r={min(radii)}-{max(radii)}px' if radii else ' r=0px'
            original = frames(frame)
            for index, (mask, title) in enumerate(((first, 'STAGE 1 / SAM + OBJECTS AND TRACKS'),
                                (final, ('STAGE 2 / SKIPPED DIAGNOSTIC' if skipped else 'STAGE 2 / FINAL MASK')+dilation_label))):
                # Orange denotes the FULL conservative contribution, including
                # overlap with reliable coverage; it is not just the added rim.
                panel = panels[index][:h,:w]
                panel[:]=original
                if mask is not None:
                    overlay=(first_overlay if index==0 else MaskOverlay(mask,[140,240,40],.35))
                    overlay.draw(panel)
                conservative_key = engine.cache.key('conservative_mask', frame)
                if index == 1 and conservative_key in engine.cache:
                    conservative = np.unpackbits(engine.cache.get(conservative_key)['packed'],
                        count=h*w).reshape(h, w).astype(bool) & mask
                    # The conservative color starts from the original pixels,
                    # not the normal overlay already composited in this panel.
                    overlay=MaskOverlay(conservative,[30,150,255],.55,
                                        background_weight=.45,rounding=False)
                    overlay.draw(panel,source=original)
                for trajectory in prepared[index]:
                    trajectory.draw(panel)
                if index == 0:
                    marks = engine.first_pass_marks.get(frame, ())
                else:
                    marks = final_marks(engine, frame)
                for mark in marks:
                    l, t, r, b = map(int, mark['box'])
                    shade = color(mark['instance'])
                    cv2.rectangle(panel, (l, t), (r, b), shade, layout['box_thickness'])
                    cv2.putText(panel, f"I{mark['instance']}/T{mark['template']} {mark['status']}",
                        (max(0, l), max(42, t-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, shade, 1)
                messages = uncertainty_banner(engine.render_metrics.get(frame, {})) if index == 1 else []
                cv2.rectangle(panel, (0, 0), (pw, 44+16*min(3, len(messages))), (0, 0, 0), -1)
                cv2.putText(panel, title+f' f={frame}', (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                cv2.putText(panel, 'DASHED: FULL PATH',
                    (5, 36), cv2.FONT_HERSHEY_SIMPLEX, .38, (220, 220, 220), 1)
                for line, message in enumerate(messages[:3]):
                    cv2.putText(panel, message, (5, 52+line*16), cv2.FONT_HERSHEY_SIMPLEX, .38, (30, 170, 255), 1)
            combined.write(memoryview(canvas).cast('B'))
            if frame % engine.config.global_stride == 0:
                print(json.dumps(dict(event='template_review', frame=frame, frames=frames.count, skipped=skipped)), flush=True)
    templates = write_template_overlays(engine, frames, output) if write_templates else []
    write_page(output, layout, templates, skipped, filename=filename)
    return dict(layout, path=filename, html='index.html',
                first_panel='raw_sam_objects_and_tracks', sam_hold_frames=engine.config.global_stride,
                templates='templates.json', frames=frames.count,
                trajectories='dashed_full_path', trajectory_anchor='final_template_extent_center')


def write_template_overlays(engine, frames, output):
    """Project each final shared mask onto one of its actual original crops."""
    e = engine
    templates, processed = [], set()
    for tid in sorted(e.bank.templates):
        if tid in processed:
            continue
        members = e.bank.members(tid)
        processed.update(members)
        row = dict(id=tid, members=sorted(members), path=None, sources=[])
        if getattr(e.bank.templates[tid],'composite',False):
            row['reason']='复合观测已由独立运动证据拆分，不作为共享模板输出'
            templates.append(row)
            continue
        try:
            data, coarse = e.bank.output_fusion(tid)
            selected = e.bank.selected_views(tid)
            if coarse:
                selected = {}
                for member, transform in members.items():
                    for oid in e.bank.templates[member].observations:
                        obs = e.bank.observations[oid]
                        if (obs.mask_role == 'discovery' and obs.reliable and not obs.ambiguous
                                and obs.source_id in data['sources']):
                            selected.setdefault(obs.source_id, []).append((obs, transform))
            jobs = [(o, transform) for source, views in selected.items()
                    if source in data['sources'] for o, transform in views]
            if not jobs:
                raise ValueError('no_original_for_template_overlay')
            obs, transform = max(jobs, key=lambda item: (item[0].score, -item[0].frame))
            matrix = obs.pose @ np.linalg.inv(transform)
            x, y = data['origin']
            height, width = data['mean'].shape
            target = bounds(matrix, (x, y, x+width, y+height))
            crop = context_crop(target, e.shape, refinement_config(e.config, e.shape))
            l, t, r, b = crop
            original = frames(obs.frame)[t:b, l:r].copy()
            if not original.size:
                raise ValueError('empty_template_crop')
            mask = warp(data['mean'] > e.config.pixel_threshold,
                        translation(-l, -t) @ matrix @ translation(*data['origin']),
                        original.shape[:2], nearest=True) > 0
            overlay = original.copy()
            overlay[mask] = np.rint(original[mask]*.55+np.array([60, 220, 70])*.45).astype(np.uint8)
            name = f'template-{tid}.png'
            if not cv2.imwrite(str(output/name), overlay):
                raise RuntimeError('template overlay write failed')
            row.update(path=name, origin=data['origin'].tolist(), crop=list(crop),
                original_frame=obs.frame, original_source=obs.source_id,
                original_observation=obs.id, mask_pixels=int(mask.sum()),
                sources=[dict(observation=o.id, source_frame=o.source_id, resolution=o.resolution)
                         for o, _ in jobs], mask_sources=data['sources'].tolist(),
                fusion='rolling_reference_mean', refined=bool(data['refined']),
                coarse_fallback=coarse, overlay='final_binary_template_on_original_crop')
        except ValueError as error:
            row['reason'] = str(error)
        templates.append(row)
    (output/'templates.json').write_text(json.dumps(templates, indent=2))
    return templates


def write_page(output, layout, templates, skipped, *, filename='mask-review.mp4'):
    """A self-contained review page using only relative asset links."""
    status = '诊断结果：最终输出未通过' if skipped else '处理完成，待视觉审核'
    pieces = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>UAV debug</title><style>body{margin:20px auto;padding:0 16px;max-width:1600px;',
        'font-family:system-ui;background:#151515;color:#eee}a{color:#9cd3ff}',
        'video,img{max-width:100%;height:auto}video{width:100%}figure{margin:0}',
        '.templates{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}',
        '</style>',
        '<h1>UAV debug</h1><p>'+status+'</p>',
        '<p>Stage1：原始 SAM3、框、对象/模板与初步轨迹。Stage2：最终 mask、修正后的框/对象/轨迹与合并后的模板 ID。</p>',
        '<p>Stage1 的 SAM3 检测保持到下一次全局检测。虚线：完整轨迹。橙色：保守覆盖。</p>',
        '<p>'+layout['first']+'：Stage1；'+layout['final']+'：Stage2。</p>',
        '<p><a href="report.json">数值报告</a></p>',
        '<video controls playsinline preload="none" src="'+html.escape(filename,quote=True)+'"></video>',
        '<h2>模板 mask</h2><div class="templates">']
    for row in templates:
        ids = ', '.join('T'+str(t) for t in row['members'])
        pieces.append('<figure><h3>'+html.escape(ids)+'</h3>')
        if row['path']:
            label = f"原始帧 {row['original_source']}"+(' · 粗 mask 回退' if row['coarse_fallback'] else '')
            pieces.append('<p>'+label+'</p><a target="_blank" href="'+html.escape(row['path'],quote=True)+'"><img loading="lazy" src="'+html.escape(row['path'],quote=True)+'"></a>')
        else:
            pieces.append('<p>'+html.escape(row.get('reason', '无可用原图'))+'</p>')
        pieces.append('</figure>')
    pieces.append('</div></html>')
    (output/'index.html').write_text(''.join(pieces))
