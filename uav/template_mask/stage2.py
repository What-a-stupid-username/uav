"""Stage2: resolve identities, repair affected geometry, refine shared masks.

No reverse timeline pass. Image work is restricted to explicit sparse jobs.
Fine segmentation never supplies new motion measurements.
"""
import time


def repair_geometry(engine,image_at,affected,previous=None):
    e = engine
    jobs = [job for job in e._unresolved_pose_jobs(affected)
            if previous is None or job not in previous
            or e.instances[job[1]].template_id in affected]
    for frame,iid in jobs:
        e._track(e.instances[iid],frame,e.source_ids[frame],image_at(frame),
                 backward=True,image_only=True,boundary=e.presence.covers(e.instances[iid],frame))
    if previous is not None:previous.update(jobs)
    return len(jobs)


def run_stage2(engine,image_at):
    from .composite import review_composite_templates
    from .refinement import refine_templates, review_refinement_ownership
    from .output_plan import build_output_plan
    e = engine
    started = time.monotonic()
    e.sam_stages['first_pass'] = getattr(e.detector,'calls',0)-e._initial_sam_calls
    e.sam_stages['first_pass_tasks'] = {k:v-e._initial_sam_tasks.get(k,0)
        for k,v in getattr(e.detector,'task_calls',{}).items()}
    e.stage = 'second_pass'
    e.stage2_phases = {}
    def phase(name,operation):
        tick = time.monotonic()
        result = operation()
        e.stage2_phases[name] = time.monotonic()-tick
        return result

    def resolve():
        for instance in e.active_instances():e._motion(instance)
        e.pending.retry()
        e._verify_links()
        review_composite_templates(e)
        e.presence.resolve(image_at)
        e.reconciler.trajectory_identities()
        e.interactions.resolve()
        return e.reconciler.global_review()

    affected = phase('association',resolve)
    phase('trajectory_gaps',lambda:e.gaps.resolve(None))
    attempted = set()
    jobs = phase('geometry',lambda:repair_geometry(e,image_at,affected,attempted))
    def refresh_geometry():
        for instance in e.active_instances():e._motion(instance)
        e.presence.resolve(image_at)
    phase('presence_refresh',refresh_geometry)
    e.sam_stages['second_pass_tracking'] = getattr(e.detector,'calls',0)-e._initial_sam_calls-e.sam_stages['first_pass']

    feedback = set()
    phase('refinement_probe',lambda:refine_templates(e,image_at,probe=True))
    def refine_identity():
        from .edge_identity import resolve_edges
        # Nested measurements are diagnostic subdivisions, not additional
        # top-level stages; do not add them twice when aggregating time.
        details = {}
        def timed(name, operation):
            tick = time.monotonic()
            result = operation()
            details[name] = time.monotonic()-tick
            return result
        e.stats['identity_feedback_seconds'] = details
        changed = timed('composite_review',lambda:review_composite_templates(e,refined=True))
        # Boundary fragments need the complete registered group and its
        # measured occurrences, including identities established by this
        # fine-evidence review. Do not inspect yesterday's split groups.
        changed |= timed('global_review',e.reconciler.global_review)
        boundary_changed = timed('edge_review',lambda:resolve_edges(e,image_at))
        if boundary_changed:
            first_checks=e.edge_identity_checks
            # A boundary registration may connect the complete group
            # needed to evaluate another fragment. One bounded feedback,
            # using cached image matches, avoids order-dependent results.
            boundary_changed |= timed('edge_feedback',lambda:resolve_edges(e,image_at))
            e.edge_identity_checks=first_checks+e.edge_identity_checks
        changed |= boundary_changed
        return changed
    feedback = phase('identity_feedback',refine_identity)
    if feedback:
        jobs += phase('geometry_feedback',lambda:repair_geometry(e,image_at,feedback,attempted))
    # Identity review consumes the inexpensive probe snapshot. Complete only
    # the resulting shared groups; probe originals are reused in their budget.
    phase('refinement',lambda:refine_templates(e,image_at))
    e.stats['second_pass_pose_jobs'] = jobs
    e.stats['feedback_rounds'] = int(bool(feedback))
    from .pose_review import review_weak_intervals
    reviewed = phase('measurement_review',lambda:review_weak_intervals(e,image_at))
    if reviewed:
        for instance in e.active_instances():
            if instance.template_id in reviewed:e._motion(instance)
    phase('final_trajectory_gaps',lambda:e.gaps.resolve(None))
    if feedback or reviewed:
        # Identity/geometry changes invalidate derived pixel ownership. Reuse
        # original probabilities; no SAM calls or destructive accumulation.
        phase('ownership_refresh',lambda:review_refinement_ownership(e))
    e.sam_stages['second_pass_refinement'] = getattr(e.detector,'calls',0)-e._initial_sam_calls-e.sam_stages['first_pass']-e.sam_stages['second_pass_tracking']
    phase('finalize',e._finalize_poses)
    from .edge_trajectory import prepare_boundary_output
    phase('boundary_output',lambda:prepare_boundary_output(e))
    e.output_plan = phase('output_plan',lambda:build_output_plan(e))
    e.timings['second_pass_seconds'] += time.monotonic()-started
