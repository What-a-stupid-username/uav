"""Production orchestration: shared-template masks and ProPainter repair."""
import json
from pathlib import Path
from .config import get_config


def mask_config(config):
    from .template_mask.config import TemplateConfig
    return TemplateConfig(
        fps=config.fps, global_stride=config.sam3_global_stride,
        global_resolution=config.sam3_global_resolution,
        pixel_threshold=config.mask_pixel_threshold, dilation=config.mask_dilation,
        mask_reference_limit=config.mask_reference_limit,
        sift_backend=config.sift_backend, match_backend=config.match_backend,
        uncertain_mask_policy=config.uncertain_mask_policy,
        sam3_compile=config.sam3_compile, sam3_real_rope=config.sam3_real_rope,
        sam3_trt_engine=config.sam3_trt_engine,
        sam3_graph_grounding=config.sam3_graph_grounding)


def main(session=None):
    from .model_session import ModelSession
    from .segmented_pipeline import run_segments
    config = get_config()
    owned = session is None
    session = session or ModelSession()
    try:
        report = run_segments(config, mask_config(config), session)
        report['config'] = config.as_dict()
        for result in report['runs']:
            result['public_config'] = config.as_dict()
        preflight = next((r['coverage_preflight'] for r in report['runs'] if 'coverage_preflight' in r),None)
        if preflight is not None:report['coverage_preflight'] = preflight
        (Path(config.output)/'report.json').write_text(json.dumps(report, indent=2))
        return report
    finally:
        if owned:session.close()
