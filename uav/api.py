"""Sequential video removal API with explicit, validated run parameters."""
from dataclasses import replace
from pathlib import Path
from .config import UAVConfig,use_config


def run(source, output, config=None, *, _session=None):
    source=Path(source).resolve();output=Path(output).resolve()
    if not source.is_file(): raise ValueError('source file does not exist')
    if (output/'report.json').exists(): raise ValueError('use a fresh output directory')
    config=replace(config or UAVConfig(),source=str(source),output=str(output))
    from .settings import validate_models
    validate_models(repair=config.repair)
    output.mkdir(parents=True,exist_ok=True)
    with use_config(config):
        from . import pipeline
        report = pipeline.main(session=_session)
    return report


def run_batch(sources, output, config=None):
    """Process videos sequentially with shared model weights; retain each result."""
    import json
    from .model_session import ModelSession
    config = config or UAVConfig()
    sources = [Path(source).resolve() for source in sources]
    if not sources:raise ValueError('batch requires at least one video')
    for source in sources:
        if not source.is_file():raise ValueError(f'source file does not exist: {source}')
    from .settings import validate_models
    validate_models(repair=config.repair)
    output = Path(output).resolve()
    if (output/'batch-report.json').exists():raise ValueError('use a fresh batch output directory')
    output.mkdir(parents=True,exist_ok=True)
    session = ModelSession()
    jobs = []
    try:
        for index,source in enumerate(sources):
            destination=output/f'{index+1:04d}-{source.stem}'
            try:
                report=run(source,destination,config,_session=session)
                jobs.append(dict(source=str(source),output=str(destination),report=report))
            except Exception as error:
                import traceback
                traceback.print_exc()
                jobs.append(dict(source=str(source),output=str(destination),
                                 error=type(error).__name__+': '+str(error)))
                session.close()
                session=ModelSession()
            summary=dict(complete=len(jobs)==len(sources),
                         failed=sum('error' in j for j in jobs),
                         skipped=sum(bool(j.get('report',{}).get('skipped')) for j in jobs),
                         successful=sum(bool(j.get('report',{}).get('complete')) and
                                        not j.get('report',{}).get('skipped',False) for j in jobs),
                         jobs=jobs)
            (output/'batch-report.json').write_text(json.dumps(summary,indent=2))
        return summary
    finally:session.close()
