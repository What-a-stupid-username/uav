"""Serial JSON-lines jobs sharing one lazy GPU model session.

Input: {"id": "...", "source": "...", "output": "...", "config": {...}}
Output: ready/result/error events. Inference logs go to output/worker.log.
EOF shuts down cleanly; malformed jobs do not terminate the worker.
"""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

from .config import UAVConfig

PER_JOB = {'source', 'output', 'debug', 'max_seconds', 'start_source_frame',
           'random_start', 'seed', 'max_watermark_coverage',
           'max_single_watermark_coverage', 'segment_seconds', 'segment_overlap',
           'uncertain_mask_policy', 'mask_dilation', 'mask_pixel_threshold'}


class Worker:
    def __init__(self, config=None, *, session_factory=None, run=None):
        self.config = config or UAVConfig()
        self.session_factory = session_factory
        self.run = run
        self.session = None
        self.signature = None

    def close(self):
        session, self.session = self.session, None
        self.signature = None
        if session is not None:
            session.close()

    def process(self, request):
        if not isinstance(request, dict) or set(request) - {'id', 'source', 'output', 'config'}:
            raise ValueError('expected id, source, output and optional config')
        source, output = request.get('source'), request.get('output')
        if not isinstance(source, str) or not isinstance(output, str):
            raise ValueError('source and output must be paths')
        overrides = request.get('config', {})
        if not isinstance(overrides, dict) or {'source', 'output'} & overrides.keys():
            raise ValueError('invalid config overrides')
        config = replace(self.config, **overrides)
        if not Path(source).is_file():
            raise ValueError('source file does not exist')
        folder = Path(output)
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('use a fresh output directory')
        folder.mkdir(parents=True, exist_ok=True)
        signature = {k: v for k, v in config.as_dict().items() if k not in PER_JOB}
        with (folder / 'worker.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                if self.session is not None and self.signature != signature:
                    self.close()
                if self.session is None:
                    factory = self.session_factory
                    if factory is None:
                        from .model_session import ModelSession
                        factory = ModelSession
                    self.session = factory()
                    self.signature = signature
                run = self.run
                if run is None:
                    from .api import run
                report = run(source, output, config, _session=self.session)
                return dict(event='result', id=request.get('id'), complete=report.get('complete', False),
                            skipped=report.get('skipped', False), reason=report.get('reason'),
                            report=str(folder / 'report.json'))
            except Exception:
                traceback.print_exc()
                self.close()
                raise
            finally:
                if self.session is not None:
                    detector = getattr(self.session, 'detector', None)
                    if detector is not None:
                        detector.cache = None
                    self.session.events.clear()


def serve(config=None, *, incoming=None, outgoing=None, worker=None):
    incoming, outgoing = incoming or sys.stdin, outgoing or sys.stdout
    worker = worker or Worker(config)
    def emit(value):
        outgoing.write(json.dumps(value, ensure_ascii=False) + '\n')
        outgoing.flush()
    emit(dict(event='ready', protocol=1))
    try:
        for line in incoming:
            request = None
            try:
                if len(line) > 131072:
                    raise ValueError('request too large')
                request = json.loads(line)
                emit(worker.process(request))
            except Exception as error:
                emit(dict(event='error', id=request.get('id') if isinstance(request, dict) else None,
                          error=type(error).__name__ + ': ' + str(error)))
    finally:
        worker.close()
