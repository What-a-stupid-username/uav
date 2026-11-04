"""Session lifecycle and protocol tests; no GPU or media reads."""
import io
import json
from pathlib import Path
import tempfile
import unittest

from uav.config import UAVConfig
from uav.worker import Worker, serve


class Session:
    def __init__(self):
        self.events = []
        self.detector = type('Detector', (), {'cache': object()})()
        self.closed = False

    def close(self):
        self.closed = True


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # Existence check only: no media contents read.
        self.source = self.root / 'synthetic-input'
        self.source.touch()
        self.sessions, self.calls = [], []
        def factory():
            session = Session()
            self.sessions.append(session)
            return session
        def run(source, output, config, *, _session):
            self.calls.append((config, _session))
            print('inference diagnostic')
            _session.events.append({'phase': 'sam'})
            if config.seed == 99:
                raise RuntimeError('synthetic inference failure')
            report = dict(complete=True, skipped=config.seed == 98)
            (Path(output) / 'report.json').write_text(json.dumps(report))
            return report
        self.worker = Worker(session_factory=factory, run=run)

    def tearDown(self):
        self.worker.close()
        self.temp.cleanup()

    def request(self, index, **config):
        return dict(id=str(index), source=str(self.source), output=str(self.root / str(index)), config=config)

    def test_reuses_session_and_releases_evidence(self):
        first = self.worker.process(self.request(1))
        self.assertEqual(first['event'], 'result')
        self.worker.process(self.request(2, debug=True, mask_dilation=6))
        self.assertEqual(len(self.sessions), 1)
        self.assertIs(self.calls[0][1], self.calls[1][1])
        self.assertIsNone(self.sessions[0].detector.cache)
        self.assertEqual(self.sessions[0].events, [])
        self.assertIn('inference diagnostic', (self.root / '1/worker.log').read_text())

    def test_model_options_reset_session(self):
        self.worker.process(self.request(1))
        self.worker.process(self.request(2, sam3_compile=False))
        self.assertEqual(len(self.sessions), 2)
        self.assertTrue(self.sessions[0].closed)

    def test_failure_resets_but_skip_reuses(self):
        self.worker.process(self.request(1, seed=98))
        with self.assertRaises(RuntimeError):
            self.worker.process(self.request(2, seed=99))
        self.assertTrue(self.sessions[0].closed)
        self.worker.process(self.request(3))
        self.assertEqual(len(self.sessions), 2)
        with self.assertRaises(ValueError):
            self.worker.process(self.request(3))
        self.assertEqual(len(self.sessions), 2)

    def test_protocol_and_eof_cleanup(self):
        incoming = io.StringIO('invalid json\n[]\n' + json.dumps(self.request(1)) + '\n')
        outgoing = io.StringIO()
        serve(incoming=incoming, outgoing=outgoing, worker=self.worker)
        events = [json.loads(line) for line in outgoing.getvalue().splitlines()]
        self.assertEqual([e['event'] for e in events], ['ready', 'error', 'error', 'result'])
        self.assertEqual(events[-1]['id'], '1')
        self.assertTrue(self.sessions[0].closed)

    def test_invalid_configuration_does_not_load(self):
        with self.assertRaises(TypeError):
            self.worker.process(self.request(1, nonexistent=True))
        with self.assertRaises(ValueError):
            self.worker.process(self.request(1, source='elsewhere'))
        self.assertEqual(self.sessions, [])


if __name__ == '__main__':
    unittest.main()
