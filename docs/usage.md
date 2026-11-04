# Usage guide

Run Python commands inside the UAV environment. Docker scripts accept the same options after the output directory.

## Parameters

| Option | Default | Purpose |
| --- | --- | --- |
| --fps | 24 | Output sampling rate |
| --max-seconds | Unlimited | Maximum duration to process |
| --segment-seconds | 15 | Segment length for long videos |
| --segment-overlap | 1 | Overlap between segments, in seconds |
| --high-quality | Off | Full-resolution repair, overlap 5, RAFT 8 |
| --repair-scale | 0.7 | Inpainting width/height scale; output dimensions stay unchanged |
| --repair-context-ratio | 0.0889 | Context margin as a fraction of the repair image's shorter side |
| --mask-dilation | 4 | Size-adaptive mask expansion parameter |
| --max-watermark-coverage | 0.15 | Startup limit for the union of detected boxes |
| --max-single-watermark-coverage | 0.15 | Startup limit for an individual detected box |
| --uncertain-mask-policy | conservative | Policy for uncertain intervals |

Set both coverage limits to 1 to disable the check. It samples up to four frames across the entire source and measures detection boxes, not final mask area.

The conservative policy uses bounded coverage and skips a video when an object cannot be located. The skip policy skips any unresolved interval; ignore omits uncertain parts. Check both complete and skipped in reports.

Use python -m uav --help for all options. Flags after --high-quality override individual preset values.

## Output

- report.json: video status, configuration, and stage timings.
- clip-01/repaired.mp4: repaired video without audio.
- clip-01/warnings.jsonl: uncertain intervals and actions taken.
- --debug: review HTML, combined Stage1/Stage2 video, template overlays, and frame masks.
- --no-repair: final masks only; requires SAM3 weights but no ProPainter weights.

Long-video segment reports live under segments/0001, etc. The final video remains at clip-01/repaired.mp4. An unsuccessful segment prevents an incomplete repaired video from being retained. Batch summaries are written to batch-report.json.

## Persistent worker

	./docker/worker.sh /absolute/input-directory /absolute/output-directory

Submit one JSON object per line on stdin:

	{"id":"job-1","source":"/input/a.mp4","output":"/output/job-1","config":{"debug":true}}

Startup emits event=ready. Each task emits event=result with complete, skipped, reason, and report, or event=error on failure. Use a new output directory per task. Logs are written to worker.log. Tasks run sequentially; weights remain in VRAM between tasks. EOF shuts down the worker.

## Python batch API

	from uav import UAVConfig, run_batch

	report = run_batch(["a.mp4", "b.mp4"], "batch-output", UAVConfig())

## Environment and caches

Use a source checkout with the Docker environment. The Python wheel alone does not include a complete SAM3/CUDA runtime. The image includes FFmpeg, CUDA build tools, and Python dependencies. Source is mounted read-only; restart the worker after code changes.

Weights live in models/, engines and native extensions in tmp/, and Inductor/Triton caches in cache/. Do not commit those directories. The RAM reference cache defaults to 4 GiB; adjust it with --propainter-reference-host-cache-mib. Video evidence stays in RAM; the production CLI does not automatically read or write checkpoints.

Rebuild dependencies using the same Dockerfile and normal layer caching. Pass HTTP_PROXY and HTTPS_PROXY build arguments if your build needs a proxy.

See the [template engine overview](../uav/template_mask/README.md) for code organization. Tests:

	python -m unittest discover -s tests -p '*.py'
	PYTHONPATH=. python tests/configuration.py
