# Development guidelines

## Container environment

- Use the repository Dockerfile as the single source of truth for the runtime and reuse normal Docker layer caching. Do not introduce application-image overlays or custom dependency-hash build protocols.
- Source code is mounted into the container. Code-only changes require restarting the worker, not rebuilding the image.
- Put stable system dependencies before Python requirement files so Python dependency updates do not invalidate FFmpeg and compiler installation layers.
- Keep the versioned PyTorch/CUDA base unchanged unless an environment upgrade is required. Do not routinely pull the base image, disable caching or prune caches.
- Dependency changes require a normal build. Changing requirements invalidates the Python installation layer and subsequent layers.
- UAV must be self-contained. Do not depend on host packages or experimental PYTHONPATH entries. Keep weights, TensorRT engines and compilation caches in mounted directories.
- Verify imports and model initialization in the production image before resuming tasks.

## Public source hygiene

- Keep private experiment records, datasets, media, credentials, machine-specific paths and generated outputs outside the public source tree.
- Keep reusable synthetic tests and general profiling tools; do not include their private input manifests or measured reports.
- Preserve third-party notices and licenses. Do not describe the complete model pipeline as unrestricted for commercial use.
