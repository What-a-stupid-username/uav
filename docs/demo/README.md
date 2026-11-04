# Demo

A single clip contains a static corner mark, text moving along a curve, flashing text at changing positions, and a rotating custom logo.

The repository includes input.mp4, debug.mp4, and result.mp4. README animations are 8 fps previews; the MP4 files are 12 seconds, 640×360, 24 fps. Processing uses --high-quality --debug.

## Source and attribution

Background: *Big Buck Bunny*, © 2008 Blender Foundation. [Project](https://peach.blender.org/) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

The clip uses seconds 60–72 of the official 640×360 release, removes audio, samples at 24 fps, and adds synthetic text and a custom logo. UAV produces the repaired and debug versions. Demo media is distributed under CC BY 3.0, separately from the root code license.

[Download the source archive](https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_640x360.m4v.zip), extract it, then generate the input with NumPy, Pillow, FFmpeg, and DejaVu Sans fonts installed:

	python docs/demo/create_demo.py BigBuckBunny_640x360.m4v input.mp4
	./docker/run.sh "$(realpath input.mp4)" ./demo-output --high-quality --debug

To try the included input directly:

	./docker/run.sh "$(realpath docs/demo/input.mp4)" ./demo-output --high-quality --debug
