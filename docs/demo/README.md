# Demo

一段视频同时包含右上静止角标、曲线移动文字、换位置闪现的文字和左下旋转 logo。

- [输入视频](input.mp4)
- [Debug 视频](debug.mp4)：上方 Stage1 检测与轨迹，下方 Stage2 最终 mask
- [修复结果](result.mp4)

## 素材与许可

背景为《Big Buck Bunny》，© 2008 Blender Foundation，
[项目主页](https://peach.blender.org/)，
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)。
使用官方 640×360 版本的第 60–72 秒，移除音频、采样为 24 fps，添加合成文字与自制 logo；结果和 debug 为 UAV 处理后的版本。此处的示例媒体沿用 CC BY 3.0，不适用根目录的代码许可。

源文件：[Blender 官方下载](https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_640x360.m4v.zip)。下载并解压后，在安装 NumPy、Pillow 和 FFmpeg 的环境中生成输入：

```sh
python docs/demo/create_demo.py BigBuckBunny_640x360.m4v input.mp4
./docker/run.sh "$(realpath input.mp4)" ./demo-output --high-quality --debug
```

脚本使用 DejaVu Sans 字体。完整视频为 12 秒、640×360、24 fps；README 动图降为 8 fps，仅用于预览。
