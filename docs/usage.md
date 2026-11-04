# 使用说明

以下 Python 命令在 UAV 配套环境中运行。Docker 入口可在输出目录后追加同样的参数。

## 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| --fps | 24 | 输出采样帧率 |
| --max-seconds | 不限 | 最多处理的秒数 |
| --segment-seconds | 15 | 长视频分段时长 |
| --segment-overlap | 1 | 段间重叠秒数 |
| --high-quality | 关闭 | 原分辨率、重叠 5 帧、RAFT 8 次 |
| --repair-scale | 0.7 | 修复输入的宽高比例，输出尺寸不变 |
| --repair-context-ratio | 0.0889 | 裁剪背景边距占修复图像短边的比例 |
| --mask-dilation | 4 | mask 的尺寸自适应外扩参数 |
| --max-watermark-coverage | 0.15 | 启动预检的水印总框面积比例上限 |
| --max-single-watermark-coverage | 0.15 | 启动预检的单框面积比例上限 |
| --uncertain-mask-policy | conservative | 无法可靠定位时的处理策略 |

两个面积阈值均设为 1 可关闭预检。预检从整个源视频抽取最多 4 帧，按检测框面积判断，不是真实 mask 面积测量。

conservative 使用有界保守覆盖，完全无法定位时跳过；skip 遇到不确定区间就跳过；ignore 省略不确定部分。报告中 complete=true 仍可能 skipped=true，需同时检查。

完整参数：python -m uav --help。CLI 中 --high-quality 后的显式参数可以覆盖预设。

## 输出

- report.json：总状态、参数及各阶段耗时。
- clip-01/repaired.mp4：最终视频，无音频。
- clip-01/warnings.jsonl：不确定区间和处理动作。
- --debug：额外生成审核 HTML、Stage1/Stage2 拼接视频、模板图片及逐帧 mask。
- --no-repair：只导出最终逐帧 mask，仅需 SAM3 权重。

长视频的分段报告位于 segments/0001 等目录，最终视频仍为 clip-01/repaired.mp4。任何一段跳过时不保留不完整的最终修复视频。批处理的 batch-report.json 汇总成功、跳过和失败项。

## 常驻 worker

	./docker/worker.sh /absolute/input-directory /absolute/output-directory

向标准输入逐行提交 JSON：

	{"id":"job-1","source":"/input/a.mp4","output":"/output/job-1","config":{"debug":true}}

启动返回 event=ready；任务结束返回 event=result，含 complete、skipped、reason 和 report；失败返回 event=error。每个任务使用新输出目录，日志在 worker.log。任务串行执行，模型保留在显存，输入 EOF 后退出。

## Python 批处理

	from uav import UAVConfig, run_batch
	
	report = run_batch(["a.mp4", "b.mp4"], "batch-output", UAVConfig())

## 环境与缓存

支持源码检出加 Docker，不提供独立可运行的 wheel。镜像包含 FFmpeg、CUDA 编译工具和 Python 依赖；代码通过只读挂载运行，改代码只需重启 worker。

权重在 models/，引擎及原生扩展在 tmp/，Inductor/Triton 缓存在 cache/。不要将这些目录提交到 Git。默认 4 GiB RAM 参考缓存可用 --propainter-reference-host-cache-mib 调整。视频证据只驻留 RAM，生产 CLI 不自动读写 checkpoint。

依赖变化使用同一 Dockerfile 构建，复用层缓存。需代理时通过 --build-arg HTTP_PROXY 和 --build-arg HTTPS_PROXY 传入构建环境可访问的地址。

实现结构见 [模板引擎](../uav/template_mask/README.md)。测试：

	python -m unittest discover -s tests -p '*.py'
	PYTHONPATH=. python tests/configuration.py
