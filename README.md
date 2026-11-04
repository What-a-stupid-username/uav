# UAV — Unmark Any Video

UAV 自动检测视频水印，用共享模板生成逐帧 mask，再由 ProPainter 修复。Python API、CLI 和 Docker 统一使用新实现，旧 mask 引擎已删除。

第三方代码和权重沿用各自许可，见 [第三方声明](THIRD_PARTY_NOTICES.md)。ProPainter 采用含非商业使用限制的 S-Lab License；整个处理流程不能视为不受限制的商用软件。UAV 自有代码的根许可证尚待维护者确定，当前源码为发布准备版本。

## 运行

需要 NVIDIA GPU、Docker、NVIDIA Container Toolkit 和配套镜像。镜像提供依赖，源码通过挂载使用，修改 Python 代码无需重建镜像。
当前支持的安装方式为源码检出加配套 Docker 镜像。单独安装 Python wheel 不包含 vendored SAM3 运行目录与完整 CUDA 环境，不是独立可用的安装方式。

```sh
./docker/run.sh /absolute/path/input.mp4 ./output
./docker/run.sh /absolute/path/input.mp4 ./output-debug --max-seconds 15 --debug
./docker/run.sh /absolute/path/input.mp4 ./output-masks --no-repair
```

已安装环境中三个等价入口：

```sh
uav --source input.mp4 --output output
python -m uav --source input.mp4 --output output
python -m uav.cli --source input.mp4 --output output
python -m uav --help
```

每次使用新的输出目录。默认处理完整视频，采样 24 fps。--max-seconds 限制时长，--start-source-frame 指定源帧起点；--random-start 配合 --max-seconds 和 --seed 选择可复现片段。随机起点与指定起点互斥。

## 输出

### 常驻任务模式

网页队列或其他调度程序可保持一个 UAV 进程，在标准输入逐行提交 JSON 任务。首次任务懒加载模型，完成后保留模型及文本编码，等待下一任务；空闲不做推理，但会保持模型显存。

启动容器：./docker/worker.sh /absolute/input-directory /absolute/output-directory

或在配套运行环境中直接启动：python -m uav.cli --worker

容器输入目录映射为 /input，结果目录映射为 /output。例如提交一行 JSON：

    {"id":"job-1","source":"/input/0.mp4","output":"/output/job-1","config":{"debug":true}}

每个任务使用全新输出目录。标准输出返回 JSON 行：启动时 event=ready，任务结束 event=result（包含 id、complete、skipped、reason、report）；错误为 event=error。单任务推理日志写入该输出目录的 worker.log，报告及视频仍沿用普通 CLI 的布局。complete 与 skipped 需一起判断。

任务串行执行，逐任务 config 覆盖启动配置。输出、采样、面积预检等参数变化可直接复用模型；影响模型/编译的配置变化会重建会话。每个视频的证据缓存独立，推理异常后清理模型会话，下个任务重新初始化。输入 EOF 时释放模型并退出。普通单视频和 batch 入口保持兼容。

```text
output/
  report.json                 # 总报告，runs[0] 为本次运行
  clip-01/
    report.json               # mask、轨迹、SAM、修复耗时及统计
    warnings.jsonl            # 不确定区间及处理动作
    repaired.mp4              # 默认修复结果
```

普通修复模式不写逐帧 mask、中间概率图、特征缓存或 checkpoint。解码原图与最终 mask 从内存交给 ProPainter，不重复解码。--no-repair 导出 clip-01/masks/000000.png 等最终 mask，只需 SAM3 权重。

--debug 额外生成 clip-01/index.html、一份拼接的 mask-review.mp4、模板叠加图片及逐帧 mask。展示 Stage1 的 SAM 结果、框、对象、模板和轨迹，以及 Stage2 修复后的轨迹与最终 mask。横视频竖向排列，竖视频横向排列。Stage1 全局 SAM 结果保持至下一次检测。调试可视化由独立模块实现。

## Mask 流程

1. Stage1 在首帧、每 8 帧及末帧进行 1008 全局 SAM3 检测，聚合候选，维护实例、共享模板及 SIFT 轨迹。只使用全局粗 mask，不生成局部精细 mask，也不做输出外扩。
2. Stage2 复用原图、特征及观测，结合双向轨迹和布局证据关联残片与模板并修复缺口。每个被引用的共享模板选择最多 8 份原图参考；正方形背景裁剪为目标长边 1.7 倍且至少视频短边 35%，统一 1008 精修。有效概率配准后取均值，以 0.23 阈值生成 mask；精修失败回退粗模板。
3. 按各实例位姿投影模板，再做尺寸自适应外扩。--mask-dilation 默认 4，粗模板回退使用双倍外扩；额外不确定外扩有上限。不同实例最终取并集。
4. 全部最终 mask 通过不确定性策略后，交给 ProPainter 修复。

默认 CUDA SIFT 和 GPU 匹配。--sift-backend cpu 切换 OpenCV；--match-backend auto|cpu|cuda 独立控制匹配后端。模板共享，实例的出现、消失及运动独立。预测和边界保守帧不增加模板或运动证据。启动时执行面积预检，处理过程中不加逐帧面积跳过逻辑。

--uncertain-mask-policy conservative|skip|ignore 默认 conservative：覆盖可信位置和有界的不确定范围，完全无法定位则跳过视频。skip 遇到无法可靠处理的区间就跳过视频；ignore 省略不确定部分。相关动作会输出 warning 并记入 warnings.jsonl。complete 与 skipped 必须一起判断。

## Python API

```python
from uav import UAVConfig, run

report = run('input.mp4', 'output', UAVConfig(
    max_seconds=15,
    debug=False,
    repair=True,
    sift_backend='cuda',
    uncertain_mask_policy='conservative',
    raft_scale=0.5,
    flow_completion_scale=0.5,
))
```

参数由不可变 UAVConfig 显式提供。CLI 布尔参数支持 --xxx / --no-xxx。保留 ProPainter 的参考帧、光流、编译及传播选项，完整列表见 --help。旧 two_pass、SAM 视频追踪及局部追踪参数已移除。旧实验入口 python -m uav.template_mask 已移除，统一使用公共 CLI。

不读取旧的推理环境变量。权重及编译缓存路径仍由运行环境设置。编译默认开启，--no-compile 关闭 SAM3 编译；ProPainter 编译选项独立。Docker 挂载 cache/ 和 tmp/ 保存 Inductor、Triton 及原生扩展编译产物，后续进程可复用。视频证据只驻留内存。Stage1 checkpoint 仅在内部 runner 显式调试 API 保留，生产 CLI 不暴露且不自动读写。

## 权重和镜像

权重不随代码分发，不会运行时自动下载。默认路径：

```text
models/sam3/sam3.pt
models/propainter/raft-things.pth
models/propainter/recurrent_flow_completion.pth
models/propainter/ProPainter.pth
```

SAM3 权重来自 [Meta 模型页](https://huggingface.co/facebook/sam3)，需要申请访问并接受许可。ProPainter 权重来自 [官方 Release](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0)。遵守 [SAM3 许可证](licenses/SAM3.txt) 和 [ProPainter 许可证](licenses/ProPainter.txt)。UAV_MODEL_DIR 或 SAM3_CHECKPOINT / PROPAINTER_MODEL_DIR 可覆盖路径。Docker 默认镜像 uav:local；UAV_IMAGE 和 UAV_GPU_DEVICE 覆盖镜像及 CDI 设备。

```sh
docker build -t uav:local .
```

代码修改只需重启 worker，无需构建。依赖变化使用同一 Dockerfile 正常构建，复用 Docker 层缓存。系统依赖位于 Python 依赖之前，修改 requirements.txt 不会重装 FFmpeg 和编译工具；Python 安装层会重新执行。不基于旧应用镜像叠加环境，无需例行使用 --pull、--no-cache 或清理缓存。

下载走 sing-box：普通容器代理为 http://172.30.90.1:2081。若构建网络无法访问此地址，可使用 host 网络：

```sh
docker build --network host --build-arg HTTP_PROXY=http://127.0.0.1:2080 --build-arg HTTPS_PROXY=http://127.0.0.1:2080 -t uav:local .
```

## 验证

```sh
python -m unittest discover -s tests -p 'template*.py'
python -m unittest discover -s tests -p 'production*.py'
PYTHONPATH=. python tests/configuration.py
```

实现细节见 [共享模板引擎说明](uav/template_mask/README.md)。

性能分析使用 tools/profile_pipeline.py，传入 --source 和新的 --output 目录。默认跑非 debug 完整流程并输出 profile-summary.json；--cpu-profile 增加主线程调用统计，会带来额外开销，--no-repair 可只分析 mask。阶段 wall time 不应与嵌套 CUDA/CPU 时间重复相加。

启动面积预检：--max-watermark-coverage 与 --max-single-watermark-coverage 默认均为 0.15。按 --seed 从整个源视频抽取最多 4 个不同源帧，使用全局 SAM 检测框（未膨胀）计算总框并集及最大单框占画面的比例；任一采样严格超限就跳过，写入 coverage_preflight 报告和 warnings.jsonl，不启动 Stage1 或 ProPainter。即使 --max-seconds 限制处理片段，预检仍覆盖整个源视频。两个参数均设 1 可关闭预检；Python 也可将两者设为 None。通过检查后复用同一个 SAM 模型、文本编码和处理器；仅增加预检所需的最多 4 次检测及抽帧解码。

## 长视频分段与批处理

默认 --segment-seconds 15、--segment-overlap 1。相邻段重叠 1 秒；若当前段结束后只剩不超过 5 秒的新内容，直接将当前段延长至视频结束，默认最长 20 秒，避免单独处理很短的尾段。例如 30 秒视频分为 0–15 秒、14–30 秒。重叠区对两段未编码的最终修复 RGB 帧线性 lerp，权重从上一段过渡至下一段，每个输出帧只编码一次。内存充足时可改为 --segment-seconds 30 或 60，最后一段同样最多延长 5 秒。--max-seconds 仍表示整个任务最多处理多久，并非段长。

每个视频使用一个连续采样时间轴和解码线程，仅缓存当前段及重叠原图。重叠原图直接复用，避免独立采样造成接缝错帧。面积预检每个源视频只做一次。SAM 与 ProPainter 权重加载一次后同时驻留 GPU，在各段和各视频间复用；编译产物保留，不因分段而搬移权重。两类模型使用独立 Graph 内存池；当前 CUDA 运行环境下，切换模型阶段时释放上一阶段 Graph 和静态缓冲，后续按需重新捕获，避免交替重放旧 Graph 引发非法访问。

单视频最终输出仍为 clip-01/repaired.mp4。多段报告与 debug 页面位于 segments/0001、segments/0002 等目录，--debug 时根目录 index.html 提供各段审核链接。--no-repair 输出统一逐帧 mask，重叠区取并集。任何一段跳过时，整个视频标记跳过并移除不完整的修复视频。

本地批处理命令（依次处理，不并发争用显存）：

    python -m uav --source a.mp4 b.mp4 --output batch-output
    python -m uav --source a.mp4 --source b.mp4 --output batch-output

Docker 批处理（每个输入文件单独只读挂载）：

    ./docker/batch.sh ./batch-output /absolute/a.mp4 /absolute/b.mp4 -- --segment-seconds 15

Python 批处理：

    from uav import UAVConfig, run_batch
    report = run_batch(['a.mp4', 'b.mp4'], 'batch-output', UAVConfig(segment_seconds=15))

各视频分别输出到 0001-a、0002-b 等目录，batch-report.json 汇总成功、跳过与错误。单个视频失败后记录错误并继续下一项；失败后重建模型会话，避免污染后续任务。

UAV 镜像包含完整运行依赖，包括 FFmpeg，不依赖宿主机的工具安装目录。
所有编码出口统一使用运行环境内的 FFmpeg，IMAGEIO_FFMPEG_EXE 可显式覆盖。
随机抽片、准备评测集等工作由外部工具完成，再通过公开 CLI/API 提交视频。

### 两步模板精修

高质量修复可用 CLI `--high-quality` 或 Python `UAVConfig.high_quality()`：重叠 5 帧、原分辨率修复、RAFT 8 次。参考图仍为 6 张，光流及补全比例仍为 0.5，保留全部后端与缓存优化。普通档默认重叠 3 帧、修复比例 0.7、RAFT 4 次。CLI 中预设之后的显式参数可覆盖单项；Python 使用 `UAVConfig.high_quality(raft_iters=4)` 覆盖。

ProPainter 修复裁剪的宽高按 192–512 像素、每 32 像素一档向上选择，保留 mask
支持区域外默认留修复图像短边的 8.89% 作为背景（`--repair-context-ratio`）。边距随视频尺寸和修复比例变化，单侧最多 248 像素，保证 512 像素输入内仍有可修复区域；宽水印会相应分块。裁剪不再二次缩放，位置对齐修复图像特征网格。
时间重叠默认 3 帧：外层处理 24 帧、推进 21 帧，内层窗口为 10 帧、推进 7 帧；参考帧为 6 张。

参考编码使用 128 MiB GPU 缓存；淘汰的结果在最多 4 GiB 的任务内 RAM 缓存中复用，
避免重复编码。两级缓存按时间轴清理，结束时释放，不写磁盘。
RAM 缓存使用 pinned memory 异步传输，CUDA event 保证恢复时的流依赖。
通过 --propainter-reference-host-cache-mib 调整 RAM 上限，设为 0 可关闭。
大型特征画布在 RAM 中由一个有界工作线程按原顺序融合，与后续裁剪的 GPU 推理重叠；
每个窗口输出前等待融合完成。

ProPainter 编码器默认使用 TensorRT FP16。首次使用按当前权重、GPU 与运行时构建，
缓存位于 `tmp/tensorrt/propainter/`，后续启动直接加载；镜像包含 TensorRT 和 ONNX 依赖。
超出引擎尺寸或批次范围时回退 PyTorch；`--no-propainter-trt-encoder` 可关闭。
修复报告中的 `encoder_backend` 记录实际安装的编码器后端。

RAFT 特征编码、上下文编码和循环更新默认使用 TensorRT FP16，默认迭代 4 次，光流分辨率比例为 0.5。可用 `--raft-iters 8` 恢复 8 次迭代。

默认 `--repair-scale 0.7` 将 ProPainter 输入宽高缩到约 70%（对齐 8 像素），可用 `--repair-scale 1` 恢复原分辨率修复。检测和最终输出保持原分辨率；缩小后的 mask 保留覆盖到的细笔画，修复结果使用双线性插值还原，仅替换原始 mask 内的像素，再进行分段混合。CPU 缩放通过有界队列与 GPU 修复重叠，输出还原与合成使用缓存的融合编译。背景边距及裁剪尺寸仍以修复分辨率的像素计。降低分辨率会影响修复细节，属于有损优化。
支持的 TRT 批次内每帧只编码一次，前后向相邻帧对共享特征和上下文，避免重复编码内部帧。
引擎缓存在 `tmp/tensorrt/raft/`，顺序执行的模块共享工作区，超出支持范围时回退 PyTorch。
`--no-propainter-trt-raft` 可关闭；报告中的 `raft_backend` 记录安装的后端。

特征传播中的偏移预测、传播修正和融合卷积默认使用 TensorRT FP16，仍由现有 CUDA Graph
执行；各输入尺寸保留独立上下文，共享固定工作区。引擎缓存位于
`tmp/tensorrt/propagation/`。超出范围时回退 PyTorch，
`--no-propainter-trt-propagation` 可关闭，报告字段为 `propagation_backend`。
该加速接受 FP16 算子实现带来的数值差异；不减少参考帧、传播步骤或时间窗口。

光流补全默认使用 CUDA Graph，沿用当前流的变形卷积内核以减少循环调度开销。
图缓存最多保留 8 种输入签名，并随修复阶段的图缓存一起释放。
`--no-propainter-graph-completion` 可关闭；报告中的 `completion_graph` 记录捕获、回放和回退次数。

Stage 1 在线模板归并按时间将可靠参考分成前后两组，各取质量最好的一份主动查询；
被查询模板仍保留完整匹配参考集，逐帧追踪和新观测归属不减少参考。
Stage 2 使用全部保留参考复查身份，第一遍的查询预算不会删除原始证据。

Stage 2 先为每个有效模板尝试一份可靠原图精修，失败时在最多 8 份候选内继续尝试；
随后结合原图配准和轨迹进行模板关联，再对最终共享组补齐最多 8 份独立原图。
首轮已接受的参考跨模板复用，计入共享预算；失败的相同任务不重复运行。
`--mask-reference-limit` 可调整参考上限（1–8，默认 8）。
