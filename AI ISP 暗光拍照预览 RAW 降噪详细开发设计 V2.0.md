# AI ISP 暗光拍照预览 RAW 降噪详细开发设计 V2.0

> 目标平台：麒麟 9000 系列终端，具体 NPU、DDK、固件和算子能力以目标设备实测为准
> 业务定位：30fps 拍照预览中的暗光 RAW 域 AI 残余降噪
> 推理形态：Full-Tensor Single Pass、Static Shape、三套静态 OM、共享一套权重
> 部署模型：Tiny Conditional MobileNAFNet-W16 Dark Preview，INT8 QAT
> 文档状态：工程实现与量产准入基线
> 上位参考：[AI ISP 量产设计基线 V18.0](./AI%20ISP量产设计基线%20V18.0.md)
> 历史详细设计：[AI ISP RAW 降噪增强详细开发设计 V1.0](./AI%20ISP%20RAW降噪增强详细开发设计%20V1.0.md)
> 适用关系：本文仅在 Dark Photo Preview 场景取代 V1.0 的全分辨率 Tile 路线；V1.0 仍可作为 Capture/高分辨率 RAW 增强的历史参考。
> 性能声明：本文中的时延、功耗、内存和画质数值是量产准入门槛，不是当前已达成结果。

---

## 0. 执行摘要

本项目在 Camera Preview 的线性 RAW 阶段增加一个按暗光条件触发的 AI Stream Node。前端 ISP/HAL 先完成 Bayer 相位安全的 Binning、Crop 和格式标准化，使 AI 输入已经处于预览所需的最终 RAW 格点；模型一次处理完整 Packed RAW Tensor，预测同格点噪声分量，再在 Packed RAW 域执行减法。整个链路不允许将低分辨率残差插值后加回高分辨率 Bayer。

量产基线不是动态 Shape，也不是一个任意尺寸模型，而是同一套权重导出三个静态 ONNX，并分别编译为三个静态 OM。Camera HAL 按当前主路预览流的有效宽高比和协商结果选择 Profile；Camera/Sensor 差异由 ConditionSchemaV2 和 Sensor Profile 表达。

|项目|冻结结论|
|---|---|
|业务场景|30fps Photo Preview，仅 Dark Scene 启用|
|ISP 位置|BLC/DPC/LSC、相位安全 Binning/Crop 之后，Demosaic 之前|
|图像输入|NCHW Packed RAW，通道顺序固定为 R、Gr、Gb、B|
|推理方式|Full-Tensor Single Pass；无应用层 Tile、Overlap、Hann Blend|
|静态 Profile|P0 4:3、P1 16:9、P2 3:2，三个独立 OM|
|Student|Tiny Conditional MobileNAFNet-W16，Encoder 2/2/4、Middle 2、Decoder 2/2/2|
|通道|16、32、64、128，所有主干通道至少 8 通道对齐|
|Condition|ConditionSchemaV2，FP32[1,24]|
|模型输出|noise_pred，与有效 Packed RAW 同格点|
|重建公式|raw_out = clamp(raw_in - noise_pred, 0, 1)|
|Teacher|Conditional NAFNet-W32，仅训练和蒸馏使用|
|压缩|P10/P15 结构化候选；没有真机收益时允许发布未剪枝 W16|
|量化|默认全图 INT8 QAT；任何混合精度例外必须仍在 NPU 且经过审批|
|性能门槛|P0/P1/P2 端到端 P95 分别不超过 30/25/27ms|
|失败策略|原始 Packed RAW 安全 Bypass；禁止静默 CPU AI 回退|

### 0.1 相对 V1.0 的核心变化

|V1.0|V2.0|
|---|---|
|Single/HDR/Capture、多分辨率增强|仅 Dark Photo Preview|
|12MP/50MP 全分辨率 Packed RAW|预览 RAW 流的固定有效格点|
|512/1024 Tile + Overlap + Hann|Full-Tensor Single Pass|
|K9000RawDenoiseNet-W24，约 3M 参数预算|MobileNAFNet-W16，约 0.55M～0.80M 参数工程预算|
|ConditionSchemaV1，32 维|ConditionSchemaV2，24 维|
|W64 Teacher|W32 Teacher|
|P10/P20/P30 为必要量产流程|P10/P15 为候选，以画质和真机收益自动选择|
|百毫秒至秒级 Capture 门槛|33.33ms 帧预算内的 Preview 门槛|

### 0.2 已冻结、待实测和禁止项

已冻结：

- RAW 同格点输入、噪声预测和减法；
- 三个独立 Static Shape OM；
- Full-Tensor Single Pass；
- W16 Student 拓扑、ConditionSchemaV2 和输出符号；
- 暗光触发滞回、安全 Bypass 和禁止 CPU AI 回退。

待目标机实测：

- DWConv、Static Slice/Mul、Nearest Resize 和 FiLM 是否全部下沉；
- P0/P1/P2 的真实 P50/P95/P99、Workspace、内存带宽和温升；
- P10/P15 是否比未剪枝 W16 产生可重复的端到端收益；
- 全 INT8 是否满足暗部 Banding 门槛，是否需要受控 FP16 NPU Island；
- 三个 OM 能否同时常驻，或是否采用双常驻加按需加载。

禁止项：

- 任意 H/W 动态 Shape 作为量产主路径；
- 将 Camera 类型直接绑定到固定 Profile；
- 对 Bayer Mosaic 进行普通 RGB Resize；
- 对低分辨率 noise/residual 插值后回写高分辨率 RAW；
- 应用层 Tile、Overlap、Hann 拼接；
- 未记录的 CPU/GPU 算子回退；
- 异常时继续输出可能被污染的 AI RAW。

---

## 1. 产品目标、范围与验收口径

### 1.1 产品目标

在暗光拍照预览中降低高 ISO Shot Noise、Read Noise、彩噪、行列噪声和轻度传统 RAW NR 残留，同时保护人脸、文字、毛发、织物、树叶、灯牌和天空渐变。目标是改善取景体验，不追求替代最终夜景照片的 HDR/MFNR/Capture 处理。

### 1.2 模块负责

- 对 HAL 当前主路物理 Camera 的单帧预览 RAW 做残余降噪；
- 依据 ISO、曝光、噪声模型、Camera/Sensor/Lens Profile 调整降噪强度；
- 预测与输入同格点的 Packed RAW 噪声；
- 管理 Dark Trigger、Profile 选择、OM 执行、结果校验和安全 Bypass；
- 输出可供后续 Demosaic 和 RGB ISP 使用的 Packed RAW；
- 记录非图像隐私的性能、错误和回退指标。

### 1.3 模块不负责

- Capture 静态照片处理；
- Video Temporal NR 或跨帧网络；
- HDR/MFNR Fusion、Ghost 消除或运动补偿；
- 跨摄像头配准、融合或无缝变焦的双路合成；
- AI Demosaic、AWB、CCM、Tone Mapping、Sharpen 和 JPEG；
- 恢复饱和、错误配准、错误融合或上游已经丢失的信息；
- 对未标定 Sensor 进行激进增强。

### 1.4 端到端计时边界

端到端计时起点为 HAL 主路 RAW Buffer 和本帧 Metadata 均可读，终点为增强 Packed RAW Buffer 可供 Demosaic 节点读取。计时包含：

- Bayer Pack/语义通道重排；
- Black/White Normalize；
- Profile Crop/Pad；
- Condition 构建；
- NPU 推理；
- 有效区 Crop、噪声相减、Clamp 和反归一化；
- 必要的 Buffer/DMA 和同步。

计时不包含 Sensor Readout、BLC/DPC/LSC 的固定前端时延、Demosaic、RGB ISP 和 Display Composition。

### 1.5 首发硬门槛

|类别|P0|P1|P2|
|---|---:|---:|---:|
|端到端 P50 目标|≤ 24ms|≤ 20ms|≤ 22ms|
|端到端 P95 硬门槛|≤ 30ms|≤ 25ms|≤ 27ms|
|端到端 P99 硬门槛|≤ 33ms|≤ 30ms|≤ 31ms|
|10 分钟 AI 归因掉帧率|< 0.1%|< 0.1%|< 0.1%|

全 Profile 公共门槛：

|项目|目标值|硬门槛|
|---|---:|---:|
|增量峰值内存|≤ 128MB|≤ 160MB|
|热稳态 P95 相对冷态退化|≤ 10%|≤ 15%|
|INT8 发布包单个权重载荷|≤ 0.8MB|≤ 1.0MB|
|NPU 算子覆盖|100%|100%|
|未批准 CPU/GPU 回退|0|0|
|连续预览稳定性|30 分钟无崩溃/OOM|10 分钟无崩溃/OOM|

所有门槛必须在目标机、目标固件、量产频率策略和正式 Camera Pipeline 上实测。理论 TOPS、单算子耗时或 PC 模拟不能替代整链路结果。

---

## 2. ISP Pipeline 与节点边界

### 2.1 推荐链路

~~~mermaid
flowchart LR
    S["Sensor RAW10/RAW12"] --> BLC["BLC"]
    BLC --> DPC["DPC"]
    DPC --> LSC["LSC"]
    LSC --> NORM["Exposure/Gain Metadata<br/>Noise Estimation"]
    NORM --> BC["Phase-safe Binning/Crop<br/>Preview RAW lattice"]
    BC --> PACK["Bayer Pack<br/>R,Gr,Gb,B"]
    PACK --> FMT["AI Formatter<br/>Normalize/Profile Pad"]
    FMT --> AI["Tiny Conditional MobileNAFNet-W16<br/>INT8 Static OM<br/>Full Tensor"]
    AI --> NP["noise_pred"]
    FMT --> SUB["raw_out = raw_in - noise_pred"]
    NP --> SUB
    SUB --> SAFE["Finite/Range Check<br/>Crop/Clamp"]
    SAFE --> DEM["Demosaic"]
    DEM --> RGB["RGB ISP"]
    RGB --> DISP["Preview Display"]
~~~

若传统 RAW NR 位于 AI 之前，其强度、版本和参数必须作为数据版本的一部分冻结。训练数据的输入域必须与量产上游一致，禁止训练使用纯 Sensor RAW、量产却输入重度传统 NR 后 RAW。

### 2.2 RAW 格点不变量

设有效原始 Bayer 尺寸为 Hraw×Wraw，Packed RAW 为：

~~~text
Raw Bayer : N × 1 × Hraw × Wraw
Packed RAW: N × 4 × (Hraw/2) × (Wraw/2)
~~~

模型输入、noise_pred 和有效输出必须满足：

~~~text
shape(raw_valid) == shape(noise_pred_valid) == shape(raw_out_valid)
cfa_phase(raw_valid) == cfa_phase(raw_out_valid)
raw_out = clamp(raw_valid - noise_pred_valid, 0, 1)
~~~

所谓“Preview Feature Profile”不是从更大 Packed RAW 任意 Resize 得到的普通特征图，而是 Sensor/ISP/HAL 先通过保持 CFA 相位的 Binning、Skip 或偶数边界 Crop 得到的预览 RAW 格点。AI 不承担从低格点恢复高格点的任务。

### 2.3 CFA 语义

模型通道顺序固定为：

~~~text
[R, Gr, Gb, B]
~~~

RGGB、BGGR、GRBG、GBRG 必须在 Formatter 中重排到相同语义。旋转、镜像和 Crop 改变 CFA 相位时，必须同步更新通道映射。任何无法确认 CFA 的帧直接 Bypass。

### 2.4 应用层 Full Tensor 的定义

Full-Tensor Single Pass 表示：

- Runtime 每帧只提交一次完整 Profile Tensor；
- 不生成应用层 Tile Queue；
- 不做 Overlap、Hann、Weighted Add 或接缝裁剪；
- 输出顺序与输入帧一一对应；
- 不因 Camera 内容或帧内区域改变 Shape。

NPU 编译器为适配片上存储而进行的内部切块、算子调度和 Buffer 复用属于硬件实现细节，不改变应用层 Full Tensor 语义。

---

## 3. Static Profile 与多摄设计

### 3.1 Profile 冻结表

Shape 均按 N×C×H×W 表示：

|ID|用途|有效 Packed RAW|编译输入|有效 Bayer RAW|网络倍数|Pad/Crop|
|---:|---|---|---|---|---:|---|
|P0|4:3 Photo Preview|1×4×768×1024|1×4×768×1024|2048×1536|H/W 均为 8 倍数|无|
|P1|16:9 Full-screen Preview|1×4×540×960|1×4×544×960|1920×1080|编译 H/W 均为 8 倍数|Packed 底部 Pad 4 行，输出裁回 540|
|P2|3:2 Preview/兼容流|1×4×640×960|1×4×640×960|1920×1280|H/W 均为 8 倍数|无|

表中 Bayer RAW 以 W×H 表示；Tensor Shape 以 H×W 表示。代码、Manifest 和日志必须显式使用 width/height 字段，禁止以位置猜测。

### 3.2 P1 Padding 规则

P1 的有效 Packed 高度 540 不能被三次 2× Downsample 的总倍率 8 整除，因此 Runtime 在 Packed RAW 域底部补 4 行：

~~~text
valid input : 4 × 540 × 960
compiled    : 4 × 544 × 960
padding     : bottom = 4, top/left/right = 0
valid output: crop [0:540, 0:960]
~~~

默认 Pad Mode 为 reflect；若目标转换器不支持静态 ReflectPad，则使用 edge replication。两者必须在固定验证集上 A/B，发布 Manifest 只记录最终一种。禁止使用常量 0 Pad，因为其会在底部制造不真实黑边噪声统计。

Padding 发生在已经拆为四个语义通道的 Packed 域，因此不会改变 CFA phase；输出 noise_pred 必须先裁回有效区，再与原始有效 RAW 相减。

### 3.3 Profile 选择

HAL 在 Camera Session 配置阶段协商主路 RAW Stream。选择逻辑：

1. 精确匹配有效 Packed W/H；
2. 若源尺寸更大，只允许上游通过相位安全 Binning/Crop 生成某个 Profile；
3. 不允许 Runtime 对 Packed RAW 做任意比例 Resize；
4. 若当前 Stream 无匹配 Profile，返回 UNSUPPORTED_PROFILE 并 Bypass；
5. Profile 由有效画幅决定，不由 Camera Type 决定。

主摄、超广和长焦均可使用 P0/P1/P2 中任一 Profile。Camera Type、Sensor Profile 和 Lens Profile 只进入 ConditionSchemaV2。

### 3.4 多摄和无缝变焦

V2.0 每帧只处理 HAL 标记的 Master Physical Camera：

- 单摄稳定区：正常执行 AI；
- 双摄预热但有稳定 Master：只处理 Master；
- 双摄融合过渡且 Master 不稳定：BYPASS_CAMERA_TRANSITION；
- 跨摄融合输出：不在本模块处理；
- 切换完成后，连续 3 帧 Sensor/Profile 稳定才重新启用 AI；
- Sensor 切换时重置 Dark Trigger 连续计数，但保留全局亮度状态；
- 若 Profile 同时变化，先完成静态 Buffer/OM 切换，再允许 AI 输出。

---

## 4. 输入、输出与 ConditionSchemaV2

### 4.1 RAW Normalize

每个 Packed 通道独立执行：

\[
x_c=\operatorname{clamp}\left(\frac{raw_c-black_c}{white\_level-black_c},0,1\right)
\]

要求：

- Black Level 必须为 R/Gr/Gb/B 四通道值；
- White Level 必须匹配当前 Sensor Mode 和 Bit Depth；
- white_level 必须大于每个 black_c；
- 输入不得含 NaN/Inf；
- 模型输出反归一化前先执行 finite/range 检查；
- Bypass 路径不得经过有损 Normalize/Denormalize。

### 4.2 模型接口

~~~text
Image input : INT8/FP16 logical tensor, N×4×Hcompiled×Wcompiled
Condition   : FP32, N×24
Noise output: INT8/FP16 logical tensor, N×4×Hcompiled×Wcompiled
Valid output: crop(noise output, valid_rect)
Enhanced    : clamp(valid image - valid noise, 0, 1)
~~~

量产图默认采用 INT8 QAT。文档中的 FP16 指逻辑或受控 NPU 混合精度，不表示允许 CPU/GPU 执行。

### 4.3 ConditionSchemaV2

Condition Tensor 固定为 FP32[1,24]，所有值进入模型前 Clamp 到 [0,1]。连续物理量使用明确的对数或线性归一化，类别使用 one-hot，禁止将 Sensor ID 的整数大小直接当作有序连续特征。

|索引|名称|原始范围/语义|模型值|
|---:|---|---|---|
|0|exposure_time_s|[1/16000, 1] 秒|对数归一化|
|1|iso|[50, 25600]|log2(iso/50) / 9|
|2|analog_gain|[1,64]×|log2(gain) / 6|
|3|digital_gain|[1,8]×|log2(gain) / 3|
|4|noise_level|归一化 RAW 标准差 [0,0.25]|value / 0.25|
|5|noise_shot_a|方差模型 a，范围 [1e-6,1e-1]|(log10(a)+6) / 5|
|6|noise_read_b|方差模型 b，范围 [1e-10,1e-3]|(log10(b)+10) / 7|
|7|sensor_temperature_c|[-20,100] 摄氏度|(value+20) / 120|
|8|scene_brightness|HAL/NLE 输出 [0,1]|直接使用|
|9|scene_ev|[-8,8]|(value+8) / 16|
|10|camera_main|Camera Type one-hot|0 或 1|
|11|camera_ultrawide|Camera Type one-hot|0 或 1|
|12|camera_tele|Camera Type one-hot|0 或 1|
|13|camera_other|Camera Type one-hot|0 或 1|
|14|sensor_profile_0|Sensor Profile one-hot|0 或 1|
|15|sensor_profile_1|Sensor Profile one-hot|0 或 1|
|16|sensor_profile_2|Sensor Profile one-hot|0 或 1|
|17|sensor_profile_other|Sensor Profile one-hot|0 或 1|
|18|lens_wide|Lens Profile one-hot|0 或 1|
|19|lens_ultrawide|Lens Profile one-hot|0 或 1|
|20|lens_tele|Lens Profile one-hot|0 或 1|
|21|lens_other|Lens Profile one-hot|0 或 1|
|22|metadata_valid|全部关键元数据有效|false/true → 0/1|
|23|enhancement_strength|Runtime 淡入淡出强度 [0,1]|直接使用|

exposure_time_s 的精确定义：

\[
c_0=\frac{\log_2(t)-\log_2(1/16000)}{\log_2(16000)}
\]

### 4.4 One-hot 约束

- camera_main～camera_other 必须恰有一个为 1；
- sensor_profile_0～sensor_profile_other 必须恰有一个为 1；
- lens_wide～lens_other 必须恰有一个为 1；
- 未知枚举必须进入对应 other，不得全零；
- Sensor Profile 数超过 3 时，V2.0 量产版本默认使用 other；扩展维度必须升级 Schema Major 版本；
- metadata_valid=0 时不直接调用 AI，除非产品批准使用完整 Sensor Profile 默认值；首发策略为 Bypass。

### 4.5 Noise Model

归一化 RAW 域采用：

\[
\operatorname{variance}(x)=a\cdot x+b
\]

V2.0 模型输入使用全局或当前 Camera 标定后的 a/b；训练数据内部仍应保留 R/Gr/Gb/B 四通道标定值。Runtime 将四通道值根据当前帧亮度计算成保守汇总值：

~~~text
noise_shot_a = max(a_R, a_Gr, a_Gb, a_B)
noise_read_b = max(b_R, b_Gr, b_Gb, b_B)
~~~

采用最大值是首发的保守策略，避免弱噪声通道掩盖强彩噪通道。后续若需要四通道独立值，升级 ConditionSchemaV3，不在 V2.0 中静默改变含义。

### 4.6 异常策略

- 任一关键连续值 NaN/Inf：Bypass；
- Black/White 非法：Bypass；
- one-hot 非法：Bypass；
- Condition Schema 版本不匹配：Bypass；
- 未标定 Sensor：默认 Bypass；
- enhancement_strength 不在 [0,1]：Clamp 并记录 INVALID_METADATA；
- 输入/输出发现非有限值：丢弃 AI 输出并原帧 Bypass。

---

## 5. Dark Trigger 与运行状态机

### 5.1 状态

~~~mermaid
stateDiagram-v2
    [*] --> BYPASS_BRIGHT
    BYPASS_BRIGHT --> ARMING: enter condition
    ARMING --> ACTIVE_RAMP: 3 consecutive frames
    ARMING --> BYPASS_BRIGHT: condition lost
    ACTIVE_RAMP --> ACTIVE: 5-frame strength ramp complete
    ACTIVE --> EXIT_PENDING: exit condition
    EXIT_PENDING --> ACTIVE: exit condition lost
    EXIT_PENDING --> BYPASS_BRIGHT: 10 consecutive frames
    ACTIVE --> BYPASS_ERROR: runtime/thermal/camera error
    ACTIVE_RAMP --> BYPASS_ERROR: runtime/thermal/camera error
    BYPASS_ERROR --> BYPASS_BRIGHT: subsystem recovered and reset
~~~

### 5.2 默认进入条件

满足下列任一条，累计 Enter Counter：

~~~text
dark_score >= 0.70
OR (ISO >= 1600 AND scene_ev <= -1.5)
OR noise_level >= 0.08
~~~

连续 3 帧成立后进入 ACTIVE_RAMP。dark_score 是 HAL/Scene Manager 的 [0,1] 暗光置信度，不属于 ConditionSchemaV2；若无该字段，则只使用 ISO、EV 和 Noise Level。

### 5.3 默认退出条件

以下条件全部成立，累计 Exit Counter：

~~~text
dark_score <= 0.45
AND ISO <= 1200
AND scene_ev >= -1.0
AND noise_level <= 0.06
~~~

连续 10 帧成立后退出。处于进入与退出阈值之间时保持当前状态，不反复切换。

### 5.4 强度淡入淡出

进入时 enhancement_strength 在 5 帧内使用：

~~~text
0.20, 0.40, 0.60, 0.80, 1.00
~~~

正常退出时反向使用同一序列。异常、OOM、超时、非有限输出和 Camera Transition 不等待淡出，立即 Bypass。强度参数进入模型 FiLM，同时允许 Runtime 对最终 noise_pred 再乘同一强度，但禁止两处重复衰减；V2.0 冻结为只在模型输入 Condition 中使用强度，图外不再二次乘。

### 5.5 可配置项

所有阈值、连续帧数和强度序列写入版本化 Runtime Manifest。修改阈值属于 MINOR 版本更新，必须重跑触发稳定性和画质回归；不得散落在 HAL 和 AI Runtime 的硬编码常量中。

---

## 6. Student 网络结构

### 6.1 模型命名

~~~text
Tiny Conditional MobileNAFNet-W16 Dark Preview V2
~~~

“MobileNAFNet”表示继承 NAFNet 的 U-Net 多尺度、SimpleGate 和残差恢复思想，但按移动 NPU 算子白名单改造；它不是官方 NAFNet 代码的原样部署。

### 6.2 拓扑冻结

~~~yaml
image_channels: 4
base_width: 16
encoder_blocks: [2, 2, 4]
middle_blocks: 2
decoder_blocks: [2, 2, 2]
feature_channels: [16, 32, 64, 128]
condition_dim: 24
film_injection: [encoder_stage_2, encoder_stage_3, middle]
output_semantics: noise_pred
~~~

### 6.3 总体结构

~~~mermaid
flowchart LR
    X["Packed RAW<br/>4×H×W"] --> I["Intro 3×3 Conv<br/>16×H×W"]
    I --> E0["Encoder 1<br/>2× MobileNAFBlock<br/>16×H×W"]
    E0 --> D0["3×3 Conv s2<br/>32×H/2×W/2"]
    D0 --> E1["Encoder 2 + FiLM<br/>2× MobileNAFBlock<br/>32×H/2×W/2"]
    E1 --> D1["3×3 Conv s2<br/>64×H/4×W/4"]
    D1 --> E2["Encoder 3 + FiLM<br/>4× MobileNAFBlock<br/>64×H/4×W/4"]
    E2 --> D2["3×3 Conv s2<br/>128×H/8×W/8"]
    D2 --> M["Middle + FiLM<br/>2× MobileNAFBlock<br/>128×H/8×W/8"]
    M --> U2["Nearest ↑2 + 3×3 Conv<br/>64×H/4×W/4"]
    E2 --> A2(("+"))
    U2 --> A2 --> R2["Decoder 3<br/>2× MobileNAFBlock"]
    R2 --> U1["Nearest ↑2 + 3×3 Conv<br/>32×H/2×W/2"]
    E1 --> A1(("+"))
    U1 --> A1 --> R1["Decoder 2<br/>2× MobileNAFBlock"]
    R1 --> U0["Nearest ↑2 + 3×3 Conv<br/>16×H×W"]
    E0 --> A0(("+"))
    U0 --> A0 --> R0["Decoder 1<br/>2× MobileNAFBlock"]
    R0 --> O["Ending 3×3 Conv<br/>4×H×W noise_pred"]
    C["Condition 24"] --> CE["Condition Encoder<br/>24→64→128"]
    CE -.-> E1
    CE -.-> E2
    CE -.-> M
~~~

### 6.4 MobileNAFBlock

每个 Block 使用两个残差子块：

~~~text
Spatial branch:
Input
→ Conv1×1 C→2C
→ DWConv3×3 2C
→ Static Slice into C/C
→ Elementwise Mul (SimpleGate)
→ Conv1×1 C→C
→ residual scale beta
→ Add

Channel branch:
Input
→ Conv1×1 C→2C
→ Static Slice into C/C
→ Elementwise Mul
→ Conv1×1 C→C
→ residual scale gamma
→ Add
~~~

beta/gamma 初始化为 0。部署 Student 不使用 LayerNorm、AdaptiveAvgPool/SCA、动态 Split 或 PixelShuffle。若训练稳定性不足，优先调整初始化、Learning Rate、Gradient Clip 和 Warmup，不在未验证情况下重新引入部署高风险算子。

### 6.5 FiLM

Condition Encoder：

~~~text
FP32[24]
→ Linear 24→64 + ReLU
→ Linear 64→128 + ReLU
→ Head E1: 128→64    (gamma/beta 各 32)
→ Head E2: 128→128   (gamma/beta 各 64)
→ Head M : 128→256   (gamma/beta 各 128)
~~~

FiLM 公式：

\[
F'=F\cdot(1+0.1\tanh(\gamma))+0.1\tanh(\beta)
\]

FiLM 分别注入 Encoder Stage 2、Encoder Stage 3 和 Middle 的第一个 Block 之前。Head 权重零初始化，使初始模型接近无条件主干。

### 6.6 输出和符号

模型只输出 noise_pred，不直接输出 clean RAW：

\[
raw_{out}=\operatorname{clamp}(raw_{in}-noise_{pred},0,1)
\]

训练标签为：

\[
noise_{gt}=raw_{noisy}-raw_{clean}
\]

所有变量命名统一使用 noise_pred/noise_gt。禁止把同一 Tensor 在不同模块中命名为 positive_residual，避免符号反转。

### 6.7 参数和模型大小口径

工程预算：

- 未剪枝 Student 主干加 Condition 分支约 0.55M～0.80M 参数；
- INT8 原始权重载荷目标小于 0.8MB；
- 含 Q/DQ、常量、图结构和 Runtime 元数据的单 OM 文件可能大于纯权重载荷；
- “1MB 模型”在本文中指权重预算，不承诺 OM 容器物理文件恰好小于 1MB。

参数数、MACs 和各 Profile 激活峰值必须由冻结 ONNX 自动生成，不得以手工估算作为 Release 值。

### 6.8 算子白名单

|算子|要求|
|---|---|
|Conv 1×1/3×3|必须 NPU 下沉；权重优先 INT8 per-channel|
|DWConv 3×3|必须独立做最小图转换和端侧性能测试|
|Static Slice/View|Slice 参数必须为编译常量|
|Elementwise Mul/Add|必须 NPU 下沉，不允许 Host 执行|
|Nearest Resize 2×|固定倍率、固定坐标模式|
|Static Pad/Crop|P1 固定 4 行；参数不得运行时动态变化|
|Linear|允许转换为 MatMul 或 1×1 Conv，必须 NPU 下沉|
|Tanh|优先由受支持算子执行；失败时使用固定有界近似并重训|
|Clamp|优先在 NPU 或固定后处理 Kernel 执行|

完整图出现任何未批准 CPU/GPU 回退即发布阻断。若 DWConv 或 SimpleGate 无法高效下沉，Fallback 架构为同宽度 KBlock-R；Fallback 必须重新训练、蒸馏、剪枝和 QAT，不允许运行时动态切换 Block。

---

## 7. Teacher、数据和噪声模型

### 7.1 Teacher

Teacher 使用 Conditional NAFNet-W32：

~~~yaml
image_channels: 4
base_width: 32
encoder_blocks: [2, 2, 4, 8]
middle_blocks: 4
decoder_blocks: [2, 2, 2, 2]
condition_dim: 24
output_semantics: noise_pred
~~~

Teacher 可保留官方 NAFBlock 的 LayerNorm、SimpleGate 和 SCA，只在 GPU FP32 训练与蒸馏中使用，不参与麒麟 9000 部署。

### 7.2 数据组成

|数据|目标占比|用途|
|---|---:|---|
|目标设备真实 Noisy/Clean 配对|≥ 50%|主要质量来源|
|目标设备静态 Burst 鲁棒平均参考|≥ 20%|极暗和高 ISO|
|目标设备物理噪声合成|≤ 20%|覆盖稀有曝光/温度|
|公开 RAW 数据|≤ 10%|预训练和泛化，不替代目标设备数据|

每个首发物理 Camera 的最低场景数：

|划分|最低场景数|
|---|---:|
|训练|3000|
|验证|300|
|独立盲测|500|

同一场景、Burst、地点或连续拍摄片段只能进入一个划分。多台同型号设备应跨设备划分，验证 Sensor 个体差异。

### 7.3 场景覆盖

- ISO：800、1200、1600、3200、6400、12800 及设备上限；
- 照度：0.1、0.5、1、5、10、50 lux 分桶；
- 曝光时间、模拟/数字增益组合；
- 主摄、超广、长焦及其他首发 Camera；
- P0/P1/P2 三种有效格点；
- 温度：冷机、常温、热稳态；
- 人脸、头发、织物、文字、树叶、砖墙、灯牌、车灯、天空渐变；
- 固定图样噪声、行列噪声、热像素和弱纹理；
- Sensor 切换前后和 Trigger 边界附近。

### 7.4 Clean Reference

优先级：

1. 静态夹具下低 ISO 长曝光参考；
2. 同场景短 Burst 亚像素配准、异常帧剔除和鲁棒平均；
3. 高质量离线 Teacher 仅用于辅助，不可作为唯一 GT；
4. 无可靠参考的动态样本用于主观/无参考评测，不进入强监督 RAW Loss。

Reference 生成必须记录 Sensor、Lens、ISP 前端配置、温度、曝光、Black/White、配准置信度和生成脚本版本。

### 7.5 物理噪声合成

基础模型：

\[
y=\operatorname{clip}\left(\frac{\operatorname{Poisson}(x/a)\cdot a+n_r+n_{row}+n_{col}}{1},0,1\right)
\]

其中：

- Shot Noise 由 a 控制；
- Read Noise 由 b 和非高斯尾部拟合；
- Row/Column Noise 使用 Sensor 实测频谱；
- Hot/Dead Pixel 按温度和曝光分桶采样；
- 量化噪声匹配 RAW10/RAW12；
- 通道相关性从暗场/平场数据估计；
- 合成参数从目标 Sensor Profile 分布采样，不使用无界随机值。

### 7.6 上游传统 NR 残留

若量产输入位于轻量传统 RAW NR 之后，数据需覆盖：

- 不同传统 NR 强度；
- 残留彩噪；
- 暗部涂抹；
- 边缘周围的轻度 Halo；
- 行列噪声未完全去除；
- Sensor Mode 切换时的参数变化。

AI 训练目标仍然是保真降噪，不允许以生成纹理修补上游重度涂抹。

### 7.7 Patch 与增强

- Teacher/Student 初训 Patch：Packed 256、384 混合；
- 后期加入 Packed 512 Patch 和完整 Profile 小批验证；
- Crop 起点必须保持 Packed 格点；
- 旋转/镜像后重新映射 R/Gr/Gb/B 语义；
- 不使用改变 RAW 物理含义的 RGB Color Jitter；
- 曝光缩放必须同步更新 Shot/Read/ISO Condition；
- enhancement_strength 从 0、0.25、0.5、0.75、1.0 分层采样，其中强度 0 的恒等样本不少于 10%，强度 1 的完整降噪样本不少于 50%；
- 强度为 s 时监督目标固定为 noise_gt_s = s × (raw_noisy - raw_clean)，保证 s=0 输出零噪声、s=1 输出完整预测，并对中间强度施加单调性检查；
- 每个 Batch 平衡 Camera、ISO、照度和 Profile。

---

## 8. Loss、训练与蒸馏

### 8.1 RAW Loss

使用饱和区 Mask 和暗部加权 Charbonnier：

\[
L_{RAW}=\frac{\sum M_{sat}\cdot w_{shadow}\cdot\sqrt{(raw_{out}-raw_{gt})^2+\epsilon^2}}{\sum M_{sat}\cdot w_{shadow}}
\]

默认：

~~~text
epsilon = 1e-3
M_sat = 1 when raw_gt < 0.98, otherwise 0
w_shadow = 1 + 2 × (1 - mean_channel(raw_gt))
~~~

饱和区不参与强像素拟合，但必须单独检查高光是否出现颜色扩散。

### 8.2 Tone Loss

使用固定、可微、版本化的 Reference ISP：

~~~text
Packed RAW
→ Demosaic
→ Fixed WB
→ Fixed CCM
→ log1p tone
→ RGB/Luma L1
~~~

Tone Loss 只用于约束显示域暗部、色彩和渐变，不允许使用 GAN 或高权重感知 Loss 生成 RAW 中不存在的内容。

### 8.3 Gradient Loss

对固定 ISP 输出的线性 Luma 使用 Sobel X/Y：

\[
L_{Gradient}=|G_x(\hat{Y})-G_x(Y)|_1+|G_y(\hat{Y})-G_y(Y)|_1
\]

梯度统计按暗部、纹理和边缘分桶报告，避免平均值掩盖过平滑。

### 8.4 Feature KD

Teacher 和 Student 在 Encoder Stage 3 与 Middle 对齐：

- 使用训练期 1×1 Adapter；
- 空间尺寸不一致时只使用固定 2× Average Pool；
- Feature 先按通道 RMS Normalize；
- 采用 L1；
- Adapter 不导出到部署模型。

### 8.5 Loss 权重

Teacher 和 Student 独立训练：

\[
L=0.55L_{RAW}+0.30L_{Tone}+0.15L_{Gradient}
\]

蒸馏、剪枝恢复和 QAT：

\[
L=0.50L_{RAW}+0.25L_{Tone}+0.15L_{Gradient}+0.10L_{FeatureKD}
\]

各项 Loss 必须分别记录未加权值、加权值、梯度范数和分场景曲线。

### 8.6 默认训练顺序

|阶段|初始化|默认步数|初始 LR|退出条件|
|---|---|---:|---:|---|
|Teacher FP32|随机|500k|2e-4|验证集 20k step 无提升|
|Student FP32|随机|400k|2e-4|稳定收敛并通过算子图导出|
|Student KD|Student FP32|200k|1e-4|相对 Teacher 差距达到门槛|
|P10/P15 恢复|KD Student|各 80k|5e-5|质量恢复并导出成功|
|INT8 QAT|各候选 FP32|60k|1e-5|量化差距稳定并通过暗部门槛|

默认优化器 AdamW，weight_decay=1e-4，Gradient Clip=1.0，Cosine LR，前 5k step Warmup。实际 Batch Size 根据 GPU 显存设置，但每个有效 Batch 的 Camera/ISO/Profile 采样比例必须一致。

### 8.7 蒸馏门槛

- KD Student 相对 Teacher RAW PSNR 下降 ≤ 0.20dB；
- SSIM 下降 ≤ 0.003；
- 固定 ISP RGB 平均 ΔE00 ≤ 1.0、P95 ≤ 2.0；
- 不得在任一首发 Camera、Profile 或 ISO 大类下降超过 0.35dB；
- 过平滑率、虚假纹理和暗部 Banding 不得高于非 KD Student。

---

## 9. 结构化剪枝

### 9.1 定位

W16 已经是轻量对齐结构。剪枝是候选优化，不是为了满足名义压缩率而强制破坏通道对齐。候选：

~~~text
P0 : 未剪枝 W16
P10: 实际参数下降 8%～12%
P15: 实际参数下降 13%～17%
~~~

### 9.2 依赖组

以下通道必须成组同步：

- Encoder 输出、对应 Down 输入和 Skip；
- Decoder 对应 Up 输出和 Skip Add；
- MobileNAFBlock 内 1×1、DWConv、Static Slice 两半和投影；
- FiLM gamma/beta Head；
- QAT Observer 和 Scale；
- Teacher Adapter 只在训练态同步。

### 9.3 对齐规则

- Intro/Ending 不剪；
- Encoder Stage 1 的 16 通道默认不剪；
- 其余通道以 8 为最小剪枝组；
- SimpleGate 展开通道必须保持两半相等；
- Skip Add 两侧通道完全相同；
- 不允许非结构化稀疏作为量产加速依据；
- 候选名称按实际参数下降比例生成，不按单层通道比例命名。

### 9.4 重要度

组合指标：

\[
I_g=0.5I_{Taylor}+0.3I_{BN/Scale}+0.2I_{Activation}
\]

Student 无 BN 时，Scale 项使用 residual beta/gamma 和相邻卷积通道范数。重要度在多 Camera、三 Profile、暗光分桶的固定校准集上计算，禁止只用单一 Camera。

### 9.5 渐进流程

1. 从 KD Student 开始；
2. 每轮只移除一个 8 通道依赖组；
3. 每轮短恢复 10k step；
4. 达到 P10/P15 目标区间后完整恢复 80k step；
5. 导出静态 ONNX，完成三个 Shape 的图验证；
6. 分别执行 QAT；
7. 真机比较画质、P95、内存和功耗。

### 9.6 自动选择规则

候选必须先通过所有画质、算子和稳定性门槛，然后按以下规则选择：

1. P15 相对 P0 的 P95 改善 ≥ 8%，且增量峰值内存改善 ≥ 8%，选择 P15；
2. 否则 P10 相对 P0 的 P95 改善 ≥ 5%，选择 P10；
3. 否则选择未剪枝 P0；
4. 任一 Profile 的画质或 P95 回归超限，淘汰该候选；
5. 三个 OM 必须使用同一候选权重，不允许各 Profile 发布不同剪枝拓扑。

---

## 10. INT8 QAT

### 10.1 默认量化

- Conv/DWConv 权重：INT8 symmetric per-channel；
- 激活：INT8 per-tensor；
- Accumulator：INT32；
- 输出 Requant：按目标 DDK 支持方式；
- Condition 输入：FP32 API，图内转换到受支持精度；
- FiLM：优先全 INT8 或 NPU 原生量化算子；
- Calibration/QAT 数据覆盖全部 Camera、Profile、ISO、照度和温度。

### 10.2 Observer

- 权重使用对称 MinMax/Moving Average；
- 激活使用 Percentile 或 Moving Average，默认 99.99%；
- 暗部校准样本占比不得低于 60%；
- 高光样本单独保留，避免暗部范围压缩导致饱和异常；
- P0/P1/P2 共用权重和 Observer 统计，不为 Profile 生成独立权重。

### 10.3 QAT 门槛

- INT8 相对对应 FP32 候选 RAW PSNR 下降 ≤ 0.10dB；
- SSIM 下降 ≤ 0.002；
- RGB 平均 ΔE00 增量 ≤ 0.3；
- 暗部 Banding、色阶跳变和天空间断层无可见新增；
- 三 Profile PyTorch QAT、ONNX Q/DQ 和端侧结果进入同一数值报告；
- NPU 覆盖 100%。

### 10.4 混合精度例外

只有在全 INT8 未通过暗部画质门槛且真机证明确有收益时，允许 Condition Encoder/FiLM 使用 FP16 NPU Island。要求：

- 不产生 CPU/GPU 回退；
- Manifest 标记 quantization=mixed_int8_fp16；
- 端到端 P95 仍通过；
- 增量内存仍通过；
- 有正式架构审批记录；
- 不能用混合精度掩盖输入校准或 QAT 数据问题。

---

## 11. ONNX、OM 与运行时部署

### 11.1 三个静态 ONNX

~~~text
dark_preview_p0.onnx
  image     [1,4,768,1024]
  condition [1,24]
  noise     [1,4,768,1024]

dark_preview_p1.onnx
  image     [1,4,544,960]
  condition [1,24]
  noise     [1,4,544,960]

dark_preview_p2.onnx
  image     [1,4,640,960]
  condition [1,24]
  noise     [1,4,640,960]
~~~

三个 ONNX：

- 来自同一冻结 checkpoint；
- 网络拓扑和算子语义完全一致；
- 只允许输入 H/W 和由此推导的常量 Shape 不同；
- 分别生成 ONNX Hash；
- 共享 weights_hash；
- 不使用 dynamic_image_size 作为量产主路径。

### 11.2 OM 命名

~~~text
dark_preview_p0_4x768x1024_int8.om
dark_preview_p1_4x544x960_int8.om
dark_preview_p2_4x640x960_int8.om
~~~

目标产品若不使用 OM，而使用特定 HiAI/MindSpore Lite/NNRT 制品，保持相同的三静态图语义并替换扩展名；实际 Runtime 栈在 P0 阶段冻结。

### 11.3 加载策略

Camera Session 创建时：

1. 读取 Manifest 并校验 Schema、Runtime、Sensor Profile 和 Hash；
2. 根据 App 支持的预览画幅预加载常用 OM；
3. 主力 P1 必须常驻；
4. 内存允许时三个 OM 全常驻；
5. 内存不允许时采用 P1+当前拍照画幅双常驻；
6. 不允许在 ACTIVE 帧的关键路径同步加载模型；
7. Profile 切换需在 AI Bypass 窗口内完成并预热 3 次。

### 11.4 C++ API

~~~cpp
enum class DarkPreviewProfileId : uint32_t {
    P0_4_3 = 0,
    P1_16_9 = 1,
    P2_3_2 = 2,
};

enum class CfaPattern : uint32_t {
    RGGB = 0,
    BGGR = 1,
    GRBG = 2,
    GBRG = 3,
};

struct DarkPreviewDenoiseInput {
    const void* packed_raw;
    uint32_t plane_stride_bytes;
    uint32_t valid_width;
    uint32_t valid_height;
    uint32_t bit_depth;
    CfaPattern cfa_pattern;
    DarkPreviewProfileId profile_id;
    float condition[24];
    uint64_t frame_id;
    uint64_t timestamp_ns;
};

struct DarkPreviewDenoiseOutput {
    void* enhanced_packed_raw;
    uint32_t output_stride_bytes;
    int32_t status;
    uint32_t fallback_mask;
    float format_ms;
    float npu_ms;
    float post_ms;
    float total_ms;
};

int DarkPreviewDenoiseProcess(
    const DarkPreviewDenoiseInput* input,
    DarkPreviewDenoiseOutput* output);
~~~

输入 packed_raw 的具体 Plane Layout、元素类型和 Alignment 由 Manifest 冻结。首发默认四平面 uint16 容器承载 RAW10/RAW12 有效位，Runtime 内部转换到模型量化格式。

### 11.5 状态码

|状态|含义|输出|
|---|---|---|
|OK|AI 成功|增强 RAW|
|BYPASS_BRIGHT|未触发暗光|原始 RAW|
|BYPASS_CAMERA_TRANSITION|多摄过渡|原始 RAW|
|INVALID_ARGUMENT|指针、Stride、尺寸非法|原始 RAW 或调用失败|
|UNSUPPORTED_PROFILE|无静态图|原始 RAW|
|INVALID_METADATA|关键元数据非法|原始 RAW|
|SCHEMA_MISMATCH|Condition 版本不匹配|原始 RAW|
|HASH_MISMATCH|模型/权重/Profile 不匹配|原始 RAW|
|NPU_UNAVAILABLE|NPU/Runtime 不可用|原始 RAW|
|NPU_TIMEOUT|推理超时|原始 RAW|
|OUT_OF_MEMORY|Buffer/Workspace 失败|原始 RAW|
|INFERENCE_ERROR|模型执行失败|原始 RAW|
|NONFINITE_OUTPUT|输出 NaN/Inf|原始 RAW|
|THERMAL_BYPASS|热策略关闭 AI|原始 RAW|

### 11.6 Fallback Mask

~~~text
bit 0 : bright scene
bit 1 : camera transition
bit 2 : unsupported profile
bit 3 : invalid metadata
bit 4 : schema/hash mismatch
bit 5 : NPU unavailable
bit 6 : timeout
bit 7 : OOM
bit 8 : inference error
bit 9 : non-finite output
bit 10: thermal
bit 11: unapproved backend fallback detected
~~~

### 11.7 Manifest

~~~json
{
  "model_name": "TinyConditionalMobileNAFNetW16DarkPreview",
  "model_version": "2.0.0",
  "condition_schema_version": 2,
  "condition_dim": 24,
  "input_layout": "NCHW",
  "input_channel_order": ["R", "Gr", "Gb", "B"],
  "input_domain": "black_subtracted_normalized_linear_raw",
  "output_semantics": "noise_pred",
  "reconstruction": "clamp(input-noise_pred,0,1)",
  "weights_hash": "sha256:...",
  "quantization": "int8_qat",
  "pruning_profile": "p0|p10|p15",
  "profiles": [
    {
      "id": 0,
      "name": "p0_4_3",
      "om": "dark_preview_p0_4x768x1024_int8.om",
      "shape": [1, 4, 768, 1024],
      "valid_rect": [0, 0, 1024, 768],
      "pad_mode": "none",
      "p95_budget_ms": 30
    },
    {
      "id": 1,
      "name": "p1_16_9",
      "om": "dark_preview_p1_4x544x960_int8.om",
      "shape": [1, 4, 544, 960],
      "valid_rect": [0, 0, 960, 540],
      "pad_mode": "reflect_bottom_4",
      "p95_budget_ms": 25
    },
    {
      "id": 2,
      "name": "p2_3_2",
      "om": "dark_preview_p2_4x640x960_int8.om",
      "shape": [1, 4, 640, 960],
      "valid_rect": [0, 0, 960, 640],
      "pad_mode": "none",
      "p95_budget_ms": 27
    }
  ],
  "trigger": {
    "enter_dark_score": 0.70,
    "exit_dark_score": 0.45,
    "enter_iso": 1600,
    "exit_iso": 1200,
    "enter_ev": -1.5,
    "exit_ev": -1.0,
    "enter_noise_level": 0.08,
    "exit_noise_level": 0.06,
    "enter_frames": 3,
    "exit_frames": 10,
    "ramp_frames": 5
  },
  "sensor_profile_hash": "sha256:...",
  "runtime_min_version": "2.0.0"
}
~~~

---

## 12. 性能、内存与功耗设计

### 12.1 帧预算

30fps 理论帧周期为 33.33ms。P1 端到端 P95 25ms 的初始分解目标：

|阶段|目标|
|---|---:|
|Formatter/Normalize/Pad|≤ 4ms|
|NPU 推理|≤ 15ms|
|Crop/Subtract/Clamp|≤ 4ms|
|同步与余量|≤ 2ms|
|总计|≤ 25ms|

P0/P2 允许更高预算，但必须低于各自 P95 硬门槛。分项预算只用于定位，最终验收以同一帧的真实端到端 Timeline 为准。

### 12.2 静态内存

Session 创建阶段预分配：

- 三个或两个 OM 常驻内存；
- 每个活跃 Profile 的输入、输出 Buffer；
- 单个静态 Workspace；
- Formatter 临时 Buffer；
- Condition Buffer；
- Bypass/原始 RAW 引用；
- Timeline 和错误记录 Ring Buffer。

每帧不得进行大块 Heap 分配/释放。Buffer 使用固定生命周期和 Ping/Pong 复用，但每帧仍只提交一个 Full Tensor。

### 12.3 并发

V2.0 不以多 NPU Stream 并发处理同一帧。允许 CPU Formatter、DMA、NPU 和 Post 在相邻帧间形成流水，但必须保证：

- 输出帧序；
- 每帧独立输入/输出 Buffer；
- 不覆盖尚未消费的结果；
- Pipeline Depth 固定且写入 Runtime 配置；
- 最大排队帧数为 1，超过时优先 Bypass 最新策略由 Camera Pipeline 冻结；
- 不以累积延迟换取平均吞吐。

### 12.4 热与功耗

测试至少覆盖：

- 冷启动前 30 秒；
- 持续 2 分钟；
- 持续 10 分钟；
- 30 分钟稳定性；
- 屏幕高亮、相机 OIS/AF 活跃和后台典型负载；
- 不同环境温度。

温控策略触发时，优先完整 Bypass AI，不动态改 Shape、通道或量化精度。热恢复后重新进入 ARMING，禁止立即恢复满强度。

---

## 13. 工程问题、检测与降级

|问题|表现|根因|检测|处理|
|---|---|---|---|---|
|CFA 错误|偏色、Gr/Gb 纹理异常|Pack/Crop/旋转相位错误|四 CFA 合成图|统一语义重排；异常 Bypass|
|RAW 格点错配|Zipper、False Color|低分辨率残差插值|棋盘/单色边缘|禁止插值；输入/输出同格点断言|
|P1 Pad 错误|底边亮暗带|常量 Pad 或 Crop 错|底部边界集|Reflect/Edge；固定 valid_rect|
|Profile 误选|Shape 错误、拉伸|按 Camera 名称绑定|多摄/画幅切换测试|按有效 W/H 选择|
|Black/White 错误|抬黑、截断|Mode 元数据不匹配|灰阶/暗场|版本化 Sensor Profile；Bypass|
|Noise Profile 失配|残噪或过平滑|标定不足|ISO/温度分桶|重标定；other 默认 Bypass|
|Trigger 抖动|画面呼吸、闪烁|无滞回|Lux Ramp|3/10 帧滞回与 5 帧淡入|
|Camera 切换跳变|画质突变|双摄过渡不稳定|Zoom Sweep|Master-only；过渡 Bypass|
|DWConv 不下沉|时延激增|目标 DDK 不支持|最小图和落点报告|冻结 KBlock-R Fallback 并重训|
|SimpleGate 回退|CPU 使用升高|动态 Split|图审计|Static Slice；失败用 ReLU Block|
|Tanh 不支持|编译失败|FiLM 算子不兼容|Condition 最小图|有界近似并重训|
|INT8 Banding|天空断层|Observer 被高光主导|暗部 Histogram/FFT|暗部校准、QAT、受控 FP16 Island|
|剪枝无加速|参数降但 P95 不变|对齐/带宽瓶颈|P0/P10/P15 真机 A/B|自动选择 P0|
|OM 切换卡顿|Zoom/画幅掉帧|关键路径加载|Session Timeline|预加载、预热、切换期 Bypass|
|热降频|P95 上升|持续 NPU 负载|10/30 分钟曲线|THERMAL_BYPASS|
|输出非有限|花屏/下游异常|量化或输入异常|Finite Check|丢弃 AI 输出，原帧 Bypass|
|版本错配|崩溃或静默错误|Schema/Hash 不一致|启动强校验|拒绝加载、回滚|

### 13.1 失败安全原则

AI 是可选增强节点，不得成为 Camera Preview 可用性的单点故障。任一错误发生时：

1. 不发布当前 AI 输出；
2. 输出原始 Packed RAW 引用或位精确复制；
3. 设置 Status 和 Fallback Mask；
4. 记录非图像隐私遥测；
5. 对连续错误执行熔断；
6. 熔断恢复后从 ARMING 重新开始；
7. 禁止自动切换到 CPU AI 推理。

连续 3 帧 NPU_TIMEOUT、INFERENCE_ERROR 或 NONFINITE_OUTPUT 时，本 Camera Session 熔断 AI；仅 Session 重建或显式 Runtime Reset 后恢复。

---

## 14. 测试与验收矩阵

### 14.1 接口和数值正确性

- RGGB/BGGR/GRBG/GBRG；
- RAW10/RAW12 和 uint16 容器；
- 每通道 Black Level、不同 White Level；
- P0/P1/P2 Shape；
- P1 540→544 Pad 和 544→540 Crop；
- 奇数上游尺寸必须在 HAL 拒绝或偶数 Crop；
- Pack/Unpack 位精确可逆；
- Bypass 位精确；
- PyTorch FP32→ONNX 最大绝对误差 ≤ 1e-4；
- ONNX QAT→端侧误差进入量化报告；
- Condition 24 维索引、Clamp 和 one-hot；
- Schema/Hash/Profile/Stride 错误；
- NaN/Inf、全黑、全白、饱和和坏点；
- Frame ID、时间戳和输出帧序。

### 14.2 模型阶段质量预算

|阶段|参考|RAW PSNR 最大下降|SSIM 最大下降|
|---|---|---:|---:|
|KD Student FP32|W32 Teacher|0.20dB|0.003|
|P10/P15 恢复|KD Student|0.10dB|0.002|
|INT8 QAT|对应 Pruned/Unpruned FP32|0.10dB|0.002|
|最终 INT8|W32 Teacher|0.35dB|0.005|

任何单 Camera、单 Profile 或高 ISO 大类相对 Teacher 下降超过 0.50dB，均不得以全局平均值通过。

### 14.3 画质指标

分别按 Camera、Profile、ISO、Lux、曝光、温度和场景报告：

- RAW PSNR/SSIM；
- 固定 ISP RGB PSNR/SSIM/MS-SSIM；
- ΔE00 Mean/P95；
- Noise Power Spectrum；
- Row/Column Noise；
- Edge MTF 和梯度保持；
- 暗部纹理保留率；
- 过平滑率；
- Hot/Dead Pixel 修复率和误伤率；
- Banding、色斑、Halo、Ringing、Maze、False Color；
- 文字错误率和虚假纹理；
- Trigger 前后相邻帧亮度/噪声变化。

RGB 色彩门槛：平均 ΔE00 ≤ 1.0，P95 ≤ 2.0。极暗样本可单独报告，但不得出现系统性偏色。

### 14.4 主观评测

采用双盲 A/B/C：

- Baseline ISP；
- Final AI；
- Clean/高质量 Reference。

至少 5 名评审者，覆盖人脸肤色、头发、织物、文字、树叶、夜景灯牌、天空渐变、极暗、车灯和 Sensor 切换。记录偏好率与缺陷标签，不以单一 MOS 掩盖严重伪影。

### 14.5 性能测试

每个 Profile：

- 冷/热 P50、P90、P95、P99；
- Formatter、DMA、NPU、Post 和总时延；
- 模型加载、Profile 切换和预热；
- 系统内存、NPU Workspace 和模型常驻；
- NPU/CPU/GPU 算子比例；
- 10/30 分钟能耗、温升和降频；
- 低/中/高屏幕亮度；
- 后台典型负载；
- 至少 10,000 帧稳定性；
- 触发边界和 Camera 切换期间的掉帧。

### 14.6 Trigger 测试

- 线性 Lux 上升/下降；
- ISO 在 1200～1600 反复波动；
- EV 在 -1.0～-1.5 波动；
- Noise Level 在 0.06～0.08 波动；
- 5 帧淡入淡出；
- enhancement_strength=0 时输出与 Bypass 的最大归一化误差不超过 1e-5，中间强度的降噪量随强度单调增加；
- Camera Transition 打断；
- Thermal Bypass 和恢复；
- Metadata 短时缺失；
- 连续错误熔断。

要求：稳定光照下不得出现周期性开关；淡入淡出期间固定 ISP 输出无可见亮度跳变和色彩跳变。

### 14.7 Go/No-Go

全部满足才允许发布：

1. 三 Profile 画质门槛通过；
2. 三 Profile 端到端 P95/P99 通过；
3. 10 分钟 30fps、掉帧和热退化通过；
4. 增量峰值内存通过；
5. NPU 覆盖 100%；
6. 未批准 CPU/GPU 回退为 0；
7. Trigger、Camera 切换和失败安全通过；
8. Model/Schema/Sensor/Runtime 版本一致；
9. 回滚包验证通过；
10. 阻断级和高优先级缺陷为 0。

---

## 15. 开发阶段与交付物

|阶段|周期|主要工作|必须输出|退出条件|
|---|---:|---|---|---|
|P0 需求/设备/DDK 冻结|2 周|目标机、Camera、RAW Stream、Runtime 栈、权限/IP|PRD、接口、设备矩阵、算子 Spike 计划|三 Profile 和责任人明确|
|P1 CPU/ONNX PoC|4 周|Pack、Profile、Condition、MobileNAFBlock、Full Tensor|CPU Pipeline、三静态 ONNX、小图转换报告|核心单测 100%，ONNX 误差通过|
|P2 数据与标定|6 周|目标 Sensor 数据、Noise Profile、划分|Dataset V1、Sensor Profile、Data Card|每 Camera 最低数据量和泄漏检查通过|
|P3 Teacher/Student FP32|6～8 周|W32 Teacher、W16 Student、消融|Teacher、Student、质量/算子报告|Student 质量门槛通过|
|P4 KD/剪枝/QAT|6 周|KD、P10/P15、恢复、QAT|P0/P10/P15 FP32/INT8 候选|阶段质量预算通过|
|P5 麒麟 9000 集成|6 周|三 OM、Runtime、预加载、Trigger、Profile 切换|端侧库、Manifest、性能报告|三 Profile 硬门槛通过|
|P6 Camera 闭环/DVT|4～6 周|盲测、温升、异常、灰度、回滚|RC、Model Card、量产报告|Go/No-Go 全部通过|

总周期预算 28～34 周。目标硬件、DDK、RAW 权限或配对数据晚到时，项目按依赖到位日顺延，不压缩端侧和 DVT 验证周期。

### 15.1 制品命名

~~~text
artifacts/
├── teacher/conditional_nafnet_w32_fp32_<data_version>.pth
├── student/mobile_nafnet_w16_full_fp32_<run_id>.pth
├── student/mobile_nafnet_w16_kd_fp32_<run_id>.pth
├── pruning/p0|p10|p15/model_fp32.pth
├── quant/p0|p10|p15/model_int8_qat.pth
├── onnx/dark_preview_p0.onnx
├── onnx/dark_preview_p1.onnx
├── onnx/dark_preview_p2.onnx
├── release/dark_preview_p0_4x768x1024_int8.om
├── release/dark_preview_p1_4x544x960_int8.om
├── release/dark_preview_p2_4x640x960_int8.om
├── release/model_manifest.json
├── release/condition_schema_v2.json
├── release/sensor_profiles.json
└── reports/quality|performance|fallback|stability
~~~

---

## 16. 代码与工程组织建议

~~~text
AI-ISP/
├── docs/
│   └── AI ISP 暗光拍照预览 RAW 降噪详细开发设计 V2.0.md
├── configs/
│   ├── model/mobile_nafnet_w16.yaml
│   ├── model/teacher_w32.yaml
│   ├── train/student_kd.yaml
│   ├── prune/p10.yaml
│   ├── prune/p15.yaml
│   ├── quant/qat_int8.yaml
│   └── runtime/dark_preview_profiles.json
├── ai_isp/
│   ├── data/pack_raw.py
│   ├── data/noise_model.py
│   ├── data/condition_v2.py
│   ├── models/mobile_nafnet.py
│   ├── models/nafnet_teacher.py
│   ├── losses/dark_preview_losses.py
│   ├── pruning/dependency_groups.py
│   ├── quantization/qat.py
│   └── export/static_profiles.py
├── runtime/
│   ├── include/dark_preview_denoise.h
│   ├── src/profile_selector.cpp
│   ├── src/dark_trigger.cpp
│   ├── src/raw_formatter.cpp
│   ├── src/npu_executor.cpp
│   └── src/failsafe.cpp
├── tests/
│   ├── test_pack_unpack.py
│   ├── test_condition_schema_v2.py
│   ├── test_static_profiles.py
│   ├── test_p1_padding.py
│   ├── test_onnx_equivalence.py
│   ├── test_trigger_state_machine.cpp
│   ├── test_profile_switch.cpp
│   └── test_failsafe.cpp
└── tools/
    ├── build_sensor_profile.py
    ├── count_params_macs.py
    ├── inspect_onnx_ops.py
    ├── inspect_backend_placement.py
    └── generate_release_report.py
~~~

### 16.1 CI 门禁

每次合并至少执行：

1. Python/C++ 静态检查；
2. RAW Pack/Unpack 和四 CFA；
3. ConditionSchemaV2；
4. P0/P1/P2 Shape，特别是 P1 Padding/Crop；
5. MobileNAFBlock 和 FiLM Shape；
6. P0/P10/P15 依赖组结构测试；
7. 三静态 ONNX 导出、算子白名单和数值对齐；
8. 固定 RAW Golden 回归；
9. Trigger 状态机；
10. Bypass 位精确；
11. Manifest/Model/Sensor Hash；
12. Markdown/Schema/Manifest 链接校验。

真机全量性能不要求每个提交执行，但每个 Release Candidate 必须运行三 Profile 的完整画质、性能、功耗、稳定性、Backend 落点和失败安全矩阵。

---

## 17. 发布、回滚与观测

### 17.1 Release 包

- 三个静态 OM；
- 唯一 INT8 权重语义；
- Model Manifest；
- ConditionSchemaV2；
- Sensor/Lens Profile；
- Trigger 配置；
- Runtime Library/Header；
- Model Card、Data Card；
- 质量、性能、功耗、Backend Placement 和稳定性报告；
- SHA-256、签名和许可证 Notice；
- 前一稳定版本回滚包。

### 17.2 版本

采用 MAJOR.MINOR.PATCH：

- 输入输出语义、Condition 维度或 Profile 语义变化：MAJOR；
- 兼容权重、Trigger/Sensor Profile 更新：MINOR；
- 不改变推理语义的修复：PATCH。

Model、三个 OM、Manifest、Condition Schema、Sensor Profile 和 Runtime 必须形成不可拆分的兼容集合。

### 17.3 灰度

建议 1%→10%→50%→100%。自动停止线：

- Camera Crash/OOM 高于 Baseline；
- NPU_TIMEOUT 或 INFERENCE_ERROR 超过 0.1%；
- NONFINITE_OUTPUT 任意出现；
- 未批准 Backend Fallback 任意出现；
- AI 归因掉帧率 ≥ 0.1%；
- 热 Bypass 长时间异常升高；
- Sensor/Profile 不匹配。

### 17.4 端侧观测

只记录：

- Model/Runtime/Schema 版本；
- Profile、Camera Type、Sensor/Lens Profile；
- Trigger 状态和 enhancement_strength；
- P50/P95 统计所需时延；
- 峰值内存和温度分桶；
- Status、Fallback Mask 和熔断次数；
- Backend Placement 摘要。

未经明确授权不得记录或上传 RAW、缩略图、可逆图像特征或可识别场景内容。

### 17.5 回滚

启动加载失败、Hash 不匹配或灰度停止线触发时：

1. 原子切回前一稳定兼容集合；
2. 重建 Camera Session；
3. 默认 Bypass AI，直到回滚完成；
4. 不混用新 Runtime 与旧 Schema/OM；
5. 记录回滚原因和版本；
6. 回滚演练必须在每个 RC 完成。

---

## 18. 最终架构冻结表

|项目|V2.0 冻结值|
|---|---|
|业务|Dark Photo Preview，30fps|
|触发|3 帧进入、10 帧退出、5 帧淡入淡出|
|Camera|HAL Master Physical Camera only|
|输入语义|Black-subtracted normalized Packed RAW|
|CFA 通道|R、Gr、Gb、B|
|P0|1×4×768×1024|
|P1|编译 1×4×544×960，有效 1×4×540×960|
|P2|1×4×640×960|
|Profile 发布|三个独立 Static Shape OM|
|应用层推理|Full-Tensor Single Pass|
|应用层 Tile|无|
|Student|Tiny Conditional MobileNAFNet-W16|
|通道|16、32、64、128|
|Encoder/Middle/Decoder|2/2/4、2、2/2/2|
|Condition|ConditionSchemaV2，FP32[1,24]|
|FiLM|Encoder Stage 2、Stage 3、Middle|
|输出|noise_pred|
|重建|raw_out=clamp(raw_in-noise_pred,0,1)|
|Teacher|Conditional NAFNet-W32|
|剪枝|P0/P10/P15 候选，8 通道组，真机规则自动选择|
|量化|默认全图 INT8 QAT|
|P95|P0/P1/P2 ≤30/25/27ms|
|内存|目标 ≤128MB，硬门槛 ≤160MB|
|Backend|NPU 100%，未批准 CPU/GPU 回退为 0|
|失败|原始 Packed RAW Bypass|
|动态分辨率|仅备选调研，不作为 V2.0 发布基线|

---

## 19. 参考资料与依据

1. [Simple Baselines for Image Restoration / NAFNet](https://arxiv.org/abs/2204.04676)：NAFNet、SimpleGate 和图像恢复基线。
2. [megvii-research/NAFNet 官方实现](https://github.com/megvii-research/NAFNet)：Teacher 结构与官方代码依据。
3. [Huawei ATC dynamic_image_size](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devaids/auxiliarydevtool/atlasatc_16_0053.html)：动态分辨率档位能力及其约束；本文因此将其视为备选而非麒麟 9000 量产默认。
4. [Huawei CANN Kit](https://developer.huawei.com/consumer/cn/sdk/cann-kit/)：端侧 NPU Runtime、模型转换和算子能力入口。
5. [PMRID: Practical Mobile Raw Image Denoising](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf)：移动 RAW 降噪与端侧效率参考。
6. [Real-World Mobile Image Denoising Dataset with Efficient Baselines](https://openaccess.thecvf.com/content/CVPR2024/html/Flepp_Real-World_Mobile_Image_Denoising_Dataset_with_Efficient_Baselines_CVPR_2024_paper.html)：多传感器移动端真实噪声与数据设计参考。
7. [An In-depth Performance Characterization of CPU, GPU, and NPU on Mobile Platforms](https://arxiv.org/abs/2109.12426)：移动端异构执行和必须以目标设备实测为准的依据。

公开 Ascend/Atlas 文档只能证明相关工具链存在某类能力，不能直接证明目标麒麟 9000 设备、固件和 Camera 权限具备完全相同支持。所有 NPU 结论以目标产品的转换、Backend Placement 和 Timeline 为最终依据。

---

## 20. 结论

V2.0 将项目从“全分辨率 Capture RAW 的 Tile 增强”收敛为“暗光 Preview 的固定格点、完整 Tensor、静态图残余降噪”。其核心不是简单把输入缩小，而是由 Camera Pipeline 先生成具有正确 CFA 相位和最终预览格点的 RAW Stream，然后让模型在同一 Packed RAW 格点预测噪声并相减。

最终量产链路为：

~~~text
目标设备数据与 Sensor 标定
→ Conditional NAFNet-W32 Teacher
→ MobileNAFNet-W16 Student 独立训练
→ Feature KD
→ P0/P10/P15 结构化候选
→ INT8 QAT
→ P0/P1/P2 三静态 ONNX
→ 三静态 OM
→ Full-Tensor Runtime + Dark Trigger + Failsafe
→ 麒麟 9000 三 Profile 真机验收
→ Camera DVT/灰度/回滚
~~~

发布决策只接受目标机证据：完整图 NPU 落点、端到端 P95/P99、增量内存、持续功耗、暗部画质、Camera 切换和失败安全。任何单项平均结果都不能替代三 Profile、全 Camera 和热稳态的完整准入矩阵。
