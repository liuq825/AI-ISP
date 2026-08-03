# AI ISP 量产设计基线 V18.0

> 项目名称：NAFNet-Kirin9000 AI RAW Denoise & Enhancement Engine<br>
> 文档版本：V18.0<br>
> 当前阶段：CPU 工程验证 / 云端 GPU 训练准备 / 麒麟 9000 可迁移设计<br>
> 目标场景：Capture 优先<br>
> 文档状态：设计基线

---

## 0. 文档约定

本文使用以下状态标记区分设计成熟度：

|标记|含义|
|---|---|
|**[冻结]**|当前项目必须遵守的接口、流程或设计约束|
|**[PC 可验证]**|无目标设备时可在 PC/ONNX 环境完成验证|
|**[真机后冻结]**|必须取得麒麟 9000 目标设备、固件和工具链后通过实测确定|

V18.0 不把模型参数量、理论 TOPS 或单 Tile 估算直接等同于端到端性能。只有目标设备上的模型转换、算子落点、峰值内存、功耗和时延数据可以作为量产结论。

---

## 1. 项目定位

### 1.1 建设目标

开发一套面向手机拍照链路的 AI RAW 降噪与增强模块：

> **AI RAW Denoise & Enhancement Engine**

模块在校准后的线性 RAW 域工作，目标能力包括：

- 单帧 RAW 去噪与细节恢复；
- HDR Fusion 后 RAW 的残留噪声、融合瑕疵和细节增强；
- 主摄、超广角、长焦等多摄像头 RAW 的逐摄独立增强；
- 高 ISO、低照度、暗部和高动态范围场景的稳定处理；
- FP32 训练、结构化剪枝、INT8 QAT 和 Tile 推理；
- 为后续迁移到麒麟 9000 NPU 保留标准模型接口和部署约束。

### 1.2 当前阶段边界

**[冻结]** 当前没有目标麒麟 9000 设备、自有 RAW 数据或本地 GPU，因此本阶段只承诺：

- CPU 上的数据管线和网络小样验证；
- 可复现的训练、蒸馏、剪枝、QAT 和评测设计；
- PyTorch 至 ONNX 的标准化导出路径；
- 为云端 GPU 正式训练准备配置和验收规则；
- 为未来真机转换与 profiling 准备明确的进入条件。

本阶段不承诺：

- 已达到量产画质；
- 已满足麒麟 9000 时延、内存或功耗指标；
- 所有 ONNX 算子已经在目标固件上实现 NPU 全下沉；
- 公开数据训练结果可以替代目标传感器标定和实拍数据。

### 1.3 不在首版范围内

**[冻结]** V18.0 不包含以下功能：

- HDR/MFNR 帧间配准与 Fusion；
- 多摄像头之间的配准、视差处理或跨摄融合；
- 独立部署的 AI Demosaic 模型；
- Tone Mapping、AWB、CCM、Sharpen 等完整 RGB ISP；
- 视频时域去噪和连续帧一致性优化。

训练阶段允许使用固定、可微的 Demosaic 与简化 ISP 生成 RGB 监督信号，但该渲染链路不属于首版部署模型。

---

## 2. ISP Pipeline 定位

### 2.1 推荐链路

```text
Sensor Bayer RAW
        │
        ▼
RAW Calibration
(BLC / DPC / LSC / White Level / Gain Normalize)
        │
        ├──────── Single RAW ────────────────┐
        │                                    │
        └─ HDR/MFNR Fusion → Fused RAW ──────┤
                                             ▼
                              AI RAW Denoise & Enhancement
                              (Shared Backbone + Adapter)
                                             │
                                             ▼
                                      Enhanced RAW
                                             │
                                             ▼
                                      Demosaic / RGB ISP
                                             │
                                             ▼
                                        JPEG / HEIF
```

### 2.2 模块责任

AI 模块只增强输入给它的单张 RAW：

- Single RAW 模式输入一张经过校准的传感器 RAW；
- HDR Fused RAW 模式输入上游 Fusion 已经生成的单张线性 RAW；
- Multi-camera Single RAW 模式对每个摄像头分别调用同一个共享骨干，不同时输入多路 RAW。

任何上游已经丢失或错误融合的信息都不保证可以由后增强模型恢复。HDR 鬼影、饱和和置信度信息应尽量由上游以 Conditioning 元数据形式提供。

---

## 3. RAW 输入输出接口

### 3.1 Bayer 拆分

设原始 Bayer 为：

```text
RawBayer: N × 1 × Hraw × Wraw
```

经过 2×2 Bayer 拆分后，模型图像输入固定为：

```text
PackedRAW: N × 4 × (Hraw / 2) × (Wraw / 2)
```

**[冻结]** 模型空间分辨率是原始 Bayer 宽高的一半，不得把原始 Bayer 分辨率直接写成模型输入分辨率。

例如：

|原始 Bayer|Packed RAW 模型输入|
|---|---|
|4000×3000（约 12MP）|4×2000×1500|
|8192×6144（约 50MP）|4×4096×3072|

四通道按颜色语义统一排列为：

```text
[R, Gr, Gb, B]
```

数据预处理模块根据 RGGB、GRBG、GBRG 或 BGGR 的实际 CFA，把四个空间位置映射到统一语义顺序。模型内部不得依赖某一种固定 CFA 的物理位置。

### 3.2 RAW 归一化

每个颜色通道分别减去 Black Level，并使用统一 White Level 归一化：

```text
x[c] = clip((raw[c] - black_level[c]) /
            (white_level - black_level[c]), 0, 1)
```

推理输出执行对应逆变换：

```text
raw_out[c] = round(
    clip(y[c], 0, 1) × (white_level - black_level[c])
    + black_level[c]
)
```

**[冻结]** Black Level、White Level、位深和 CFA 必须与 RAW 一起进入预处理接口，禁止在网络代码中硬编码具体传感器数值。

### 3.3 模型输入

模型公开输入由两部分组成：

```text
raw:       FP32/INT8, N × 4 × Hp × Wp
condition: FP32,      N × Ccond
```

其中：

```text
Hp = Hraw / 2
Wp = Wraw / 2
```

主图像输入严格保持四通道，元数据不得通过额外图像通道改变 `raw` 的定义。

Conditioning 使用版本化记录结构，至少包含：

|类别|字段|
|---|---|
|曝光|Exposure Time、Analog Gain、Digital Gain|
|噪声|Shot Noise 系数、Read Noise 系数或等价标定参数|
|RAW 标定|Black Level 摘要、White Level、有效位深|
|场景|Single RAW、HDR Fused RAW、Multi-camera Single RAW|
|摄像头|Sensor Profile、Main/Ultra-wide/Tele/Other Camera Role|
|HDR 质量|Fusion Confidence、Motion Score、Ghost Score；无数据时使用中性默认值并设置有效标志|

连续物理量执行对数或区间归一化；场景、摄像头角色和 Sensor Profile 使用离散编码或嵌入，不再用 `0/0.5/1` 表示彼此无顺序关系的类别。

`Ccond` 在正式导出模型时固定，并由 `condition_schema_version` 管理。目标传感器清单未确定前，PC PoC 使用默认 Sensor Profile。

### 3.4 模型输出

模型输出：

```text
residual: N × 4 × Hp × Wp
```

增强结果：

```text
EnhancedRAW = InputRAW + PredictedResidual
```

最终输出保持线性 RAW 语义、原始 CFA 和原始位深。网络内部不执行 Gamma、Tone Mapping 或显示域锐化。

---

## 4. 场景统一与多摄设计

### 4.1 共享骨干

Single RAW、HDR Fused RAW 和多摄单帧 RAW 共享同一套主干参数，以降低重复训练和版本维护成本。

```text
Packed RAW ───────────────► Shared NAFNet Backbone ─► Residual RAW
                                ▲
Condition Record ─► Encoder ─► FiLM / Adapter
```

### 4.2 条件适配器

Condition Encoder 将连续元数据与离散场景/传感器信息编码成 FiLM 参数：

```text
FiLM(x) = gamma(condition) × x + beta(condition)
```

**[冻结]** FiLM 注入位置为：

- Encoder Stage 0；
- Encoder Stage 3；
- Middle Blocks。

Stage 0 用于控制早期噪声与低层统计，Stage 3 和 Middle 用于处理场景域差异和高层恢复强度。

### 4.3 多摄边界

多摄模式的调用方式为：

```text
Main RAW  ─► Shared Model + Main Sensor Adapter
UW RAW    ─► Shared Model + UW Sensor Adapter
Tele RAW  ─► Shared Model + Tele Sensor Adapter
```

各摄像头单独执行 Bayer 拆分、归一化和增强，不进行跨摄特征交换。传感器差异通过噪声标定、Sensor Profile 和轻量适配器处理。

---

## 5. 模型架构

### 5.1 Teacher

Teacher 使用 NAFNet-W64，主要目标是建立云端 FP32 画质上限，不作为端侧部署模型。

```yaml
width: 64
encoder_blocks: [2, 2, 6, 8]
middle_blocks: 4
decoder_blocks: [2, 2, 2, 2]
input_channels: 4
output_channels: 4
```

按标准 NAFBlock 估算，未计入 Conditioning 分支时参数量约为 57.93M。

### 5.2 Full Student

未剪枝 Student 使用 NAFNet-W32：

```yaml
width: 32
encoder_blocks: [2, 2, 6, 8]
middle_blocks: 4
decoder_blocks: [2, 2, 2, 2]
input_channels: 4
output_channels: 4
```

按标准 NAFBlock 估算，未计入 Conditioning 分支时参数量约为 14.58M。

**[冻结]** V18.0 不设置 8M 参数上限。部署模型必须经过结构化剪枝，但剪枝强度由 MACs 降幅和画质门槛共同决定。

参数估算依据：[megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)

### 5.3 Student 的部署约束

Student 的计算图必须：

- 可以导出静态 ONNX；
- 不包含 Python 控制流或输入相关动态分支；
- 不使用自定义运行时算子；
- 使用固定输入 Tile Shape 导出部署版本；
- 对 LayerNorm、Depthwise Conv、ReduceMean/SCA、SimpleGate、DepthToSpace、FiLM、Add 和 Mul 建立独立算子导出测试；
- 导出后使用 ONNX Runtime 对比 PyTorch 数值结果。

Teacher 可以保留训练友好的标准实现，Student 必须把部署兼容性作为结构约束。

---

## 6. 训练总路线

**[冻结]** 完整训练顺序如下：

```text
数据与噪声模型准备
        │
        ▼
W64 Teacher FP32 训练
        │
        ├───────────────┐
        ▼               │
W32 Full 独立训练       │
        │               │
        ▼               │
W64 → W32 蒸馏 ◄────────┘
        │
        ▼
结构化剪枝 P10/P20/P30
        │
        ▼
双 Teacher 恢复训练
        │
        ▼
PTQ 可行性基线
        │
        ▼
INT8 QAT
        │
        ▼
Pruned Q/DQ INT8 ONNX
```

剪枝和剪枝恢复是必要阶段，不得因为未设置参数上限而跳过。

### 6.1 Teacher 训练

Teacher 先在 Single RAW 数据上建立稳定的 RAW 恢复能力，再逐步加入 HDR Fused RAW 和多传感器适配数据。

核心监督包括：

- 噪声加权 RAW 重建；
- RAW 梯度与边缘约束；
- 固定 Demosaic/ISP 渲染后的 RGB 重建；
- 低权重感知约束；
- Tile 一致性约束。

感知损失不得成为主导项，避免在 RAW 域生成无法由传感器数据支持的纹理。

### 6.2 Full Student 独立训练

W32 Full 必须先独立训练，再进入蒸馏阶段。这样可以避免 Student 在训练早期完全受 Teacher 局部偏差限制，也能建立独立可比较基线。

### 6.3 Teacher-to-Student 蒸馏

蒸馏包含：

1. Output KD：W32 输出逼近 W64 输出；
2. Feature KD：约束 Encoder Stage 0、Encoder Stage 3 和 Middle；
3. Attention KD：约束通道或空间注意力统计；
4. Ground Truth Loss：始终保留真实目标监督，防止 Student 只复制 Teacher 错误。

---

## 7. 必要结构化剪枝

### 7.1 剪枝定位

**[冻结]** 剪枝位于 W32 蒸馏之后、QAT 之前：

```text
W64 Teacher
→ W32 Full
→ Distilled W32
→ Structured Pruning
→ Dual-Teacher Recovery
→ INT8 QAT
```

只允许使用能够真实删除特征通道、权重和 MACs 的结构化剪枝。禁止使用仅将权重置零、但部署图仍保留原计算形状的非结构化稀疏化作为最终方案。

### 7.2 剪枝评价基准

剪枝 MACs 统一在以下输入上计算：

```text
Batch: 1
Input: 4 × 1024 × 1024 Packed RAW
Condition: 1 × Ccond
Precision for MAC accounting: independent of FP32/INT8
```

参数量、模型文件大小和理论 MACs 均记录，但候选分档以实际删除后的 MACs 降幅为准。

### 7.3 三档剪枝候选

|候选|相对 Distilled W32 的目标 MACs 降幅|定位|
|---|---:|---|
|P10|约 10%|保守量产候选，最终模型最低要求|
|P20|约 20%|默认部署候选|
|P30|约 30%|高压缩候选|

候选之间不是简单继续训练同一权重。每个候选必须保存独立结构配置、检查点、ONNX 和评测报告。

### 7.4 通道重要度

使用验证/校准集计算一阶 Taylor Saliency：

```text
importance(channel) = mean(|weight × gradient|)
```

同时记录激活幅度和激活稀疏度，用于处理仅依靠权重梯度可能产生的误判。最终排序使用归一化后的 Taylor 分数为主、激活统计为辅。

重要度数据至少覆盖：

- 不同 ISO/增益；
- 明亮、低照、极暗区域；
- 平坦区与高频纹理；
- Single RAW 与 HDR Fused RAW；
- 不同 Camera Role。

### 7.5 剪枝分组约束

**[冻结]** 剪枝以 8 个通道为最小粒度，所有保留的隐藏通道数必须为 8 的倍数。

成组约束包括：

- Encoder 输出与对应 Decoder Skip 输入同步；
- 残差分支两侧通道索引同步；
- PixelShuffle 前卷积按照目标输出通道同步重构；
- 所有依赖特征通道的 Conv、Depthwise Conv、LayerNorm、FiLM、beta 和 gamma 同步切片；
- RAW 的四个输入/输出语义通道不得删除；
- Conditioning 的物理语义维度不得删除或重排。

不能把 LayerNorm、beta、gamma 简单标记为“不剪”。如果其关联特征通道被删除，相应参数必须按照同一通道索引物理裁剪。

### 7.6 渐进式剪枝

剪枝采用渐进流程：

```text
重要度统计
→ 删除一组低重要度通道
→ 短周期恢复训练
→ 重新统计重要度
→ 下一剪枝步
```

每一步只向目标 MACs 降幅推进约 5 个百分点：

- P10：约 2 个剪枝/恢复循环；
- P20：约 4 个剪枝/恢复循环；
- P30：约 6 个剪枝/恢复循环。

实际循环数允许因 8 通道对齐产生一档偏差，但不得一次性删除全部目标通道。

### 7.7 双 Teacher 恢复

剪枝恢复同时使用：

- W64 Teacher：保持画质上限；
- 未剪枝、已完成蒸馏的 W32 Full：保持结构行为、场景条件和局部响应。

固定恢复 Loss：

```text
Lrecover =
    1.00 × Lground_truth
  + 0.20 × Loutput_KD_W64
  + 0.15 × Loutput_KD_W32
  + 0.10 × Lfeature_KD
  + 0.05 × Ltile
```

所有分项进入加权前必须按各自有效元素数量归一化，避免不同 Patch 或 Tile 尺寸改变相对权重。

### 7.8 最终候选选择

选择规则固定为：

1. 分别完成 P10、P20、P30 的恢复训练与完整评测；
2. 剔除超过画质退化门槛的候选；
3. 在合格候选中选择 MACs 降幅最大的模型；
4. P20 为默认目标；P20 不合格时采用 P10；
5. P10 仍不合格时，调整剪枝分组、重要度校准集或恢复训练并重新执行，不能跳过剪枝；
6. 最终部署模型必须至少达到 P10 级结构化 MACs 降幅。

每个剪枝模型必须重新生成物理结构和独立 ONNX，禁止依赖运行时 Mask 模拟剪枝。

---

## 8. INT8 PTQ 与 QAT

### 8.1 顺序

**[冻结]** QAT 只能在最终剪枝结构确定之后进行：

```text
Pruned FP32
→ PTQ Baseline
→ Sensitivity Analysis
→ Fake Quant / QAT
→ Q/DQ INT8 ONNX
```

不得先对未剪枝 W32 完成 QAT，再通过通道删除破坏量化统计。

### 8.2 量化策略

- Weight：INT8 Per-channel；
- Activation：INT8，具体 Per-tensor/Per-channel 以目标工具链支持情况为准；
- 输入/输出量化尺度显式记录；
- 校准集覆盖全部场景、ISO、摄像头角色和亮暗分布；
- 对 LayerNorm、乘法、残差 Add、FiLM 和输出层单独执行敏感度检查；
- 必要时允许少量敏感层保留更高精度，但必须在真机报告中披露实际落点。

PTQ 只用于建立可行性和敏感层基线，正式部署候选必须完成 QAT。

---

## 9. 数据设计

### 9.1 当前数据路线

在没有自有 RAW 数据时，PC PoC 使用：

- 公开配对 RAW 数据；
- 干净 RAW 或高质量 RAW 上的物理噪声合成；
- 合成 HDR Fusion 残留；
- 虚拟 Sensor Profile 和 Camera Role 验证 Conditioning 接口。

公开数据只用于验证方法、训练稳定性和工程链路，不作为目标传感器量产结论。

### 9.2 物理噪声合成

噪声模型至少覆盖：

- Shot Noise；
- Read Noise；
- ADC/量化噪声；
- 行噪声和列噪声；
- Black Level 漂移；
- 热点、坏点和随机脉冲；
- 模拟增益与数字增益误差；
- 近饱和区域的裁剪与非理想响应。

基础异方差模型：

```text
variance(x) = a × x + b
```

其中 `a` 和 `b` 来自 Sensor Profile 或合成采样范围，并作为 Conditioning 的噪声参数输入。

### 9.3 HDR Fusion 残留模拟

HDR 训练数据额外加入：

- 局部饱和；
- 运动边界；
- 鬼影残留；
- Fusion Confidence 降低；
- 局部亮度或增益不一致；
- 纹理丢失和融合噪声非均匀性。

AI 模块只学习增强融合结果，不承担重新配准或重新融合职责。

### 9.4 目标设备数据闭环

取得目标设备后必须采集：

- 不同 ISO/曝光的暗场和灰阶平场；
- 各摄像头 Black Level、White Level 和噪声参数；
- 静态 Burst 平均得到的 Clean Reference；
- 低照、高动态、纹理、人像、运动和高光场景；
- HDR Fusion 中间 RAW 及其 Confidence/Motion/Ghost 元数据；
- 主摄、超广角、长焦分别覆盖的训练与盲测数据。

数据集按原始拍摄场景或序列划分。禁止同一源图像的不同裁剪块进入不同集合。

---

## 10. Loss 体系

### 10.1 Ground Truth Loss

Ground Truth Loss 由以下部分组成：

```text
Lground_truth =
    λraw  × Lraw_reconstruction
  + λgrad × Lraw_gradient
  + λrgb  × Lrendered_rgb
  + λperc × Lperceptual
```

约束：

- `Lraw_reconstruction` 为主损失；
- RAW 重建按照噪声方差进行加权；
- `Lrendered_rgb` 使用固定 Demosaic、白平衡、CCM 和简单 Tone Curve；
- `Lperceptual` 保持低权重，只辅助用户可见纹理，不允许主导训练；
- RGB 渲染参数必须随数据样本保存，不能对所有传感器硬编码一套 CCM。

### 10.2 Tile Consistency Loss

训练时对同一图像随机产生两个重叠 Tile：

```text
Ltile = mean(|OutputA_overlap - OutputB_overlap|)
```

同时在可完整推理的小尺寸样本上约束：

```text
Lfull_tile = mean(|FullFrameOutput - TiledOutput|)
```

Tile Loss 只计算两个输出都处于有效感受野的区域，填充区和无效边缘不参与监督。

---

## 11. Tile Scheduler

### 11.1 坐标定义

所有 Tile 尺寸和重叠量均基于 Packed RAW 空间，而不是原始 Bayer 空间。

50MP 示例：

```text
Original Bayer: 8192 × 6144
Packed RAW:     4 × 4096 × 3072
Tile:           1024 × 1024
Overlap:        64
Stride:         960
```

Tile 数量：

```text
Nw = ceil((4096 - 1024) / 960) + 1 = 5
Nh = ceil((3072 - 1024) / 960) + 1 = 4
Total = 5 × 4 = 20 Tiles
```

V18.0 使用 20 Tile 作为该配置的正确估算，不沿用 12 Tile 结论。

### 11.2 调度流程

```text
Packed RAW
→ Reflect Padding
→ Tile Coordinate Generation
→ Fixed-shape Model Inference
→ Valid Region Crop
→ Hann Weighted Overlap-Add
→ Weight Normalization
→ Enhanced Packed RAW
```

边缘 Tile 优先移动到图像有效边界内；无法满足固定 Shape 时才使用反射填充。输出合成使用可分离 Hann/raised-cosine 权重，并保证所有有效像素的累计权重大于零。

### 11.3 Tile 验收

Tile 验证必须覆盖：

- 纯色和平滑渐变；
- 斜线、文字和重复纹理；
- 极暗区域；
- 高光饱和边界；
- 图像四边与四角；
- Tile 交界处穿过主体边缘的情况。

不得只用随机自然图像的全局 PSNR 判断是否存在拼接缝。

---

## 12. PC 阶段工程输出

### 12.1 CPU 可执行验证

当前 CPU 阶段应完成：

- Bayer 四种 CFA 的拆分与无损重组；
- Black/White Level 归一化与逆变换；
- 合成噪声模型的统计验证；
- W16/W32 小输入 Forward/Backward Smoke Test；
- Conditioning 和 FiLM Shape Test；
- Taylor Saliency 统计与结构化通道删除；
- 剪枝后模型的 Shape、Skip 和 PixelShuffle 一致性；
- Tile 切分、合成和全帧对比；
- PyTorch/ONNX Runtime 数值一致性；
- Q/DQ 图的导出 Smoke Test。

CPU 阶段不训练正式 W64/W32 权重。

### 12.2 模型制品

云端 GPU 训练阶段应输出：

|制品|用途|
|---|---|
|W64 FP32 Teacher|画质上限与蒸馏|
|W32 FP32 Full|未剪枝 Student 基线|
|W32 Distilled|剪枝输入基线|
|P10/P20/P30 FP32|剪枝候选|
|Final Pruned FP32|QAT 输入和 FP32 对照|
|Final Pruned Q/DQ INT8 ONNX|端侧转换输入|

每个制品必须同时保存：

- 网络结构配置；
- Condition Schema 版本；
- 训练数据版本；
- Git/代码版本；
- 量化与剪枝配置；
- 参数量、MACs、模型大小和评测报告。

---

## 13. 麒麟 9000 迁移路线

### 13.1 PC 中间格式

PC 阶段以静态 ONNX 作为部署中间格式：

```text
PyTorch Pruned QAT
→ Static Q/DQ ONNX
→ ONNX Checker
→ ONNX Runtime Parity
→ Target Converter
```

### 13.2 目标运行时选择

获得目标设备、系统版本和厂商工具链后，只维护目标固件实际支持的一条路径：

- 若提供兼容的 HiAI/CANN DDK 和 `.om` 模型路径，则使用对应转换与推理接口；
- 若目标系统要求 MindSpore Lite/NNRT，则转换为 `.ms` 并启用 NPU Delegate；
- 不在缺少设备和固件信息时预先承诺具体 DDK 版本。

参考：

- [HUAWEI CANN Kit](https://developer.huawei.com/consumer/cn/sdk/cann-kit/)
- [HUAWEI MindSpore Lite Kit](https://developer.huawei.com/consumer/cn/sdk/mindspore-lite-kit)

### 13.3 真机准入条件

进入真机阶段前必须具备：

- 明确的麒麟 9000 设备型号和系统版本；
- 可安装和调试的固件权限；
- 对应 DDK/SDK、模型转换器和算子说明；
- 至少一个真实 Sensor Profile；
- 一套固定的画质、性能和稳定性测试集；
- Pruned FP32 与 Pruned Q/DQ INT8 ONNX。

---

## 14. 测试与验收标准

### 14.1 接口正确性

- 支持 RGGB、GRBG、GBRG、BGGR；
- Bayer 拆分再重组必须逐像素一致；
- 支持不同 Black Level、White Level 和位深；
- 明确定义奇数宽高的拒绝或预裁剪行为；
- 饱和、全黑、坏点和 NaN/Inf 输入不得导致异常扩散；
- Conditioning 缺失字段使用版本化默认值；
- 未知 Sensor Profile 使用默认适配器并产生可追踪告警。

### 14.2 画质报告

分别对 Single RAW、HDR Fused RAW 和各 Camera Role 报告：

- RAW PSNR、SSIM；
- 渲染 RGB PSNR、SSIM；
- 噪声功率谱和残留固定模式噪声；
- 平坦区噪声方差；
- 纹理和边缘保持；
- 色差和亮度偏移；
- 鬼影、过平滑、锐化光晕、色块和伪纹理。

对比组至少包括：

- No Processing；
- 传统或简单 RAW NR 基线；
- W64 Teacher；
- W32 Full；
- W32 Distilled；
- P10/P20/P30；
- Final Pruned FP32；
- Final Pruned INT8；
- Full-frame 与 Tiled Output。

### 14.3 剪枝门槛

每个候选必须记录：

- 实际参数量；
- 1024×1024 Packed Tile 的 MACs；
- ONNX 文件大小；
- CPU/云端 GPU 参考时延；
- 各场景画质指标。

相对未剪枝、已蒸馏的 W32：

```text
RAW PSNR 下降 ≤ 0.10 dB
SSIM 下降 ≤ 0.002
```

同时必须通过人工伪影审查。最终部署模型至少达到 P10 级结构化 MACs 降幅。

### 14.4 量化门槛

相对 Final Pruned FP32：

```text
RAW PSNR 下降 ≤ 0.10 dB
SSIM 下降 ≤ 0.002
```

量化报告还必须列出混合精度层、CPU 回退风险和异常离群样本。

### 14.5 Tile 门槛

- Tile 边界误差不得形成可见亮度、颜色或纹理断层；
- 边界带误差不得显著高于相同图像非边界区域；
- 小图上的 Tiled Output 必须与 Full-frame Output 保持数值一致性；
- 所有 Tile 配置必须记录 Tile Size、Overlap、Padding 和 Blend Window。

### 14.6 真机门槛

**[真机后冻结]** 必须统计：

- P50/P95 端到端时延；
- 单 Tile 与整帧时延；
- 峰值内存；
- NPU 驻留率和 CPU/GPU 回退；
- 模型加载时间；
- 功耗和温升；
- 连续拍照稳定性；
- 失败回退和旁路行为。

暂定 12MP Capture 端到端目标不高于 300 ms。50MP 指标在真机 profiling 后冻结，不沿用 V17.0 的 40–60 ms 估算。

---

## 15. 项目阶段与退出条件

### Phase A：CPU 工程基线

完成条件：

- RAW 拆分、归一化、噪声合成和 Tile 单元测试通过；
- NAFNet 小样和 Conditioning 可运行；
- 剪枝能够物理改变网络结构和 MACs；
- ONNX 导出和 Runtime 对齐通过。

### Phase B：云端 GPU FP32 训练

完成条件：

- W64 Teacher 和 W32 Full 收敛；
- 三类场景均有独立评测；
- W32 蒸馏相对独立训练取得稳定收益；
- 训练和评测可复现。

### Phase C：剪枝与 QAT

完成条件：

- P10/P20/P30 均完成恢复训练；
- 选择满足画质门槛的最大 MACs 降幅候选；
- 最终模型至少达到 P10；
- QAT 模型满足 INT8 画质门槛；
- 输出静态 Q/DQ ONNX。

### Phase D：目标传感器闭环

完成条件：

- 完成目标传感器噪声标定；
- 采集各摄像头和 HDR Fusion 数据；
- 重新训练/微调 Sensor Adapter；
- 完成真实场景盲测。

### Phase E：麒麟 9000 真机量产验证

完成条件：

- 模型成功转换；
- 不存在未批准的 CPU 回退；
- 画质、时延、内存、功耗和稳定性全部达标；
- 建立异常检测、旁路和传统 ISP 回退机制；
- 固化模型、工具链、固件和 Sensor Profile 版本。

---

## 16. 风险与控制

|风险|影响|控制措施|
|---|---|---|
|公开 RAW 与目标传感器域差异|真机画质下降|目标传感器重新标定、采集和微调|
|NAFNet 算子无法全量下沉 NPU|时延和功耗失控|尽早执行单块 ONNX/转换验证，保留部署友好替代实现|
|剪枝破坏 Skip/PixelShuffle 结构|模型无法运行或画质突降|成组通道约束、8 通道对齐、渐进剪枝和双 Teacher 恢复|
|QAT 对乘法、归一化敏感|INT8 精度下降|逐层敏感度分析和有限混合精度|
|Tile 出现拼接缝|高分辨率照片可见伪影|Tile Consistency、有效区裁剪和 Hann 融合|
|完全统一模型发生场景负迁移|某些摄像头或 HDR 退化|共享骨干配合场景/传感器适配器并分场景验收|
|无目标设备时过早冻结性能|量产承诺失真|性能指标标记为真机后冻结，只报告可复现测量|

---

## 17. 最终冻结配置

|项目|V18.0 基线|
|---|---|
|任务|RAW 单帧降噪与增强|
|场景|Single RAW / HDR Fused RAW / Multi-camera Single RAW|
|多摄方式|逐摄独立增强，不做跨摄融合|
|图像输入|Bayer 拆分后 4 通道 RAW|
|空间尺寸|`Hraw/2 × Wraw/2`|
|输出|同尺寸 4 通道增强 RAW|
|Teacher|NAFNet-W64|
|Full Student|NAFNet-W32|
|Encoder|`[2,2,6,8]`|
|Middle|`4`|
|Decoder|`[2,2,2,2]`|
|条件方式|版本化 Metadata + FiLM/Adapter|
|FiLM 位置|Encoder 0 / Encoder 3 / Middle|
|蒸馏|Output + Feature + Attention|
|剪枝|必要流程，P10/P20/P30 结构化候选|
|剪枝粒度|8 通道对齐|
|剪枝选择|画质门槛内最大 MACs 降幅，最低 P10|
|恢复|W64 + W32 双 Teacher|
|量化|剪枝后 INT8 QAT|
|高分辨率|Packed RAW Tile 推理|
|Tile 融合|有效区裁剪 + Hann Weighted Overlap-Add|
|PC 中间格式|静态 Q/DQ ONNX|
|目标平台|麒麟 9000，运行时真机后确定|
|参数约束|无 8M 硬上限|
|性能结论|仅真机测量后冻结|

---

## 18. 结论

V18.0 将项目从“带有固定参数量和推测时延的网络方案”调整为一套可验证的工程基线：

```text
Calibrated Bayer RAW
→ 4-channel Packed RAW at Half Width/Height
→ Conditional Shared NAFNet
→ Distillation
→ Mandatory Structured Pruning
→ Dual-Teacher Recovery
→ INT8 QAT
→ Seam-safe Tile Inference
→ Kirin 9000 Runtime Validation
```

模型参数量不再被 8M 指标直接限制，但结构化剪枝被明确为不可跳过的量产流程。最终模型以画质门槛内的真实 MACs 降幅、标准模型可转换性和麒麟 9000 真机端到端表现作为验收依据。
