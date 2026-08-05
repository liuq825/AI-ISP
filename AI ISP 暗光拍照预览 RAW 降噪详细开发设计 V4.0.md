# AI ISP 暗光拍照预览 RAW 降噪详细开发设计 V4.0

> 文档版本：4.0.0
> 更新日期：2026-08-05
> 目标平台：麒麟 9000 手机平台
> 部署模型：Conditional MobileNAFNet Dark Preview V4
> 发布形态：RYYB 主摄/长焦共享一个固定 Shape 的 W8A8 OM
> 当前状态：算法工程链已在 CPU 小样本跑通；真实 RYYB、商用 DDK、OM 和麒麟 9000 验收未完成，`release_ready=false`

---

## 0. 执行摘要

V4.0 将项目从“三 Camera、三 Profile、三静态 OM”的宽泛方案收敛为一条可验证的算法与模型压缩主线：

```text
真实 RYYB 数据建设
→ W32 Teacher
→ W16 Student
→ Feature KD
→ Torch-Pruning P10/P15
→ 剪枝恢复
→ 对称 LSQ / 非对称 LSQ+ W8A8 QAT
→ 单一固定 Shape Q/DQ ONNX
→ 单一 OM
→ 麒麟 9000 画质、8ms P95 与整条 4K30 验收
```

冻结结论：

|项目|V4.0 决策|
|---|---|
|AI Camera|RYYB 广角主摄、RYYB 潜望式长焦|
|超广角|RGGB 超广角完全解耦，固定走传统 ISP NR|
|RAW 输入|`packed_raw[1,4,768,1024]`，对应 `2048×1536`、4:3 RYYB|
|通道语义|`[R,Yr,Yb,B]`|
|Condition|ConditionSchemaV2 `float32[1,24]`，仅允许 main/tele 映射|
|模型输出|同格点 `noise_pred`|
|发布制品|最终发布目录只有一个 `dark_preview_ryyb_4x3_int8.om`|
|剪枝|真实结构化 P10/P15，不使用非结构化稀疏冒充端侧加速|
|量化|完整实现 LSQ 和 LSQ+；是否启用非零 Offset 由目标微基准决定|
|AI 节点预算|P50≤6ms、P95≤8ms、P99≤9ms、10ms硬超时|

V4.0 的工作重点是数据、模型、训练、蒸馏、剪枝、QAT 和验收。Camera HAL、传统 ISP、Trigger 和 Runtime 只规定输入输出契约、准入条件与失败安全，不在本文展开外围实现细节。

---

## 1. 范围和职责边界

### 1.1 模块负责

- 归一化 RYYB Packed RAW 的单帧噪声预测；
- 通过 Camera、Sensor、噪声和曝光条件适配主摄与长焦；
- 暗部随机噪声、行列噪声、色噪及纹理保真；
- Teacher、Student、蒸馏、剪枝、恢复、LSQ/LSQ+ QAT；
- 固定 Shape ONNX、Q/DQ、Quant Policy、拓扑与发布 Hash；
- 输出算法画质、压缩率、数值和目标端性能验收报告。

### 1.2 模块不负责

- RGGB 超广角降噪；
- Demosaic、HDR Merge、Tone Mapping、锐化、JPEG/HEIF 编码；
- 原生 4K RAW 全格点推理；
- 用 CPU/ONNX Runtime 作为量产实时回退；
- 在输入相位错误时自动交换通道或猜测 CFA；
- 在缺少真实 RYYB 数据和麒麟 9000 证据时宣称量产完成。

### 1.3 4K30 解释

4K30 是整条拍照预览输出 Pipeline 的产品要求。AI 节点处理上游专门协商的 `2048×1536`、4:3 RYYB 支路，输出回到后续 ISP，最终显示链完成 4K 构图和输出。

禁止把固定 AI Tensor 偷换为原生 `4096×3072` RAW。后者会把 Dense W16 的计算量从约 35.58 GMAC 放大至约 142.31 GMAC，P10/P15 剪枝不足以抵消约四倍面积增长。

---

## 2. 固定输入输出契约

### 2.1 模型接口

|名称|类型|Shape|语义|
|---|---|---|---|
|`packed_raw`|训练 FP32；OM 边界由 DDK 冻结为 FP16|`1×4×768×1024`|归一化 `[R,Yr,Yb,B]`|
|`condition`|FP32|`1×24`|ConditionSchemaV2|
|`noise_pred`|QAT仿真/OM输出|`1×4×768×1024`|与输入同符号、同格点的噪声|

唯一输出公式：

\[
raw_{out}=\operatorname{clamp}(raw_{in}-noise_{pred},0,1)
\]

`enhancement_strength` 位于 Condition 第 23 维，并在模型图内只缩放一次 `noise_pred`。强度为 0 时输出降噪量必须为 0；Runtime 不得二次缩放。

### 2.2 RYYB 通道语义

- `R`：红色像素；
- `Yr`：与 R 同行的黄色像素；
- `Yb`：与 B 同行的黄色像素；
- `B`：蓝色像素。

支持四种 2×2 旋转相位：`ryyb/byyr/yryb/ybyr`。每颗 Sensor 必须由版本化 Profile 明确声明物理相位，Formatter 统一打包为 `[R,Yr,Yb,B]`。

四通道 Pack 保持 Mosaic 样本总数不变，不宣称天然减少 DMA 字节。其价值是固定颜色语义、保持 2×2 邻域和把二维 Mosaic 转为适合卷积的四平面。

### 2.3 2×2 相位死规则

HAL 的 Crop `x/y/width/height` 必须全部为非负偶数，只能以 2×2 Bayer 宏像素为步长。奇数行或奇数列偏移会改变四通道语义，必须在调用 AI 前拒绝。

禁止：

- 奇数 Crop 后运行时交换通道；
- 普通 Resize 到固定 Shape；
- 通过补一行或丢一列掩盖相位错误；
- 根据 Camera 名称猜测 CFA。

### 2.4 准入条件

以下全部满足才允许调用 AI：

1. Camera 为 `main` 或 `tele`；
2. Sensor Profile 在发布清单注册；
3. CFA 是该 Sensor 注册的 RYYB 相位；
4. RAW/Crop 精确为 `2048×1536`；
5. Crop 起点和宽高均为偶数；
6. RAW10/12/14/16，uint16 容器 Stride 合法；
7. 四通道 White Level 分别大于 Black Level；
8. Camera/Sensor/Lens Condition 组合合法；
9. Model、Topology、Condition、Sensor 和 Quant Policy Hash 匹配。

RGGB 超广角、过渡帧、热保护、超时、Hash 错误或 NPU 不可用时，直接交回传统 ISP NR。禁止静默执行 CPU AI。

---

## 3. 训练数据建设

### 3.1 数据清单

真实 RYYB 数据以 JSONL Manifest 管理，最少包含：

- Sample、Scene、Burst、Split；
- Camera、Sensor Profile、CFA 相位；
- Noisy/Clean 路径；
- 位深、Black/White Level；
- Exposure、ISO、模拟/数字增益；
- Sensor 温度、Lux/EV、场景亮度；
- Shot/Read Noise、Noise Level；
- 数据来源与 `smoke_only` 标记。

规范训练文件使用二维 Mosaic NPY。供应商 DNG/RAW/BIN 必须先由独立转换工具完成 Endian、Stride、位深和 Metadata 校验，再进入训练集。

### 3.2 数据量硬门槛

每颗物理相机分别满足：

|集合|最低独立场景数|
|---|---:|
|Train|3000|
|Validation|300|
|Blind Test|500|

同一 Scene、Burst 或连续拍摄片段只能属于一个集合。主摄与长焦分别报告，不允许用主摄数量填补长焦缺口。

### 3.3 采样与 Patch

- Packed Patch：前期 256、中期 384、后期在显存允许时 512；
- Crop 起点始终乘 2，保持 RYYB 相位；
- main/tele 等量采样；
- 按 ISO、Lux、曝光、温度和场景类型分层；
- 暗部样本为主，但必须保留灯牌、高光、天空渐变、肤色、文字、织物和树叶；
- 旋转/镜像在语义 Packed 域同步作用于 Noisy/Clean，不交换语义通道。

### 3.4 Clean Reference

优先级：

1. 同场景低 ISO/长曝光且运动可控的配对参考；
2. 多帧鲁棒对齐与融合生成的参考；
3. 经人工和统计审查的高质量传统 Pipeline 参考。

Reference 生成版本、对齐残差、运动 Mask、饱和 Mask 和坏点处理必须进入数据卡。存在配准重影的区域不得作为强监督真值。

### 3.5 SIDD 边界

当前仓库只有 RGGB/BGGR/GRBG/GBRG 的 SIDD。它只用于：

- Dataset、梯度、Checkpoint 和阶段运行器冒烟；
- 剪枝依赖图与拓扑重建；
- LSQ/LSQ+ 校准和 Q/DQ 导出工程验证。

所有 SIDD Batch 返回 `smoke_only=true`。不得用 SIDD PSNR、SSIM 或 CPU QAT 结果选择 RYYB 量产模型。

---

## 4. Teacher 与 Student

### 4.1 Teacher

Conditional NAFNet-W32：

- 四级 Encoder，宽度 `[32,64,128,256,512]`；
- Encoder Block `[2,2,4,8]`，Middle 4，Decoder `[2,2,2,2]`；
- 仅用于 FP32 训练和 KD，不进入发布图；
- 输出 `noise_pred`；
- 向 KD 暴露 Encoder Stage 3 与 Middle 特征。

### 4.2 Student

Dense MobileNAFNet-W16：

- Feature Width `[16,32,64,128]`；
- Encoder `[2,2,4]`，Middle 2，Decoder `[2,2,2]`；
- Dense 参数量 660,836；
- 固定 Shape 计算量约 35.58 GMAC；
- 仅使用 Conv、DWConv、Add、Mul、Tanh、Nearest Resize、Linear 和常量 Slice 等可审计算子。

每个 MobileNAFBlock-DW：

```text
x
→ 1×1 Expand(2C)
→ 3×3 DWConv
→ StaticSimpleGate(C/C)
→ 1×1 Project(C)
→ Residual × beta
→ 1×1 Expand(2C)
→ StaticSimpleGate(C/C)
→ 1×1 Project(C)
→ Residual × gamma
```

### 4.3 StaticSimpleGate

当前发布实现使用构造期常量 `torch.narrow`，不使用 `torch.chunk`。导出器不执行全局 Monkey Patch。

只有导入外部遗留模型并明确检测到动态 `SimpleGate` 时，才允许在模型副本上执行显式模块替换；训练模型对象不得被全局修改。

### 4.4 RYYB Reference ISP

Tone/Gradient Loss 使用按 Camera 选择的可微 3×4 RYYB 光谱解混矩阵、CCM 和 log1p Tone。仓库中的矩阵仅用于工程验链，真实主摄/长焦标定到位前必须保持 `release_ready=false`。

---

## 5. 训练与蒸馏

### 5.1 监督 Loss

\[
L_{sup}=0.55L_{RAW}+0.30L_{Tone}+0.15L_{Gradient}
\]

- RAW：暗部加权 Charbonnier，并屏蔽饱和目标；
- Tone：版本化 RYYB Reference ISP 后 RGB/Luma L1；
- Gradient：Reference ISP Luma 的 Sobel X/Y。

### 5.2 Feature KD

\[
L_{KD}=0.50L_{RAW}+0.25L_{Tone}+0.15L_{Gradient}+0.10L_{Feature}
\]

- 对齐 Encoder Stage 3 和 Middle；
- Student 使用训练期 1×1 Adapter；
- Middle 只允许固定 2× Average Pool；
- 特征按通道 RMS Normalize 后计算 L1；
- Adapter 不进入 Student 发布拓扑。

### 5.3 默认训练阶段

|阶段|初始化|默认 Step|初始 LR|
|---|---|---:|---:|
|Teacher FP32|随机|500k|2e-4|
|Student FP32|随机|400k|2e-4|
|Student KD|Student FP32|200k|1e-4|
|P10恢复|KD Student|80k，最多120k|5e-5|
|P15恢复|KD Student|120k，最多180k|5e-5|
|QAT Q1/Q2/Q3|恢复候选|2k/48k/10k|见第7章|

统一使用 AdamW、Weight Decay `1e-4`、Gradient Clip `1.0`、Cosine LR；长训练阶段前 5k Step Warmup。

### 5.4 24GB 显存策略

- 训练使用 Patch，不进行 Teacher/Student 全帧联合 KD；
- Teacher：`eval()`、`requires_grad=False`、`torch.no_grad()`、默认 FP32；
- Student：AMP + GradScaler；
- Micro Batch 1，梯度累积形成有效 Batch 8；
- 每个累积窗口固定包含 4 main + 4 tele；
- 显存不足依次启用 Student 激活检查点、降低 Patch 并提高累积次数、Teacher 特征 FP16 缓存；
- 默认不做 CPU Offload；
- 记录 `max_memory_allocated`，22GB 为软门槛；
- 禁止捕获 OOM 后静默跳过 Feature KD。

梯度累积只提高有效 Batch，不降低单样本激活显存。全帧验证时 Teacher 和 Student 顺序运行。

### 5.5 Checkpoint

训练 Checkpoint 保存模型、优化器、AMP Scaler、Step、阶段名和配置 Hash，只允许在可信研发环境加载。发布使用“Topology JSON + safetensors + Hash”，不得把 Pickle 作为发布格式。

---

## 6. Torch-Pruning 结构化剪枝

### 6.1 实现要求

实现锁定 `torch-pruning==1.6.1`，必须真正调用 DependencyGraph 和 DependencyGroup 改变张量 Shape。

同步依赖：

- Down 输出、Encoder、Skip、Up 和对应 Decoder；
- MobileNAFBlock 的 Feature Width；
- Spatial/Channel Expand 的 Gate 两半；
- DWConv `groups=in_channels=out_channels`；
- FiLM gamma/beta Head；
- 残差 beta/gamma 参数；
- 剪枝后 StaticSimpleGate 通道属性。

### 6.2 重要度

\[
I=0.5I_{Taylor}+0.3I_{Magnitude/Scale}+0.2I_{Activation}
\]

每项先归一化。Student 无 BN，Scale 项使用卷积通道范数与残差 beta/gamma。校准 Batch 必须覆盖 main/tele 和暗光分桶。

### 6.3 对齐与最小宽度

- Stage 1、Intro、Ending 不剪；
- 每轮 Feature Width 恰好裁 8 通道；
- Stage 2/3/Bottleneck 最低宽度为 `[24,48,96]`；
- 每个 NAFBlock 的隐藏宽度剪后恢复为严格 `2C`；
- Gate 两半采用成对索引裁剪；
- 非结构化稀疏不计入端侧压缩成果。

### 6.4 候选

|候选|参数下降目标|恢复要求|
|---|---:|---|
|P10|8%～12%|默认80k，最多120k|
|P15|13%～17%|默认120k，最多180k|

剪枝器在副本上预演每个 Stage 的结构影响，避免 P10 因单次高影响裁剪越界。每轮后必须完成前向、拓扑、8通道、Gate、DWConv和可重建验证。

本机 SIDD 冒烟实测：P10 参数下降约 9.23%、MAC下降约8.07%，宽度 `[16,32,56,128]`；P15 参数下降约14.44%、MAC下降约9.60%，宽度 `[16,32,56,120]`。这些只证明真实结构发生变化，不代表量产画质或真机加速。

### 6.5 自动选择

1. 所有画质、ONNX、QAT和NPU门槛通过；
2. P15 相对 Dense 的 P95 与峰值内存均改善至少8%，选P15；
3. 否则 P10 的 P95 改善至少5%，选P10；
4. 否则 Dense 只有满足AI P95≤8ms才可选；
5. 三者均不满足则 No-Go。

最终发布目录只复制被选中的一个 OM。

---

## 7. LSQ 与 LSQ+ W8A8 QAT

### 7.1 定义

- Conv/DWConv/Linear 权重：INT8 Signed、Symmetric、Per-output-channel、Offset=0；
- 激活：INT8 Signed、Per-tensor；
- LSQ：Activation Offset=0；
- LSQ+：Activation Scale 与 Offset 可学习；
- 实现 STE、LSQ Gradient Scale、Observer、校准、Phase冻结和Q/DQ导出。

LSQ+ 是必须实现和评测的候选，不预设 W8A8 必然优于对称 LSQ。

### 7.2 前置 Offset 微基准

在长周期 KD、剪枝恢复和 Q2 前，从 Dense W16 提取两个连续 MobileNAFBlock，生成对称与非对称 Q/DQ 图。

非对称 Offset 出现任一情况即冻结为0：

- 编译失败；
- CPU/GPU 回退；
- Conv/DWConv/Gate/Residual 融合断裂；
- P95 回归超过 `min(3%,0.1ms)`。

目标环境不可用时默认训练/发布候选使用 Offset=0；LSQ+ 仍以 Smoke 模式跑通代码，但不得进入量产选择。

### 7.3 Q0 校准

校准集至少4096帧，main/tele各不少于2048帧，暗部占比不低于60%。

每层搜索 `99.9/99.95/99.99/100` Percentile 候选，并在以下亮度桶分别计算归一化MSE后等权平均：

- Black Level附近；
- 暗部；
- 中灰；
- 高光。

该策略防止数量占优的暗部样本淹没高光。出现高光饱和、Banding或输出异常的候选直接淘汰。

### 7.4 QAT调度

|阶段|Step|网络权重|量化参数|默认 LR|
|---|---:|---|---|---:|
|Q0|校准|冻结|范围搜索/MSE初始化|无|
|Q1|2k|冻结|可学习|1e-4|
|Q2|48k|可学习|可学习|Weight 1e-5，Quant 1e-4|
|Q3|10k|可学习|冻结|5e-6|

Q3 只切换 `requires_grad`，不得重新估计 Scale/Offset；冻结前后最大绝对漂移必须严格为0。

### 7.5 FiLM精度门禁

默认全W8A8。只有满足以下任一画质触发条件才训练FiLM FP16 NPU Island候选：

- FiLM量化消耗PSNR预算超过0.03dB；
- 关键层饱和率超过1%；
- main/tele或高ISO分桶出现异常。

FP16 Island必须100%位于NPU、无CPU/GPU回退，并继续满足P95≤8ms。否则返回全INT8并优化量化范围。

### 7.6 Q/DQ

QAT冻结后把连续 Offset 吸收到可表示的 int8 Zero-point 网格，导出显式 `QuantizeLinear/DequantizeLinear`。

逐层 Quant Policy 至少记录 Bits、Signed、Symmetric、Scale、Offset/Zero-point、Axis、是否 Per-channel及Hash。训练态Observer和可学习参数不得残留在发布图。

---

## 8. ONNX、OM 与失败闭锁

### 8.1 单 Shape ONNX

只导出：

```text
dark_preview_ryyb_4x3.onnx
packed_raw = 1×4×768×1024
condition  = 1×24
noise_pred = 1×4×768×1024
```

Smoke 模式可用等拓扑小 Shape 验证，Release 模式必须导出固定发布 Shape。

### 8.2 图审计

- 禁止动态 `Split/SplitToSequence`；
- Slice Starts/Ends/Axes/Steps 必须为 Constant 或 Initializer；
- Gate 两半相等；
- 不得存在动态 Shape；
- PyTorch→ONNX 最大绝对误差≤`1e-4`；
- QAT ONNX 必须显式包含 Q/DQ；
- 算子不在白名单即阻断 ATC。

### 8.3 单 OM

最终名称：

```text
dark_preview_ryyb_4x3_int8.om
```

ATC 输入 Shape固定为：

```text
packed_raw:1,4,768,1024;condition:1,24
```

研发阶段可生成 P10/P15、LSQ/LSQ+ 候选 OM 做 A/B；最终 Release 包只允许一个 OM。缺少商用 DDK/ATC 时编译器返回 `available=false`，禁止生成占位 OM。

---

## 9. 4K30 和 AI 节点预算

30fps 帧周期为 33.33ms。AI RAW Denoise 顶层预算：

|统计|门槛|
|---|---:|
|P50|≤6ms|
|P95|≤8ms|
|P99|≤9ms|
|硬超时|10ms|

P95 定位预算：

|阶段|预算|
|---|---:|
|Pack/Normalize/Input DMA|≤1.0ms|
|NPU|≤6.0ms|
|Subtract/Clamp/Output DMA|≤0.7ms|
|Queue/Fence|≤0.3ms|

计时边界从输入 Fence 可读开始，到输出 Fence 可供下游消费结束，包含排队、DMA和同步。最终以同一帧完整 Timeline 为准，不能把 NPU Kernel 单项时间当作 AI 节点时间。

P95 预算给外围 Pipeline 约25.33ms，硬超时时剩余约23.33ms。若外围流程不能在余量内完成，应优化外围节点或判定系统 No-Go，不得把AI预算静默扩大。

10ms超时立即走传统ISP路径，禁止调用CPU MobileNAFNet。

---

## 10. 验收矩阵

### 10.1 阶段质量预算

|阶段|参考|RAW PSNR最大下降|SSIM最大下降|
|---|---|---:|---:|
|KD Student|W32 Teacher|0.20dB|0.003|
|P10/P15恢复|KD Student|0.10dB|0.002|
|INT8 QAT|对应FP32候选|0.10dB|0.002|
|最终INT8|W32 Teacher|0.35dB|0.005|

任一Camera或高ISO分桶相对Teacher下降超过0.50dB直接失败。INT8相对FP32的平均ΔE00增量≤0.3。

### 10.2 画质指标

- RAW PSNR/SSIM；
- Reference ISP RGB PSNR/SSIM/MS-SSIM；
- ΔE00 Mean/P95；
- Noise Power Spectrum；
- Row/Column Noise；
- Edge MTF、梯度和暗部纹理保留；
- 过平滑、Banding、色斑、Halo、Ringing、Maze、False Color；
- 文字错误和虚假纹理；
- 主摄/长焦、ISO、Lux、温度分桶；
- 连续帧亮度、色彩和噪声稳定性。

### 10.3 数值和工程

- RYYB四相位Pack/Unpack位精确；
- 奇数Crop拒绝；
- Condition main/tele组合校验；
- `strength=0`恒等；
- Teacher无梯度，Student真实反向；
- Checkpoint配置Hash与恢复；
- P10/P15参数和MAC真实下降；
- 剪枝模型可按Topology重建并加载；
- Q3冻结零漂移；
- PyTorch、ONNX Q/DQ和目标端进入同一数值报告。

### 10.4 目标设备

- NPU覆盖100%，未批准CPU/GPU回退为0；
- AI节点P50/P95/P99和10ms超时；
- 整条4K30 Pipeline；
- 冷态与10/30分钟热稳态；
- 10000帧稳定性；
- 模型常驻、Workspace、I/O Buffer和峰值内存；
- 功耗、温升、降频；
- 主摄/长焦切换及传统ISP Bypass。

---

## 11. 当前工程实现与证据

截至2026-08-05已实现：

- 固定RYYB Profile、四相位Pack/Unpack、HAL准入和Condition收紧；
- RYYB JSONL Manifest、相位安全Patch和Camera平衡采样；
- W32 Teacher、W16 Student、Stage3/Middle特征接口；
- 可恢复监督训练和KD，Teacher `no_grad`、Student AMP/累积；
- Torch-Pruning DependencyGraph真实P10/P15及拓扑重建；
- LSQ/LSQ+权重Per-channel、激活Per-tensor整网QAT；
- 长周期训练前双Block对称/非对称Offset Q/DQ微基准生成与失败闭锁；
- 亮度桶均衡MSE校准、Q1/Q2/Q3和FiLM门禁函数；
- 显式Q/DQ ONNX、单Shape导出、单OM编译适配和失败闭锁；
- Python单元测试与CPU全阶段流水线。
- 固定RYYB C++ Runtime已用MSVC 19.42/C++17重新编译，CTest `1/1`通过。

CPU Smoke 使用少量SIDD、32×32 Patch、各训练阶段1 Step，已运行：

```text
Offset前置微基准 → Teacher → Student → KD → P10 → P10恢复 → P15 → P15恢复
→ P10/P15 × 对称LSQ Q0/Q1/Q2/Q3
→ P10/P15 × 非对称LSQ+ Q0/Q1/Q2/Q3
→ Q/DQ ONNX → OM编译门禁 → Manifest
```

结果为 `engineering_passed`；ONNX 包含显式 `QuantizeLinear/DequantizeLinear`，OM 因缺少目标 ATC 明确为 unavailable。该结果不代表训练收敛、RYYB画质或麒麟9000性能。

---

## 12. 发布 Go/No-Go

全部满足才允许把 `release_ready` 改为 `true`：

1. 真实主摄/长焦RYYB数据量、标定、无泄漏和盲测通过；
2. Teacher、Student、KD、剪枝恢复和最终QAT完整训练收敛；
3. 阶段画质预算及主观盲测通过；
4. LSQ/LSQ+与FiLM精度选择有目标端A/B证据；
5. Q/DQ→CANN适配→ATC→OM逐层参数一致；
6. 最终发布包只有一个固定Shape OM；
7. 100% NPU且CPU/GPU回退为0；
8. AI节点6/8/9/10ms和整条4K30通过；
9. 内存、功耗、热稳态、10000帧和Camera切换通过；
10. Schema、Sensor、Topology、Quant Policy、ONNX、OM和Runtime Hash一致；
11. 阻断级和高优先级缺陷为0；
12. 回滚到传统ISP NR验证通过。

任何一项缺失均保持 `release_ready=false`。

---

## 13. 实施入口和制品

CPU全阶段：

```powershell
.\.venv\Scripts\python.exe -m ai_isp.pipeline --config configs\train\v4_cpu_全流程.yaml
```

量产训练配置：

```powershell
.\.venv\Scripts\python.exe -m ai_isp.pipeline --config configs\train\v4_量产训练.yaml
```

核心发布文件：

```text
model_fp32.safetensors
topology.json
condition_schema_v2.json
sensor_profiles_ryyb.json
quant_policy.json
dark_preview_ryyb_4x3.onnx
dark_preview_ryyb_4x3_int8.om
model_manifest_v4.json
```

开发过程和限制持续记录在 [V4开发关键步骤与注意事项](./docs/V4开发关键步骤与注意事项.md)，CPU证据见 [V4 CPU全阶段验证报告](./docs/V4%20CPU全阶段验证报告.md)，项目介绍与学习入口见 [V4项目总结与学习指南](./AI%20ISP%20V4%20项目总结与学习指南.md)。

---

## 14. 参考资料

1. [NAFNet](https://arxiv.org/abs/2204.04676)：非线性激活自由图像恢复基础。
2. [Torch-Pruning](https://github.com/VainF/Torch-Pruning)：DependencyGraph结构化剪枝。
3. [LSQ](https://arxiv.org/abs/1902.08153)：可学习量化步长及梯度缩放。
4. [LSQ+](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Bhalgat_LSQ_Improving_Low-Bit_Quantization_Through_Learnable_Offsets_and_Better_Initialization_CVPRW_2020_paper.pdf)：可学习Offset和MSE初始化。
5. [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)：`eval/no_grad/inference_mode`边界。
6. [CANN QAT模型适配](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/devaids/amct/atlasamct_16_0148.html)：Q/DQ模型适配与ATC链路。

---

## 15. 结论

V4.0 的核心不是增加更多 Camera、Profile 或运行时分支，而是把主摄/长焦 RYYB 的单模型算法链做到可训练、可压缩、可审计和可验收。超广角传统 NR 解耦、固定 Shape、单 OM 和严格失败闭锁降低了部署不确定性；真实结构化剪枝和完整 LSQ/LSQ+ QAT 使模型压缩不再停留在文档或校验器层面。

当前工程已经证明全部算法阶段可以在无 GPU 电脑上用少量数据跑通。下一步决定量产成败的不是继续扩展框架，而是真实 RYYB 数据质量、完整收敛训练以及麒麟 9000 上的 NPU、8ms P95、4K30、功耗和热稳态证据。
