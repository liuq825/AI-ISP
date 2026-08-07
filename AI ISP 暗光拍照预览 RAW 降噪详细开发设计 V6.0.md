# AI ISP 暗光拍照预览 RAW 降噪详细开发设计 V6.0

> 文档版本：6.0.0
> 更新日期：2026-08-07
> 目标平台：麒麟 9000 手机平台
> 目标模式：Dark Scene Photo Preview，30fps 预览
> 部署模型：Conditional MobileNAFNet-W16，Main/Tele RYYB 共享模型
> 强制精度：FP16 + INT8 混合精度量化
> 强制剪枝：通过开发环境模拟验证后的 16 通道对齐结构化剪枝
> 开发收敛原则：候选仅存在于临时评测区；模拟验证结束后立即只保留综合最优的一个拓扑、一个 Quant Policy 和一个 ONNX 作为后续开发基线
> 当前状态：V4 算法工程链已在 CPU 冒烟环境跑通；V6 新增项、真实 RYYB、商用 DDK、OM 和麒麟 9000 验收尚未完成，`development_selected=true` 仅可在完成模拟选优后设置，当前及选优后均保持 `dynamic_affine_target_pending=true`、`target_validated=false`、`release_ready=false`

---

## 0. 执行摘要

V6.0 以 V4.0 已有工程闭环为基础，吸收《Kirin9000 Dark Photo Preview AI RAW Denoise V5.6 量产研发设计规范》中可落地的方案，并修正其中关于物理 CFA 相位、剪枝比例、时序损失和量化支持性的冲突。

最终研发主线为：

```text
真实 Main/Tele RYYB 数据与标定
→ Linear RAW 与物理 CFA 准入
→ W32 Teacher
→ W16 Student + Feature KD
→ RYYB 相关噪声与 Preview Temporal Stability 训练
→ 16 通道对齐静态分析与 CPU/GPU/ONNX 模拟验证
→ P36-16 指导拓扑与 P10-16 / P18-16 保护候选模拟验证
→ 三候选分别执行 Phase1 全尺寸适配、Q0 与 Q1 LSQ+ 探针
→ FakeQuant 画质、算子、Cast、参数/MAC 与内存代理指标综合选择
→ 开发主线立即只保留唯一模拟最优方案
→ 获胜方案执行 Q2/Q3 FP16 + INT8 Mixed Precision LSQ+ QAT
→ 固定 Shape Q/DQ ONNX
→ 未来商用 DDK/真机仅验证该唯一方案
```

冻结结论：

|项目|V6.0 决策|
|---|---|
|AI Camera|RYYB 广角主摄、RYYB 潜望式长焦|
|超广角|RGGB 超广角不接入本模型，固定走传统 ISP NR|
|模式边界|暗光 Photo Preview；不覆盖 Capture 最终照片、4K30/4K60 视频录制|
|RAW Domain|`LINEAR_POST_BLC_LSC_PRE_DGAIN`|
|模型输入|`packed_raw[1,4,768,1024]`，对应 `2048×1536`、4:3 RYYB|
|规范通道|`[R,Yr,Yb,B]`；Main/Tele 物理相位允许不同|
|Condition|ConditionSchemaV2，`float32[1,24]`|
|模型输出|同格点 `noise_pred`|
|剪枝|优先验证指导拓扑 P36-16，并与 P10-16/P18-16 比较后立即只保留一个|
|量化|发布方案必须同时包含 FP16 Island 和 INT8 Backbone|
|推理|Full Tensor Single Pass，固定 Shape，禁止 Tile/Stitch|
|发布|最终目录只允许一个获胜模型及 `dark_preview_ryyb_4x3_mixed_int8_fp16.om`|
|AI 节点预算|P50≤6ms、P95≤8ms、P99≤9ms、10ms 硬超时|

### 0.1 V5.6 指导方案可行性结论

|V5.6 主张|结论|V6.0 处理|
|---|---|---|
|Main/Tele 共享 Conditional W16|采纳|共享 Backbone，通过 Condition 和 FiLM 适配|
|Main/Tele 物理 CFA 相位一致|拒绝作为前提|每颗 Sensor 独立声明相位，模型输入统一语义|
|Linear RAW Contract|采纳并加强|增加明确 RAW Domain 状态和 Hash|
|Full Tensor、Static OM|采纳|固定 `1×4×768×1024`|
|HardTanh FiLM|采纳|用常量边界 `Clip`，必须重新训练与导出审计|
|固定 `[16,32,48,96]` 且剪枝 10%～15%|修正后采纳|保留该指导拓扑并按真实压缩率命名 P36-16，不再错误标注为 10%～15%|
|16 通道对齐|模拟验证后执行|通过 Shape、MAC、内存、FakeQuant、ONNX 和算子兼容性检查后成为正式开发约束|
|RYYB 相关噪声|修正后采纳|使用分 Sensor、ISO、温度标定的信号相关协方差|
|`||N1_pred-N2_pred||` 时序损失|拒绝原公式|改为同一 Clean 下两次降噪输出的一致性|
|默认混合精度|强制采纳|所有量产候选必须为 FP16+INT8；失败时 No-Go，不回退全 INT8|
|Gate Per-channel Activation Scale|拒绝|V6 激活固定使用成对 Per-tensor Scale，不建立 Per-channel 分支|
|验证后多方案并存|拒绝|模拟评测完成即淘汰其他模型制品，开发主线只保留综合最优方案|

---

## 1. 项目范围与职责边界

### 1.1 负责范围

- Main/Tele RYYB Linear Packed RAW 的单帧噪声预测；
- 通过 Sensor、Camera、噪声、曝光、温度和计划数字增益条件完成域适配；
- 暗部随机噪声、行列噪声、RYYB 通道相关噪声、色噪和纹理保真；
- W32 Teacher、W16 Student、蒸馏、时序稳定性训练、16 通道剪枝与恢复；
- FP16+INT8 Mixed Precision LSQ+ QAT；
- 固定 Shape ONNX、Q/DQ、Quant Policy、Topology、OM 和发布 Hash；
- Main/Tele 独立画质、数值、时延、内存、功耗和热稳态验收。

### 1.2 不负责范围

- RGGB 超广角 AI RAW 降噪；
- Capture 最终照片的独立多帧降噪模型；
- 4K30/4K60 视频录制时域降噪；
- Demosaic、AWB、CCM、Tone Mapping、锐化和 JPEG/HEIF 编码；
- 原生 `4096×3072` RAW 全格点推理；
- Tile、Overlap、Stitch 或 CPU 拼接；
- 输入相位错误时自动交换通道或猜测 CFA；
- 用 CPU/ONNX Runtime 作为量产实时 AI 回退；
- 在缺少真实数据、商用 DDK 和真机证据时宣称量产完成。

### 1.3 30fps 与 4K 的边界

30fps 是暗光 Photo Preview 的帧率要求。AI 节点处理上游协商的 `2048×1536` RYYB 支路，后续 ISP 和显示链可完成 4K 构图与输出，但本项目不等同于 4K30 视频录制降噪。

禁止把 AI Tensor 偷换为原生 `4096×3072` RAW。Dense W16 在固定发布 Shape 下约为 35.58 GMAC；面积放大四倍后该统计与现有代理验证均失效，不具备进入 8ms P95 目标验证的准入资格。

---

## 2. 系统 Pipeline 与 Linear RAW 契约

### 2.1 固定 Pipeline

```text
Sensor RAW
→ Black Level Correction
→ Lens Shading Correction
→ Preview RAW Binning / Skip
→ Even Boundary Crop
→ RYYB Semantic Packing
→ FP16 Normalize
→ Mixed Precision AI RAW Denoise
→ FP16 raw_out = clamp(raw_in - noise_pred, 0, 1)
→ Semantic Unpack
→ 2D Physical RYYB CFA RAW
→ Digital Gain
→ Demosaic
→ AWB / CCM
→ Tone Mapping
→ Display
```

AI 输入必须保持线性。进入 AI 前禁止执行 Digital Gain、Gamma、Tone Mapping、Local Contrast 和锐化。若 LSC 已在 AI 前执行，训练数据和噪声标定必须处于相同的 Post-LSC 域。

Post-BLC/LSC 输入归一化公式冻结为：

\[
x_c=\operatorname{clip}\left(\frac{raw_{post\_lsc,c}}{white_c-black_c},0,1\right)
\]

HAL 提供的 `raw_post_lsc` 已完成 BLC，Formatter 禁止再次减去 Black Level。Black/White Level 只用于范围校验和分母计算；LSC 增益、裁剪和饱和语义必须与训练数据完全一致。

HAL 必须显式提供：

```text
raw_domain_state = LINEAR_POST_BLC_LSC_PRE_DGAIN
blc_applied = true
lsc_applied = true
lsc_profile_hash = <versioned hash>
```

状态缺失、不匹配，或 RAW Domain/LSC Profile Hash 不一致时禁止调用 AI。

Condition 中的 `digital_gain` 表示 AI 输出之后计划施加的数字增益，不表示该增益已经作用于输入 Tensor。

### 2.2 模型接口

|名称|类型|Shape|语义|
|---|---|---|---|
|`packed_raw`|OM 边界 FP16|`1×4×768×1024`|归一化 `[R,Yr,Yb,B]`|
|`condition`|FP32|`1×24`|ConditionSchemaV2|
|`noise_pred`|FP16 输出|`1×4×768×1024`|与输入同符号、同格点的噪声残差|

`condition` 在公共接口保持 FP32，模型图内只允许一次显式 FP32→FP16 Cast，随后进入 FP16 Condition MLP；禁止在每个 FiLM Head 前重复 Cast。

唯一重建公式：

\[
raw_{out}=\operatorname{clamp}(raw_{in}-noise_{pred},0,1)
\]

禁止在文档、模型或 Runtime 中把该操作称为含义相反的 `Residual Add`。OM 只输出 FP16 `noise_pred`；Subtract/Clamp 与 Semantic Unpack 位于 OM 外的 NPU/ISP 向量后处理路径，禁止逐像素 CPU 实现，并计入 AI 节点的输出后处理预算。

`enhancement_strength` 位于 Condition 第 23 维，只在模型图内缩放一次 `noise_pred`。强度为 0 时必须满足位级可解释的恒等路径，Runtime 不得二次缩放。

### 2.3 Main/Tele 物理相位与规范语义

支持四种物理 2×2 RYYB 相位：`ryyb/byyr/yryb/ybyr`。

- Main 和 Tele 分别由版本化 Sensor Profile 声明物理相位；
- 不要求两颗 Sensor 的物理相位相同；
- Formatter 把所有合法相位统一打包为 `[R,Yr,Yb,B]`；
- `Yr` 是与 R 同行的黄色像素，`Yb` 是与 B 同行的黄色像素；
- 模型看到的通道语义必须一致，物理相位差异不得进入模型分支；
- Subtract/Clamp 后必须执行 Pack 的位精确逆操作，根据同一 Sensor Profile 把 `[R,Yr,Yb,B]` 恢复为二维物理 RYYB Mosaic；
- Unpack 禁止固定输出 `ryyb` 相位，也禁止根据 Camera 名称猜测相位。

### 2.4 2×2 相位死规则

HAL Crop 的 `x/y/width/height` 必须全部为非负偶数。禁止：

- 奇数 Crop 后交换通道；
- 普通 Resize 到固定 Shape；
- 补一行、丢一列或根据 Camera 名称猜测 CFA；
- 未注册 Sensor Profile 进入 AI。

### 2.5 准入条件

以下全部满足才允许调用 AI：

1. Camera 为 `main` 或 `tele`；
2. Sensor Profile 和物理 CFA 相位在发布清单注册；
3. RAW Domain 为 `LINEAR_POST_BLC_LSC_PRE_DGAIN`；
4. RAW/Crop 精确为 `2048×1536`，起点和宽高均为偶数；
5. RAW10/12/14/16、uint16 容器和 Stride 合法；
6. 四通道 White Level 分别大于 Black Level；
7. Camera/Sensor/Lens/Condition 组合合法；
8. `blc_applied/lsc_applied` 为真，LSC Profile Hash 匹配；
9. Model、Topology、Condition、Sensor、RAW Domain、Unpack Profile 和 Quant Policy Hash 全部匹配；
10. 共享 Buffer、Stride、Offset 和 Fence 合法；
11. NPU 可用且无热保护或 Camera 过渡状态。

任一条件失败直接交回传统 ISP NR，禁止静默执行 CPU AI。

### 2.6 DMA-BUF 零拷贝共享契约

Camera HAL、AI Runtime 和后级 ISP 必须通过平台共享 Buffer 抽象传递数据。Android/Linux 优先使用 DMA-BUF Heap；遗留平台可由 `libdmabufheap` 映射到底层 ION，但 ION 不是公共接口要求。

- Buffer FD 在 Stream 初始化时导入一次，通过池化跨帧复用；
- 每帧只传递 Buffer Index、Plane Offset、Row/Plane Stride、有效区和生产者 Fence；
- 禁止额外 CPU `memcpy`、逐帧 `mmap/munmap` 和逐像素 CPU Pack/Unpack；
- Exporter/Importer 必须实现 Cache Clean/Invalidate、独占/共享 Fence 和超时后的 Buffer 回收；
- Buffer 生命周期必须覆盖 NPU 读、向量后处理、Unpack 和下游 ISP 消费；
- 允许 IOMMU 映射和 Scatter-Gather，不要求物理连续；只有设备 Importer 明确要求时才使用 CMA/专用连续 Heap；
- Pack/Normalize 是二维 RAW 到 FP16 四平面的真实格式变换，只有上游能直接写入模型布局或由硬件融合执行时才属于完全零拷贝。

单个 FP16 输入或 `noise_pred` 为 6,291,456 Byte，两者合计约 12.58 MB；该数字不包含源 RAW、重建结果、Unpack 和其他中间写流量。硬门槛为每帧额外 CPU `memcpy=0 Byte`。Buffer 已导入后的 Fence/同步开销以 ≤0.2ms 为优化目标，不替代第 9.1 节的 1.0ms 输入和 0.7ms 输出总预算。

---

## 3. 数据建设与物理噪声

### 3.1 数据规模与分割

每颗物理 Camera 的采集目标不少于 5,000 组通过质量审查的 Noisy/Clean RAW Pair。数量之外还必须分别满足：

|集合|每颗 Camera 最低独立场景数|
|---|---:|
|Train|3000|
|Validation|300|
|Blind Test|500|

同一 Scene、Burst 或连续片段只能属于一个集合。主摄和长焦分别计数，不允许相互填补。

### 3.2 Manifest

真实 RYYB 数据使用 JSONL Manifest，至少记录：

- Sample、Scene、Burst、Split、Camera 和 Sensor Profile；
- 物理 CFA 相位、RAW Domain、Noisy/Clean 路径；
- 位深、四通道 Black/White Level；
- Exposure、ISO、模拟/计划数字增益；
- Sensor 温度、Lux/EV、场景亮度；
- Shot/Read Noise、行列噪声、相关噪声 Profile；
- Reference 生成版本、对齐残差、运动/饱和/坏点 Mask；
- 数据来源和 `smoke_only` 标记。

### 3.3 Clean Reference

优先级：

1. 同场景低 ISO/长曝光、运动受控的配对参考；
2. 多帧鲁棒对齐和融合生成的参考；
3. 经人工与统计审查的高质量传统 Pipeline 参考。

存在配准重影、饱和、运动遮挡或坏点污染的区域不得作为强监督真值。

### 3.4 Patch 与增强

- 严格按指导方案使用 Online Dynamic Crop，不做 Offline 重复扩增；
- Stage1 使用 Mosaic `256×256`，对应 Packed `4×128×128`；
- Stage2/剪枝恢复使用 Mosaic `512×512`，对应 Packed `4×256×256`；
- 后期部署尺寸适配、QAT、验证和导出使用 Packed `4×768×1024`；
- Crop 起点始终为偶数；
- 默认允许 H Flip、V Flip 和 180° Rotation；
- 绝对禁止 90° Rotation；
- 每个有效 Batch 中 Main/Tele 数量相等；
- 按 ISO、Lux、曝光、温度和场景类型分层采样。

### 3.5 RYYB 相关噪声模型

真实 RAW 占训练样本 80%～90%，合成样本占 10%～20%。合成噪声不得只使用固定标量 Poisson-Gaussian，而应包含：

\[
\Sigma(x,c)=A(c)\operatorname{diag}\!\left(a(c)\odot\max(x,0)\right)A(c)^T+\Sigma_{read}(c)
\]

其中 `c` 表示 Sensor、ISO、温度和增益条件，`a(c)` 为四通道 Shot Noise 系数，`A(c)` 为 RYYB Crosstalk/相关性映射，`Σ_read(c)` 为信号无关读噪协方差。该表达保持 Shot Noise 对信号的一次关系，并通过 `A diag(.) A^T` 保证相关项可解释。实现要求：

- 每个 Sensor 独立标定；
- `Σ` 必须对称、正半定；
- 通过 Cholesky 或特征分解生成四通道相关噪声；
- 同时建模 Shot、Read、Row、Column、Fixed Pattern 和量化噪声；
- 对比真实/合成噪声的 Histogram、NPS、通道相关矩阵和行列统计；
- 标定不足时降低合成比例，不得伪造固定协方差作为量产数据。

### 3.6 SIDD 边界

SIDD 只有 RGGB/BGGR/GRBG/GBRG 数据，只允许用于 Dataset、梯度、Checkpoint、剪枝、QAT、ONNX 和流水线冒烟。SIDD 结果不得用于选择 RYYB 量产模型。

---

## 4. Teacher、Student 与 FiLM

### 4.1 Teacher

Conditional NAFNet-W32：

- Feature Width `[32,64,128,256,512]`；
- Encoder Block `[2,2,4,8]`，Middle 4，Decoder `[2,2,2,2]`；
- 只用于 FP32 训练和 KD，不进入发布图；
- 输出 `noise_pred`；
- 向 KD 暴露 Encoder Stage3 和 Middle 特征。

### 4.2 Dense Student

Conditional MobileNAFNet-W16：

- Dense Feature Width `[16,32,64,128]`；
- Encoder `[2,2,4]`，Middle 2，Decoder `[2,2,2]`；
- Dense 参数量 660,836；
- 固定发布 Shape 约 35.58 GMAC；
- 仅使用可审计的 Conv、DWConv、Add、Mul、Clip、Nearest Resize、Linear 和常量 Slice。

MobileNAFBlock-DW：

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

当前发布拓扑不包含 LayerNorm，部署审计不得引用不存在的 LayerNorm 融合。

### 4.3 Conditional FiLM

24 维 Condition 经过 FP16 MLP：

```text
24 → Linear 64 → ReLU → Linear 128 → ReLU
```

FiLM 注入 Encoder Stage2、Encoder Stage3 和 Middle。公式冻结为：

\[
F'=F\left(1+0.1\operatorname{clip}(\gamma,-1,1)\right)
+0.1\operatorname{clip}(\beta,-1,1)
\]

要求：

- Condition Encoder 和 FiLM Head 不主动剪枝；
- Feature Width 变化时 Gamma/Beta 输出被动同步；
- ONNX 导出为常量边界 `Clip`；
- 从 Tanh 改为 Clip 后必须重新训练，不允许直接替换已训练权重并宣称等价。

---

## 5. 训练、蒸馏与时序稳定性

### 5.1 Teacher 监督损失

\[
L_{teacher}=0.55L_{RAW}+0.30L_{Tone}+0.15L_{Gradient}
\]

- RAW：暗部加权 Charbonnier，屏蔽饱和和无效参考；
- Tone：按 Camera 选择的版本化 RYYB Reference ISP 后 RGB/Luma Loss；
- Gradient：Reference ISP Luma 的 Sobel X/Y。

### 5.2 Student 与 KD

Student 使用监督损失、Stage3/Middle Feature KD 和 Attention/统计蒸馏。训练期 Adapter 不进入发布拓扑。

特征按通道 RMS Normalize 后计算 L1；Middle 只允许固定 2× Average Pool 对齐。Stage3 和 Middle 各占 `L_FeatureKD` 的 50%。

Attention KD 对每个对齐特征计算：

\[
\mathcal{A}(F)=\operatorname{normalize}\left(\operatorname{mean}_{c}|F|\right)
\]

Teacher/Student 的 Stage3 与 Middle Attention Map 分别计算 L1，再等权平均。完整 Student Loss 冻结为：

\[
\begin{aligned}
L_{student}={}&L_{RAW}+0.5L_{Tone}+0.1L_{Gradient}\\
&+0.1L_{FeatureKD}+0.05L_{AttentionKD}\\
&+0.05L_{Temporal}+0.01L_{Gate}
\end{aligned}
\]

### 5.3 Preview Temporal Stability

V5.6 的 `||N1_pred-N2_pred||` 会错误约束两次独立噪声的真实残差相同，可能把模型推向零预测或过平滑。V6.0 将 Synthetic Temporal 的输入、预测和重建显式定义为：

\[
z_i=\operatorname{clip}(x+n_i,0,1),\quad N_i=f_\theta(z_i,c)
\]

\[
y_i=\operatorname{clip}(z_i-N_i,0,1)
\]

\[
L_{temp}=\frac{\|M\odot(y_1-y_2)\|_1}{\sum M}
\]

其中 `x` 为同一 Clean RAW，`n1/n2` 为独立采样噪声。Mask 冻结为：

\[
M=M_{valid}\cdot M_{motion}\cdot M_{sat}
\]

\[
M_{sat}=\mathbf{1}_{\{x<0.98\}}\cdot\mathbf{1}_{\{z_1<0.98\}}\cdot\mathbf{1}_{\{z_2<0.98\}}
\]

`0.98` 使用归一化 RAW 单位；高光截断、运动、遮挡和无效参考区域均不得贡献 Temporal 梯度。

真实连续帧训练或验证必须先做运动对齐，并使用 Forward/Backward Flow 一致性生成遮挡 Mask。模型推理仍是单帧、无时序状态，不增加 Runtime 帧缓存。

`λ_temp` 默认 0.05，只允许在 0.02～0.05 内搜索；任何时序收益不得以明显纹理损失或拖影为代价。

### 5.4 Gate Scale 约束

SimpleGate 两个输入分支默认使用成对 Per-tensor Scale，并加入：

\[
L_{gate}=|\log(S_{X1}+\epsilon)-\log(S_{X2}+\epsilon)|
\]

默认权重 0.01。V6 激活量化固定使用 Per-tensor Scale，不允许切换为 Activation Per-channel。

### 5.5 默认训练阶段

|阶段|初始化|默认 Step|初始 LR|
|---|---|---:|---:|
|Teacher FP32|随机|500k|2e-4|
|Student FP32|随机|400k|2e-4|
|Student KD + Temporal|Student FP32|200k|1e-4|
|P10-16 恢复|KD Student|80k，最多 120k|5e-5|
|P18-16 恢复|KD Student|120k，最多 180k|5e-5|
|P36-16 恢复|KD Student|180k，最多 240k|5e-5|
|候选 Mixed Precision 探针|三个已恢复剪枝候选|Phase1/Q0/Q1，见第 7 章|分组 LR|
|获胜方案 Mixed Precision LSQ+ QAT|唯一剪枝获胜方案|Q2/Q3，见第 7 章|分组 LR|

统一使用 AdamW、Weight Decay `1e-4`、Gradient Clip `1.0`、Cosine LR；长阶段前 5k Step Warmup。

### 5.6 24GB 显存策略

- Teacher `eval()`、`requires_grad=False`、`torch.no_grad()`；
- Student AMP + GradScaler；
- Micro Batch 1，梯度累积 16，形成指导文件要求的有效 Batch 16；
- 每个累积窗口固定包含 8 Main + 8 Tele；
- 训练使用 Patch，不强制 Teacher/Student 全帧联合 KD；
- 全帧用于顺序验证、ONNX 导出和模拟数值检查；
- 显存不足依次启用 Student 激活检查点、降低 Patch、提高累积次数和 Teacher 特征 FP16 缓存；
- 禁止捕获 OOM 后静默跳过 KD 或 Temporal Loss。

---

## 6. 16 通道对齐结构化剪枝

### 6.1 开发环境前置模拟验证

当前没有商用 DDK 和麒麟 9000 真机，因此 16 通道对齐先通过可复现的开发环境模拟验证：

1. Torch-Pruning DependencyGraph 能合法生成 16 对齐拓扑；
2. 剪枝模型能完成 FP32/FakeQuant 前向、反向、保存和无 Pickle 重建；
3. 所有 Feature Width、Gate 两半、DWConv Groups、Skip、Decoder 和 FiLM Head 严格同步；
4. PyTorch→ONNX 导出成功，静态 Shape、算子白名单和 Q/DQ 图审计通过；
5. 分析参数量、MAC、峰值 Activation、Workspace 代理量、Cast 数量和理论 DDR 流量；
6. CPU/GPU FakeQuant 数值稳定，无 NaN/Inf、异常饱和或明显画质崩溃；
7. 相对 Dense，剪枝候选必须获得与其压缩目标相符的参数或 MAC 收益。

以上全部通过后，正式剪枝器冻结 `round_to=16`。验证失败时不得静默切回 8 通道剪枝；整改失败则保持 `release_ready=false`。

模拟验证只用于确定开发方案，不能表述为 ATC 编译成功、100% NPU 或麒麟 9000 时延已通过。

### 6.2 候选拓扑

当前拓扑按 16 通道裁剪时不存在严格的 P15 档，因此使用真实压缩率命名：

|候选|Feature Width|参数量|参数下降|MAC|MAC下降|
|---|---|---:|---:|---:|---:|
|Dense|`[16,32,64,128]`|660,836|0|35.58G|0|
|P10-16|`[16,32,64,112]`|591,284|10.52%|34.44G|3.20%|
|P18-16|`[16,32,48,128]`|543,396|17.77%|30.06G|15.52%|
|P36-16（V5.6 指导拓扑）|`[16,32,48,96]`|419,652|36.50%|28.14G|20.92%|

V5.6 建议的 `[16,32,48,96]` 不属于其文字描述的 10%～15% 剪枝，但其 16 通道拓扑是本项目必须优先验证的指导方案。V6.0 保留该拓扑并按真实降参比例命名为 P36-16，避免用错误比例掩盖实际压缩强度。

参数与 MAC 必须由发布 Shape `1×4×768×1024` 的实际导出拓扑重算。MAC 口径固定为 Conv/Depthwise Conv/Linear 的乘加次数；Add、Mul、Resize、Clip、Q/DQ 和 Cast 不计入 MAC，但必须分别进入节点数、Activation、Workspace 与 DDR 流量报告。禁止在候选间混用 FLOPs、MAC 或缩小输入 Shape 的统计结果。

### 6.3 Torch-Pruning 要求

实现锁定 `torch-pruning==1.6.1`，必须调用 DependencyGraph 和 DependencyGroup 真实改变 Tensor Shape。

必须同步：

- Down、Encoder、Skip、Up 和 Decoder；
- Feature Width 与所有 MobileNAFBlock；
- SimpleGate 两半的成对索引；
- DWConv `groups=in_channels=out_channels`；
- FiLM Gamma/Beta Head 输出；
- 残差 Beta/Gamma 参数；
- StaticSimpleGate 的构造期通道属性。

Intro、Ending、Stage1、Condition Encoder 和 FiLM Head 不主动剪枝。非结构化稀疏不计入端侧剪枝成果。

### 6.4 剪枝候选模拟选择与立即收敛

P36-16、P18-16 和 P10-16 在临时候选区分别完成候选专属恢复训练，然后都执行 Phase1 FP16 全尺寸适配 10k、Q0 校准与 Q1 2k LSQ+ 探针，再比较 FakeQuant 画质、ONNX 数值和资源代理指标。选择完成后只有获胜拓扑继续 Q2/Q3：

1. 优先验证 V5.6 指导拓扑 P36-16；
2. 任一候选超过阶段画质预算、出现数值异常或图审计失败，立即淘汰；
3. P36-16 通过门槛时，除非其相对 P18-16 的 RAW PSNR 回归超过 0.03dB、SSIM 回归超过 0.0005、ΔE00 增量超过 0.1 或任一 Camera/高 ISO 桶异常，否则选择 P36-16；
4. P36-16 失败时，在 P18-16 与 P10-16 中按同一画质等价门槛选择 MAC 更低者；
5. 三个候选均失败则保持 Dense 作为只读问题定位基线，但 V6 剪枝目标判定未通过。

选择完成后立即把获胜拓扑设为唯一开发基线。其他候选的模型权重、Checkpoint、ONNX 和量化制品不得继续保留或参与后续训练；只归档不可执行的指标摘要、淘汰原因和配置 Hash。

---

## 7. 强制 FP16 + INT8 Mixed Precision LSQ+ QAT

### 7.1 不可变原则

量产发布模型必须同时包含 FP16 和 INT8 计算区域。以下方案禁止作为最终发布回退：

- PTQ-only；
- 全 FP16；
- 全 INT8/W8A8；
- 因不支持算子而回退 CPU/GPU 的伪混合精度；
- 未经目标端验证的文档型 Mixed Precision。

如果所有混合精度候选均未通过，项目状态为 No-Go，而不是退回全 INT8 发布。

### 7.2 公共精度骨架

所有量产候选至少满足：

|模块|精度|
|---|---|
|RAW Input / Normalize|FP16|
|Intro Conv|FP16|
|Encoder/Decoder Conv|INT8，LSQ+ QAT|
|DWConv / 1×1 Conv|INT8，LSQ+ QAT|
|SimpleGate|INT8 输入、INT32 Element-wise Multiply、INT8 Requant|
|Condition MLP / FiLM Head|FP16|
|Ending Conv|FP16|
|Residual Subtract / Clamp|FP16，OM 外 NPU/ISP 向量路径|
|Semantic Unpack|位精确，OM 外 NPU/ISP 向量路径|
|OM Output|FP16 `noise_pred`|

INT8 权重使用 Signed、Symmetric、Per-output-channel、Offset=0；INT8 激活使用 Signed、默认 Per-tensor。激活 Offset 是否非零先由开发环境 Q/DQ 模拟选择，结论必须标记为待商用 DDK 复核。

### 7.3 唯一 FiLM 混合精度映射

严格冻结 V5.6 指导映射，不再建立 MP-A/MP-C 分支：

|位置|精度策略|
|---|---|
|Condition MLP|FP16|
|Gamma/Beta 生成|FP16|
|Stage2 FiLM Apply|FP16|
|Stage3 FiLM Apply|INT8 Dynamic Affine|
|Middle FiLM Apply|INT8 Dynamic Affine|

Stage3/Middle Dynamic Affine 的整数语义冻结如下：

1. Gamma/Beta 分别量化为带明确 Scale 和 Zero-point 的整数 Tensor；
2. Feature 与 Gamma 的逐元素乘法以及 Beta 偏置累加均在 INT32 域完成；
3. 乘法 Scale、Bias Scale 与输出 Scale 必须可追溯，最后统一 Requant 为 INT8；
4. 整数参考实现、FakeQuant 与 Q/DQ ONNX 必须在第 9.5 节误差门槛内一致。

ONNX 参考图必须显式暴露 Gamma/Beta Q/DQ、Feature Q/DQ 和预期 Cast 边界。若部署图出现 `DQ→FP16 Mul/Add→Q`，即判定 INT8 Dynamic Affine 失败。标准 ONNX Q/DQ 只表达量化边界，不能保证 ATC 自动生成某种 Requantize 或单周期融合；当前没有商用 ATC 时只验证整数数学等价和静态图结构，并在 Manifest 中固定 `dynamic_affine_target_pending=true`。商用 DDK 可用后若不能生成无 FP16 回退且 100% 位于 NPU 的图，直接 No-Go。

### 7.4 Offset 前置微基准

在三个候选进入 Q1 前，分别从已恢复的 P36-16、P18-16 和 P10-16 提取对应连续 MobileNAFBlock，使用同一组非对称 LSQ+ 配置生成混合精度 Q/DQ 图，并与各自 FP16 Reference 做数值对比。Offset/Zero-point 结论必须全局冻结，禁止为了让某个剪枝候选胜出而使用不同 Quantizer 分支。

非零 Offset 在开发环境模拟中出现以下任一情况即不得进入完整训练：

- ONNX 导出、Q/DQ 适配或 ONNX Runtime 执行失败；
- 出现动态 Shape、非白名单算子或无法静态表达的 FiLM/Gate 路径；
- FakeQuant 画质或关键层饱和率异常；
- 参数、MAC、Cast 或 Activation 内存代理指标明显劣于对称方案；
- Q/DQ 适配后的 Scale/Zero-point 与训练策略不一致。

Offset 或 LSQ+ Q/DQ 失败时必须整改；整改后仍失败则 V6 量化 No-Go。禁止回退为普通 LSQ、全 INT8 或全 FP16。

### 7.5 校准与 QAT 调度

校准集至少 4096 帧，Main/Tele 各不少于 2048 帧，暗部占比不低于 60%。每层搜索 `99.9/99.95/99.99/100` Percentile，并在 Black Level、暗部、中灰和高光桶分别计算归一化 MSE 后等权平均。

|阶段|Step|网络权重|量化参数|默认 LR|
|---|---:|---|---|---:|
|Phase1|10k|FP16 尺寸适配|关闭 FakeQuant|1e-5|
|Q0|校准|冻结|范围搜索、MSE 初始化|无|
|Q1|2k|冻结|可学习，候选探针|1e-4|
|Q2|50k～80k|可学习|可学习|Weight 1e-5，Quant 1e-4|
|Q3|10k|可学习|冻结|5e-6|

三个剪枝候选分别执行 Phase1、Q0 和 Q1；模拟选优并删除落选制品后，仅获胜 Candidate ID 有资格执行 Q2、Q3。Q3 只切换 `requires_grad`，不得重新估计 Scale/Offset；冻结前后量化参数最大绝对漂移必须为 0。

### 7.6 SimpleGate 量化

```text
INT8 X1 ⊙ INT8 X2
→ INT32 Element-wise Multiply
→ Paired Per-tensor Scale Requant
→ INT8 Output
```

要求：

- 两个 Gate 分支使用共享或受约束的 Scale；
- INT32 累加不得溢出；
- 饱和率定义为量化前超出目标整数可表示区间的元素数除以总元素数；
- 按层、Camera 和 ISO 分桶统计最大饱和率，任一关键层或分桶达到 `0.1%` 即失败；
- 不得因 Requant 插入 CPU 节点或非预期 DDR 往返；
- “Per-Channel Scale Alignment”在 V6 中定义为 Gate 两分支的尺度对齐约束，不表示激活采用 Per-channel Quantization；实际激活量化固定为 Per-tensor。

### 7.7 Q/DQ 与 Quant Policy

导出显式 `QuantizeLinear/DequantizeLinear`。Quant Policy 至少记录：

- 每层 Dtype、Bits、Signed、Symmetric；
- Scale、Offset/Zero-point、Axis、Granularity；
- FP16 Island 边界；
- Gamma/Beta 与 Feature 的 Q/DQ、Scale/Zero-point 和 INT32/Requant 边界；
- Cast、预期 Fusion 和 NPU Placement；
- `dynamic_affine_target_pending`；
- Model、Topology、Sensor、Condition 和 Quant Policy Hash。

训练态 Observer 和可学习量化参数不得残留在最终发布图。Quant Policy 不得把“存在标准 Q/DQ”解释为“ATC 融合已保证”；Fusion 和 Placement 只能由未来商用 ATC 编译图与目标端 Profiling 证明。

---

## 8. ONNX、OM 与算子融合

### 8.1 固定 Shape ONNX

只允许导出：

```text
dark_preview_ryyb_4x3_mixed.onnx
packed_raw = 1×4×768×1024 FP16
condition  = 1×24 FP32
noise_pred = 1×4×768×1024 FP16
```

### 8.2 图审计

- 禁止动态 Shape 和动态 Split；
- Slice Starts/Ends/Axes/Steps 必须为 Constant 或 Initializer；
- Gate 两半相等；
- HardTanh 必须导出为常量边界 Clip；
- INT8 区域必须有显式 Q/DQ；
- FP16 Island 必须与选中的 Precision Map 一致；
- `condition` 公共输入保持 FP32，且全图只能有一次 FP32→FP16 Cast；
- Gamma/Beta Q/DQ、Feature Q/DQ、Scale/Zero-point、INT32 Multiply/Add 与输出 Requant 边界必须显式可审计；
- 参考图必须标出预期 Cast 边界；部署图出现 `DQ→FP16 Mul/Add→Q` 即判定 INT8 Dynamic Affine 失败；
- 标准 Q/DQ 节点本身不得作为 ATC Fusion、单周期执行或 100% NPU 的证明；
- PyTorch→ONNX FP 路径最大绝对误差≤`1e-4`；
- 未在白名单的算子直接阻断 ATC。

当前开发阶段的图审计结论只能覆盖整数数学等价、Q/DQ/Cast 静态结构与 ONNX Runtime 数值；必须保持 `dynamic_affine_target_pending=true`。未来商用 ATC 的编译后图必须再次审计实际 Fusion、Placement 与回退边界，通过后才允许置为 `false`。

### 8.3 模拟候选与未来唯一 OM

当前无商用 ATC，开发阶段不生成或伪造占位 OM。只有 `P36-16/P18-16/P10-16` 剪枝拓扑允许在临时目录比较；混合精度和 Quantizer 严格固定为第 7 章的唯一 FiLM 映射与 LSQ+。

剪枝选择结束后必须立即只保留一个拓扑。Q2/Q3 Mixed Precision LSQ+ 训练、最终 Q/DQ ONNX 和未来 OM 全部基于该唯一拓扑，不得继续携带 Q1 候选矩阵。

未来商用 DDK 可用时，只允许用该唯一开发最优方案编译：

```text
dark_preview_ryyb_4x3_mixed_int8_fp16.om
```

开发主线和未来 Release 目录都不得保留失败候选、第二个 OM 或备用模型。失败候选只保留不可执行的指标摘要和失败原因。

### 8.4 模拟 Profiling 与未来真机红线

开发模拟阶段必须静态统计或通过 Hook/Profiler 记录：

- 参数、MAC、Activation 峰值和 Workspace 代理量；
- Q/DQ、Cast、Clip、Mul、Add 和 Requant 数量；
- FakeQuant Activation Histogram、Scale、Offset 和饱和率；
- CPU/GPU/ONNX Runtime 的相对耗时，仅作为候选间代理指标；
- Main/Tele 独立画质和数值结果；
- DMA-BUF FD 导入次数、每帧 CPU memcpy 字节数、map/unmap 次数、Fence 等待和异常回收结果。

这些数据不得换算或宣称为麒麟 9000 P50/P95/P99。未来目标端还必须检查：

- Conv、DWConv、SimpleGate、Requant、FiLM 和残差路径融合；
- FP16/INT8 Cast 数量与位置；
- DDR 读写、Workspace、峰值内存；
- 每个算子的 NPU Placement；
- Activation Histogram、Scale、Offset 和饱和率；
- Main/Tele 独立 Timeline。

禁止把 NPU Kernel 单项时间替代完整 AI 节点时间。

---

## 9. 性能与验收矩阵

### 9.1 AI 节点预算

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
|NPU Mixed Precision Graph|≤6.0ms|
|Subtract/Clamp/Unpack/Output DMA|≤0.7ms|
|Queue/Fence|≤0.3ms|

计时从输入 Fence 可读开始，到输出 Fence 可供下游消费结束，包含 Queue、DMA、Cast、NPU 和同步。10ms 超时立即回到传统 ISP NR。

上述 `1.0ms` 输入预算和 `0.7ms` 输出预算保持不变。每帧额外 CPU memcpy 必须为 `0 byte`，这是硬门槛；Buffer 完成首次导入并进入池化复用后，Fence/Cache 同步耗时 `≤0.2ms` 仅作为优化目标，不替代 Queue/Fence 总预算，也不得在没有真机实测时宣称已达成。

### 9.2 阶段画质预算

|阶段|参考|RAW PSNR 最大下降|SSIM 最大下降|
|---|---|---:|---:|
|KD Student|W32 Teacher|0.20dB|0.003|
|剪枝恢复|KD Student|0.10dB|0.002|
|Mixed QAT|对应剪枝 FP32|0.10dB|0.002|
|最终 OM|W32 Teacher|0.35dB|0.005|

任一 Camera 或高 ISO 分桶相对 Teacher 下降超过 0.50dB直接失败。最终 OM 相对 FP32 的平均 ΔE00 增量≤0.3。

### 9.3 Main/Tele 独立门禁

Main 和 Tele 必须分别报告：

- RAW/RGB PSNR、SSIM、MS-SSIM；
- ΔE00 Mean/P95；
- NPS、Row/Column Noise；
- Edge MTF、暗部纹理、色斑、Banding、Halo、Ringing、False Color；
- ISO 800/1600/3200/6400/12800；
- Lux、温度、曝光和计划数字增益分桶；
- 10/30 分钟热稳态和 10,000 帧稳定性。

禁止只看 Main/Tele 平均值。

### 9.4 时序稳定性

- 静态场景重复噪声序列必须报告降噪输出的 Luma/Chroma Temporal Standard Deviation；
- 真实 Preview 序列必须进行 Flow 对齐和遮挡屏蔽；
- Flicker 不得差于传统 ISP 基线，并应相对不含 Temporal Loss 的 Student 改善至少 10%；
- 不得出现拖影、运动边缘污染或纹理呼吸；
- 若时序改善与单帧画质冲突，先满足画质硬门槛，再在合格候选中选择时序更优者。

### 9.5 数值与工程

- 四种 RYYB 物理相位 Pack/Unpack 位精确；
- Main/Tele 不同物理相位映射到统一 `[R,Yr,Yb,B]`；
- 检查 Stride、Offset、偶数 Crop 和输出 Fence；奇数 Crop、错误 RAW Domain 和错误 Hash 均拒绝；
- BLC 只执行一次，Formatter 不得二次减 Black Level，错误 LSC Profile/Hash 必须拒绝；
- `condition` 全图只有一次 FP32→FP16 Cast；
- `strength=0` 恒等；
- Teacher 无梯度，Student 真实反向；
- P36-16/P18-16/P10-16 参数、MAC 和 Shape 真实变化；
- 所有剪枝宽度满足 16 对齐；
- SimpleGate 图为逐元素乘法，INT32 中间值无溢出；
- Dynamic Affine 整数参考实现与 FakeQuant/Q/DQ 输出满足逐层误差门槛，部署图不得出现隐式 FP16 Mul/Add 回退；
- Temporal 实现使用 `z_i-N_i`，高光、运动、遮挡和无效区域均不贡献梯度；
- DMA-BUF FD 池化复用、Fence 顺序、Cache 一致性、生命周期和异常超时回收通过，额外 CPU memcpy 为 `0 byte`；
- Q3 冻结零漂移；
- PyTorch、ONNX Q/DQ、OM 进入同一逐层数值报告；
- 100% NPU，未批准 CPU/GPU 回退为 0。

---

## 10. 开发模拟选优与唯一方案收敛

### 10.1 临时剪枝候选评估

严格按 V5.6 主线冻结 Mixed Precision Mapping 和 LSQ+ Quantizer，开发阶段只比较剪枝强度：

1. 优先验证指导拓扑 P36-16；
2. 同时以 P18-16、P10-16 作为画质保护候选；
3. 三个候选使用相同训练数据、初始化来源、Optimizer、Loss、Mixed Precision Mapping 和 LSQ+ 配置，但严格使用第 5.5 节冻结的候选专属恢复周期；
4. 三个候选均执行 Phase1 FP16 全尺寸适配 10k、Q0 和 Q1 2k LSQ+ 探针；
5. 模拟选择结束立即只保留一个剪枝拓扑，落选可执行制品随即删除；
6. 只有获胜 Candidate ID 继续 Q2/Q3，且不再产生新的精度或 Quantizer 分支。

Dense、普通 LSQ、全 FP16 和全 INT8 只作为只读诊断基线，不具备 V6 最终开发方案资格。

### 10.2 模拟硬门槛

任一临时候选出现以下情况立即淘汰：

- 任一 Camera、ISO 或亮度分桶超过相应阶段画质预算；
- 出现 NaN/Inf、输出越界、异常 Banding 或关键层饱和率≥0.1%；
- 剪枝宽度不是 16 对齐，或 Gate/DWConv/Skip/FiLM 依赖不一致；
- 量化方案不是第 7 章冻结的 FP16+INT8 Mixed Precision LSQ+；
- PyTorch、FakeQuant、Q/DQ ONNX 数值不一致；
- ONNX 存在动态 Shape、动态 Split、非白名单算子或无法静态表达的 Dynamic Affine；
- Dynamic Affine 参考图未显式暴露整数语义与 Q/DQ/Cast 边界，或整数参考实现与 FakeQuant/Q/DQ 超出误差门槛；未来部署图若出现 `DQ→FP16 Mul/Add→Q` 同样直接失败；
- Buffer 契约存在额外 CPU memcpy、逐帧 map/unmap、Fence 顺序错误、Cache 不一致或 FD 生命周期泄漏；
- 参数、MAC、Activation 峰值或 Cast/QDQ 数量与设计声明不一致；
- Checkpoint、Topology 和 Safetensors 无法独立重建。

当前阶段不得把 ATC、NPU 覆盖、麒麟 9000 P95、功耗或热稳态列为“已通过”，因为没有对应设备和工具。

### 10.3 模拟最优方案的确定顺序

仅在模拟硬门槛通过的候选之间按以下规则选择：

1. 最小化 Main/Tele 和所有可用分桶中的最差 Teacher 画质回归；
2. RAW PSNR 差≤0.03dB、SSIM 差≤0.0005、ΔE00 差≤0.1 时视为画质等价；
3. 画质等价时先选择发布 Shape 下 Conv/Linear MAC 更低者；
4. MAC 相同时选择 Activation 峰值更低者，再选择 Cast/QDQ 节点更少者；
5. 静态资源代理仍相同时，使用相同机器、相同 Shape、相同线程设置的 ONNX Runtime P95 作为开发环境相对指标；
6. 相对耗时差≤3% 时选择参数量和 ONNX 体积更小者。

CPU/GPU/ONNX Runtime 耗时只用于同环境候选排序，不得换算为麒麟 9000 时延。

### 10.4 模拟验证后立即只保留最优方案

剪枝选择完成后立即执行开发收敛：

- 把获胜 Candidate ID 写入 `development_selection_v6.json`；
- 开发主线只保留获胜候选的权重、Topology、Quant Policy、Q/DQ ONNX 和 Manifest；
- 淘汰候选的权重、Checkpoint、ONNX 和其他可执行模型制品不再保留；
- 只归档淘汰候选的指标摘要、配置 Hash 和淘汰原因，用于审计而非恢复；
- 后续 Mixed Precision LSQ+ QAT、测试、文档和未来商用 DDK/真机验证只针对该唯一方案；
- 不保留隐藏开关、备用 AI 模型或第二条候选加载路径。

选择结果标记为 `development_selected=true`、`precision_policy=fixed_v5_6_mixed_lsqplus`、`dynamic_affine_target_pending=true`、`target_validated=false`、`release_ready=false`。未来真机若验证失败，必须重新进入开发流程，不能激活已淘汰候选或降级为非指导量化方案。

---

## 11. 当前实现证据与 V6 待开发项

截至 V4.0 的现有证据：

- 固定 RYYB Profile、四相位 Pack/Unpack、偶数 Crop 和 Condition 准入已实现；
- W32 Teacher、W16 Student、Feature KD、真实结构化剪枝、LSQ/LSQ+ QAT 和 Q/DQ ONNX 已完成 CPU 工程冒烟；
- V4 CPU 证据记录 Python 测试 47/47、C++ CTest 1/1；
- P10/P15 的 8 通道剪枝只证明工程 Shape 变化；
- 没有商用 ATC、麒麟 9000、真实 RYYB 完整训练和量产性能证据。

V6 必须新增或修改：

1. HAL Post-BLC/LSC RAW Domain 状态、LSC Hash 与二次减 Black Level 拒绝机制；
2. 四种物理 RYYB 相位 `2D RAW→Pack→Semantic Unpack→2D RAW` 位精确联合测试；
3. OM 外 NPU/ISP 向量 Subtract/Clamp/Unpack 路径与 Stride/Crop/Fence 契约；
4. DMA-BUF 池化共享、Cache 一致性、Fence、生命周期和 `0 byte` CPU memcpy；
5. HardTanh/Clip FiLM、单次 Condition Cast 和 Dynamic Affine 整数参考图；
6. Sensor 条件化 RYYB 相关噪声；
7. 完整 Student/Attention KD 与正确的 Denoised Output Temporal Loss；
8. `round_to=16` 的 P36-16/P18-16/P10-16 剪枝与拓扑重建；
9. 三候选 Phase1/Q0/Q1 探针、模拟选优、淘汰清理和唯一开发清单；
10. 只针对开发最优方案执行 Q2/Q3，并在未来完成 OM 编译、目标设备和完整 Go/No-Go 验收。

V6 新增项完成前不得把 V4 的 CPU 冒烟结果解释为 V6 已实现。

---

## 12. 发布 Go/No-Go

全部满足才允许把 `release_ready` 改为 `true`：

1. Main/Tele 真实 RYYB 数据量、标定、无泄漏和 Blind Test 通过；
2. Linear RAW、BLC/LSC、物理 CFA、Pack/Unpack、Condition 和所有 Hash 契约通过；
3. Teacher、Student、KD、Temporal、剪枝恢复和 Mixed Precision QAT 完整收敛；
4. 唯一开发最优方案的 16 通道对齐、OM 编译和目标端性能通过；
5. 最终候选同时包含 FP16 和 INT8，Dynamic Affine 无隐式 FP16 回退，且 100% 位于 NPU；
6. Main/Tele 独立画质、时序和主观盲测通过；
7. Q/DQ→CANN 适配→ATC→OM 逐层参数一致；
8. `dynamic_affine_target_pending=false`，实际 Fusion、Placement 和 Cast 边界审计通过；
9. AI 节点 6/8/9/10ms 和整条 Photo Preview 30fps 通过；
10. DMA-BUF 共享契约、额外 CPU memcpy `0 byte`、内存、功耗、热稳态、10,000 帧和 Camera 切换通过；
11. 开发模拟选优报告已完成，失败候选早已退出开发主线和 Release 路径；
12. 最终发布包只有一个固定 Shape Mixed Precision OM；
13. 阻断级和高优先级缺陷为 0；
14. 回滚到传统 ISP NR 验证通过。

任何一项缺失均保持 `release_ready=false`。

---

## 13. 最终发布制品

模拟选优结束后的开发主线只允许保留：

```text
model_mixed_qat.safetensors
topology_v6.json
condition_schema_v2.json
sensor_profiles_ryyb.json
unpack_profiles_ryyb.json
raw_domain_profile.json
lsc_profiles_ryyb.json
reference_isp_profile.json
noise_profiles_ryyb.json
buffer_contract_v1.json
quant_policy_v6_mixed.json
dark_preview_ryyb_4x3_mixed.onnx
model_manifest_v6.json
development_selection_v6.json
selection_report_v6.md
```

其中 `development_selection_v6.json` 必须记录唯一 `selected_candidate_id`、剪枝候选结果、固定 Mixed Precision LSQ+ Policy、模拟环境、指标 Hash、`development_selected=true` 和 `target_validated=false`。

未来目标验证通过后的 Release 目录只允许：

```text
model_mixed_qat.safetensors
topology_v6.json
condition_schema_v2.json
sensor_profiles_ryyb.json
unpack_profiles_ryyb.json
raw_domain_profile.json
lsc_profiles_ryyb.json
reference_isp_profile.json
noise_profiles_ryyb.json
buffer_contract_v1.json
quant_policy_v6_mixed.json
dark_preview_ryyb_4x3_mixed.onnx
dark_preview_ryyb_4x3_mixed_int8_fp16.om
model_manifest_v6.json
development_selection_v6.json
selection_report_v6.md
```

`model_manifest_v6.json` 必须记录唯一 `selected_candidate_id`、全部 Hash、精度分区、16 通道拓扑、目标 SoC/DDK 版本，以及至少以下字段：

```json
{
  "unpack_profile_hash": "sha256:...",
  "lsc_profile_hash": "sha256:...",
  "buffer_contract_version": "v1",
  "development_selected": true,
  "dynamic_affine_target_pending": true,
  "target_validated": false,
  "release_ready": false
}
```

开发模拟阶段上述四个布尔状态按示例冻结；未来只有 Dynamic Affine 商用 ATC 编译、无 FP16 回退和目标端 Placement 全部通过后，才允许把 `dynamic_affine_target_pending` 改为 `false`。`target_validated` 与 `release_ready` 仍必须分别满足第 12 章全部目标端验证和发布门槛。

淘汰候选不得保留 Checkpoint、ONNX、OM 或可恢复权重。候选比较的不可执行指标摘要可以保存在 Selection Report 中；开发主线和量产安装包都不得包含淘汰模型。

---

## 14. 参考资料

1. [NAFNet](https://arxiv.org/abs/2204.04676)：NAFNet 与 SimpleGate 图像恢复基础。
2. [Torch-Pruning](https://github.com/VainF/Torch-Pruning)：DependencyGraph 结构化剪枝。
3. [LSQ](https://arxiv.org/abs/1902.08153)：可学习量化步长与梯度缩放。
4. [LSQ+](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Bhalgat_LSQ_Improving_Low-Bit_Quantization_Through_Learnable_Offsets_and_Better_Initialization_CVPRW_2020_paper.html)：可学习 Offset 与 MSE 初始化。
5. [CANN QAT 模型适配](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/amct/atlasamct_16_0148.html)：Q/DQ 适配、权重 Per-channel 与算子限制。
6. [AMCT 量化说明](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/amct/atlasamct_16_0168.html)：量化流程与框架边界参考。
7. [CANN ONNX Clip 算子](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/operatorlist/atlasoxol_09_0034.html)：HardTanh 的 Clip 导出边界。
8. [Android DMA-BUF Heaps](https://source.android.com/docs/core/architecture/kernel/dma-buf-heaps)：Android 12 后 DMA-BUF Heap 与遗留 ION 迁移边界。
9. [Linux DMA-BUF Heap Userspace API](https://cdn.kernel.org/doc/html/latest/userspace-api/dma-buf-heaps.html)：System/CMA Heap 与 Buffer 分配语义。
10. [Linux DMA-BUF Sharing Framework](https://docs.kernel.org/driver-api/dma-buf.html)：共享 Buffer、Fence、CPU Access 与同步契约。
11. [Rethinking Noise Synthesis and Modeling in Raw Denoising](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Rethinking_Noise_Synthesis_and_Modeling_in_Raw_Denoising_ICCV_2021_paper.html)：真实 RAW 空间相关噪声与噪声合成边界。
12. [Learning Temporal Consistency for Low Light Video Enhancement](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Learning_Temporal_Consistency_for_Low_Light_Video_Enhancement_From_Single_CVPR_2021_paper.html)：低光处理的时序稳定性训练参考。

---

## 15. 结论

V6.0 严格以 V5.6 为优化主线，并修正其内部数值、物理与接口矛盾：Main/Tele 可以具有不同物理 RYYB 相位，但模型输入统一为 `[R,Yr,Yb,B]`，模型重建结果必须按 Sensor Profile 位精确 Unpack 回二维物理 CFA；Post-BLC/LSC 域禁止二次减 Black Level；Camera HAL 与 AI Runtime 采用带 Fence/Cache/生命周期约束的 DMA-BUF 共享，禁止额外 CPU memcpy。剪枝严格使用 16 通道对齐，优先验证指导拓扑 `[16,32,48,96]`，并按真实压缩率命名 P36-16；最终精度固定为 V5.6 指定的 FP16+INT8 Mixed Precision LSQ+，禁止用普通 LSQ、全 INT8 或全 FP16 代替。

只有 P36-16/P18-16/P10-16 剪枝强度允许短暂存在于临时评测区。三个候选分别完成 FP16 全尺寸适配、Q0 与 Q1 LSQ+ 探针后，必须立即淘汰其他模型制品，只保留一个综合最优开发方案继续 Q2/Q3；后续文档和未来真机验证全部基于该方案。由于当前没有商用 DDK 和麒麟 9000，模拟选优只能证明开发候选间的相对最优，不能证明 Dynamic Affine 已融合、Fence 同步达到 0.2ms、8ms P95、100% NPU、功耗或热稳态已经通过，`dynamic_affine_target_pending=true`、`target_validated=false`、`release_ready=false` 必须继续保持。
