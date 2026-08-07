# AI ISP 暗光拍照预览 RAW 降噪详细开发设计 V6.1

> 文档状态：工程实现基线（2026-08-07）
> 指导基线：《Kirin9000 Dark Photo Preview AI RAW Denoise V5.6 量产研发设计规范》
> 开发选择：`P36-16 + fixed_v6_mixed_lsqplus`
> 状态位：`development_selected=true`、`dynamic_affine_target_pending=true`、`target_validated=false`、`release_ready=false`

## 0. 本版结论

V6.1 已把 V6.0 的计划落实为可运行代码、配置、测试、C++ Runtime 契约和工程制品流水线。开发机使用 SIDD 小样本完成全部阶段的模拟执行，三个 16 通道对齐候选均通过开发硬门槛，最终选择 `P36-16=[16,32,48,96]`。落选候选的可执行制品在选优后立即删除，只有 P36-16 继续 Q2/Q3。

当前结果只能证明工程流程闭环和候选间的模拟相对最优。由于缺少真实 Main/Tele RYYB 量产数据、商用 ATC/DDK 和麒麟 9000 真机，本版不宣称 OM 可用、Dynamic Affine 已融合、100% NPU、0.2ms 同步、8ms P95、30fps、功耗或热稳定性达标。

本轮对 V6.0 的实现性优化如下：

|问题|验证结论|V6.1 最终处理|
|---|---|---|
|Q0 观察时执行临时 FakeQuant，导致下游范围依赖初始化顺序|逐层探针确认会产生伪饱和|Observer 模式只透传 FP32 并收集统计，校准结束后才启用 FakeQuant。|
|校准只保存前 65k 元素，后续 Camera/ISO 帧不可见|Main/Tele 平衡探针复现|改为按帧均匀抽样和确定性 Reservoir，最多 64 个观察块。|
|LSQ+ Offset 与 log-scale 共用绝对学习率|窄范围层单步发生明显网格漂移|Offset 学习率按当前量化 Scale 缩放。|
|固定全局余量不能兼顾分辨率和饱和率|10% 初始余量后个别层仍接近门槛|Q1 后仅对失败层每次扩张 5%，反复验证，门槛仍固定为 `<0.1%`。|
|FP16 模型内 FakeQuant 可能提升公共输出类型|ONNX 审计发现 `noise_pred` 不是 FP16|增加显式公共 FP16 输出边界；Condition 仍只有一次 FP32→FP16 Cast。|

## 1. 范围与系统边界

### 1.1 负责范围

- RYYB 广角主摄 `main` 与 RYYB 潜望长焦 `tele` 共用一个条件式模型；
- 输入域为 Linear Post-BLC/LSC、Pre-Digital-Gain RAW；
- 四语义通道固定为 `[R,Yr,Yb,B]`；
- 固定输入 `packed_raw[1,4,768,1024] FP16` 与 `condition[1,24] FP32`；
- 输出 `noise_pred[1,4,768,1024] FP16`；
- 模型外完成 FP16 Subtract/Clamp 与 Semantic Unpack；
- 数据准入、训练、蒸馏、剪枝、QAT、Q/DQ ONNX、候选选优、Manifest 和 HAL/Runtime 契约。

### 1.2 不负责范围

- RGGB 超广角进入传统 ISP NR，不进入本模型；
- Demosaic、AWB、CCM、Tone Mapping 等后级 ISP 不属于模型；
- 没有目标 DDK 时不生成伪 OM，不用 CPU/GPU AI 冒充 NPU；
- SIDD 只用于工程冒烟，不作为 RYYB 画质或量产证据。

## 2. 固定 Pipeline、RAW 域和接口

量产数据路径冻结为：

```text
Sensor RYYB RAW
→ BLC
→ LSC
→ RAW Domain/LSC Hash 准入
→ 按 Sensor Profile Pack 为 [R,Yr,Yb,B]
→ Post-BLC/LSC Normalize
→ Mixed Precision AI 预测 noise_pred
→ FP16 Subtract/Clamp
→ Semantic Unpack（Pack 的位精确逆操作）
→ 2D Physical RYYB CFA RAW
→ Digital Gain
→ Demosaic / 后级 ISP
```

归一化唯一公式：

$$x=\operatorname{clip}\left(\frac{raw_{post\_lsc}}{white-black},0,1\right)$$

Formatter 不得二次减 Black Level。准入描述必须同时携带 `raw_domain_state=LINEAR_POST_BLC_LSC_PRE_DGAIN`、`blc_applied=true`、`lsc_applied=true`、`raw_domain_profile_hash` 与 `lsc_profile_hash`。

Main/Tele 的物理相位由 Sensor Profile 注册，可以不同；模型语义始终相同。支持 `ryyb/byyr/yryb/ybyr` 四种物理相位。禁止按 Camera 名猜相位，禁止固定输出 `ryyb`。Crop 的起点和宽高必须为非负偶数。

OM 只输出 `noise_pred`。Subtract/Clamp 和 Unpack 位于 OM 外的 NPU/ISP Vector 后处理接口，生产路径禁止逐像素 CPU 实现。C++ 中的 `ReferencePackRyyb/ReferenceUnpackRyyb` 只用于位精确验证；量产接口为 `PostProcessExecutor`，未接 DDK 时明确返回 `kNpuUnavailable`。

## 3. DMA-BUF 共享契约

公共接口使用 DMA-BUF，不强制 ION，也不默认要求物理连续。允许 IOMMU/scatter-gather；只有平台驱动明确要求时才使用 CMA Heap。

硬约束：

- Stream 初始化时导入 FD 一次，池化复用；同一 Buffer Index 禁止逐帧更换 FD 或尺寸；
- 每帧只传 Buffer Index、FD、Offset、Stride、Valid Region 和 Producer Fence；
- 每帧额外 CPU `memcpy=0 Byte`、`map/unmap=0`；
- Producer/Consumer Fence 必须单调递增并按顺序等待；
- 明确 Producer Cache Clean、Consumer Cache Invalidate、生命周期和超时回收；
- I/O 预算仍为输入 1.0ms、输出 0.7ms；首次导入后的 Fence/Cache 同步 `≤0.2ms` 仅是目标。

Python `DmaBufPoolContract` 与 C++ `DmaBufPoolContract` 均实现了相同状态机：`IDLE → NPU_IN_FLIGHT → CONSUMER_READY → IDLE`，并覆盖 Busy、Fence 乱序、FD 替换、额外拷贝和超时恢复。

## 4. 数据、增强与噪声模型

量产 Manifest 只接受 `main/tele`、合法 RYYB 相位、Post-BLC/LSC RAW 和有效 LSC Hash。每颗 Camera 的独立场景下限保持：训练 3000、验证 300、盲测 500。`smoke_only=true` 禁止进入量产集合。

增强规则：

- 允许与物理相位一致的偶数 Crop、水平/垂直翻转和 180° Rotation；
- 90°/270° Rotation 绝对禁止；
- 训练 Patch 为 Packed RAW，量产 Phase1 固定 `[768,1024]` 全尺寸；
- Main/Tele 校准必须等量，Q0 不少于 4096 帧，Batch=1。

相关噪声冻结为：

$$\Sigma(x,c)=A(c)\operatorname{diag}(a(c)\odot\max(x,0))A(c)^T+\Sigma_{read}(c)$$

实现会检查系数非负、协方差对称及半正定，并分别采样 Shot/Read 相关项。SIDD 的 RGGB Pack 只能检查 Shape、梯度、导出和状态机。

## 5. 模型与条件化

Teacher 为 W32 条件式 NAFNet，Dense Student 为 W16 MobileNAFNet。主干所有结构通道按 16 对齐。Condition 是一个公共 FP32 输入，图内只允许 Condition Encoder 入口一次 FP32→FP16 Cast。

FiLM 采用 HardTanh/Clip，不使用 Tanh：

$$\gamma=1+0.1\operatorname{HardTanh}(g),\quad
\beta=0.1\operatorname{HardTanh}(b),\quad
y=\gamma\odot x+\beta$$

精度映射唯一固定：Stage1/Stage2 为 FP16 Affine；Stage3/Middle 为 INT8 Dynamic Affine 目标；Intro、Ending、Condition Encoder 和公共输入输出边界为 FP16。不存在 LSQ/LSQ+、全 INT8、全 FP16 或 MP-A/MP-C 的发布分支。

## 6. 损失与训练阶段

Student 总损失冻结为：

$$L=L_{RAW}+0.5L_{Tone}+0.1L_{Gradient}+0.1L_{FeatureKD}
+0.05L_{AttentionKD}+0.05L_{Temporal}+0.01L_{Gate}$$

Attention KD 使用 Stage3/Middle 特征的通道绝对值均值空间图，归一化后计算 L1。

Synthetic Temporal 定义：

$$z_i=\operatorname{clip}(x+n_i,0,1),\quad N_i=f_\theta(z_i,c),\quad
y_i=\operatorname{clip}(z_i-N_i,0,1)$$

$$L_{temp}=\frac{\|M\odot(y_1-y_2)\|_1}{\sum M},\quad
M=M_{valid}M_{motion}M_{sat}$$

$$M_{sat}=\mathbf1_{x<0.98}\mathbf1_{z_1<0.98}\mathbf1_{z_2<0.98}$$

真实连续帧额外叠加 Flow 一致性和遮挡 Mask。完整阶段顺序：Teacher 监督 → Dense Student 监督 → Feature/Attention KD + Temporal → 三候选剪枝恢复 → 全尺寸 Phase1 → Q0/Q1 探针 → 唯一获胜者 Q2/Q3。

## 7. 16 通道剪枝与唯一候选

只允许以下候选：

|候选|通道|恢复默认/最多|发布 Shape Conv/Linear MAC（本次模拟）|参数（本次模拟）|
|---|---|---|---:|---:|
|P10-16|`[16,32,64,112]`|80k/120k|34,438,706,688|591,284|
|P18-16|`[16,32,48,128]`|120k/180k|30,057,494,016|543,396|
|P36-16|`[16,32,48,96]`|180k/240k|28,137,018,880|419,652|

MAC 口径固定为发布 Shape 下 Conv/Linear MAC，不计 Add、Mul、Resize 和 Clip。剪枝必须通过 Torch-Pruning 真实依赖传播、前向、反向、保存/加载和拓扑重建验证。

每个候选完成恢复、Phase1、Q0 和 Q1 后先过硬门槛，再按最差分桶画质、画质等价、MAC、Activation、Cast+Q/DQ、ORT P95 代理、参数与 ONNX 体积确定唯一方案。开发模拟结果三者均过门槛，P36-16 在画质等价条件下 MAC 和参数最小，因此获胜。P10-16/P18-16 可执行目录已删除，只保留不可执行指标摘要。

## 8. LSQ+ 混合精度量化

INT8 区域固定为权重 Per-channel、激活 Per-tensor 的 W8A8 LSQ+。Q0 Observer 在 FP32 图上透传收集，不执行临时 FakeQuant；采样使用跨帧 Reservoir，防止只覆盖最前面的 Camera/ISO。范围保留 10% 初始对称余量；Q1 后执行 Main/Tele 平衡探针，只对达到 0.1% 的层逐次扩张 5%，每次重新验证。任何层最终 `saturation_rate>=0.001` 均失败。

SimpleGate 固定为：

```text
INT8 X1 ⊙ INT8 X2
→ INT32 Element-wise Multiply
→ Paired Per-tensor Requant
→ INT8 Output
```

Dynamic Affine 整数参考定义 Gamma/Beta 的 Scale/Zero-point、INT8 Feature×Gamma、INT32 Bias 累加和统一 Requant。ONNX 参考图显式包含 Feature/Gamma/Beta Q/DQ。若目标部署图出现 `DQ→FP16 Mul/Add→Q`，直接 No-Go。标准 Q/DQ 不能保证 ATC 融合，因此当前保持 `dynamic_affine_target_pending=true`。

## 9. ONNX、OM 与图审计

只导出一个固定 Shape ONNX，opset 19，公共边界为：

- `packed_raw`: FP16 `[1,4,768,1024]`；
- `condition`: FP32 `[1,24]`；
- Condition 直接 Cast 数量：1；
- `noise_pred`: FP16 `[1,4,768,1024]`。

图审计拒绝动态 Gate/Slice、非白名单算子和公共类型漂移。FP32 Q/DQ Reference 使用 ONNX Runtime 做数值一致性；CPU ORT 不作为 FP16 Q/DQ 部署图的数值证据。没有 ATC 时 `compile_single_om` 生成失败闭锁报告，禁止伪造 OM。

## 10. 工程实现映射

|能力|主要实现|
|---|---|
|RAW Pack/Unpack、准入与归一化|`ai_isp/data/ryyb_contract.py`、`pack_raw.py`|
|RYYB 数据与噪声|`ryyb_dataset.py`、`noise_model.py`|
|模型、FiLM、SimpleGate|`ai_isp/models/mobile_nafnet.py`、`static_simple_gate.py`|
|完整损失与阶段训练|`dark_preview_losses.py`、`training_stages.py`|
|16 对齐剪枝和 MAC|`nafnet_pruning_validator.py`|
|LSQ+、Q0/Q1/Q2/Q3|`lsqplus_qat.py`、`qat_training.py`|
|Dynamic Affine 整数审计|`quantization/dynamic_affine.py`|
|ONNX/OM/Manifest|`export/static_profiles.py`、`export/om_release.py`|
|唯一候选选优与清理|`selection.py`|
|DMA-BUF 契约|`runtime/dmabuf_contract.py`、`runtime/src/dmabuf_pool.cpp`|
|完整入口|`ai_isp/pipeline_v6.py`、`tools/运行V6.1全流程.py`|

## 11. 本次开发验证证据

- Python 全量测试：`66` 项设计目标；最终归档以最新测试报告为准；
- C++17 Runtime：MSVC 19.42 编译并运行通过，覆盖四物理相位、非紧密 Stride、DMA-BUF 生命周期和失败闭锁；
- CPU 全流程：Teacher、Student、KD/Temporal/Attention、三个候选、恢复、Phase1、Q0/Q1、选优、唯一 Q2/Q3、Dynamic Affine、ONNX、OM 门禁、Manifest 全部实际执行；
- 模拟饱和率：P10-16 `0.09765625%`、P18-16 `0.09765625%`、P36-16 `0.091552734375%`；
- 选中候选：P36-16；
- 每帧额外 CPU memcpy 模拟审计：0 Byte；
- OM：不可用，原因是本机无目标商用 DDK/ATC。

工程制品位于 `artifacts/v6_1_cpu_smoke/release_v6_1/`。大体积 ONNX、Safetensors、Checkpoint 默认只在本地归档，不进入普通 Git；JSON、Markdown、配置、代码和 Hash Manifest 进入代码仓。

## 12. Go/No-Go

开发机允许设置 `development_selected=true`，但只有下列证据全部满足后才能把其余状态改为真：

1. 真实 Main/Tele RYYB 数据规模、盲测和分桶画质通过；
2. 商用 ATC 生成唯一 OM，Dynamic Affine 无 FP16 回退；
3. 目标端 NPU 覆盖 100%，CPU/GPU AI 回退为 0；
4. AI 节点 P50≤6ms、P95≤8ms、P99≤9ms、10ms 硬超时；
5. Photo Preview 整链稳定 30fps；
6. DMA-BUF/Fence/Cache 和 0 Byte CPU memcpy 实测通过；
7. 10000 帧、10/30 分钟热稳态、功耗、内存和 Camera 切换通过；
8. 回滚路径验证通过。

任一项缺失时：`dynamic_affine_target_pending=true`、`target_validated=false`、`release_ready=false`。

## 13. 唯一制品

最终开发方案必须同时保留：

- `model_mixed_qat.safetensors`；
- `dark_preview_ryyb_4x3_mixed.onnx`；
- Topology、Quant Policy、Q/DQ 审计；
- Condition Schema、Sensor/Unpack/RAW Domain/LSC/Reference ISP/Noise Profile；
- Buffer Contract、Selection Report、Development Selection；
- `model_manifest_v6_1.json`；
- 目标工具链可用后生成的唯一 `dark_preview_ryyb_4x3_mixed_int8_fp16.om`。

Manifest 必须包含 `unpack_profile_hash`、`lsc_profile_hash`、`buffer_contract_version` 和 `dynamic_affine_target_pending`。淘汰候选只能保留指标、Hash 和淘汰原因，不得保留可执行权重、ONNX 或 OM。

## 14. 结论

V6.1 已完成从文档方案到工程代码的闭环，并通过模拟验证选择 P36-16 作为唯一后续方案。主要风险被显式关闭在目标端门禁，而不是被 CPU 冒烟结果掩盖。后续开发不得重新引入非 16 对齐拓扑、量化策略分支、Camera 名猜相位、二次 BLC、逐像素 CPU 后处理、额外 CPU memcpy 或伪 OM。
