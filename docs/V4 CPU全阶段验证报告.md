# AI ISP V4 CPU 全阶段验证报告

> 验证日期：2026-08-05
> 环境：Windows、CPU-only PyTorch，本机无可用 GPU、无商用 CANN/ATC、无麒麟 9000 目标机
> 数据：`SIDD_Training_Subset` 少量 Patch，强制 `smoke_only=true`

## 1. 验证结论

V4 算法与模型压缩工程链已经在本机连续执行完成，流水线返回 `engineering_passed`。本结果证明阶段代码可运行、真实参数 Shape 会被剪枝、四套 QAT 候选可训练、Q/DQ ONNX 可对齐，并证明缺少 ATC 时发布门禁会失败闭锁。

本结果不证明训练收敛、RYYB画质、OM可编译、麒麟9000性能或4K30。`release_ready=false` 保持不变。

## 2. 执行配置

|项目|CPU冒烟值|量产配置值|
|---|---:|---:|
|Patch|32|384，后期可升512|
|Batch|1|1|
|Teacher/Student/KD Step|1/1/1|500k/400k/200k|
|P10/P15恢复 Step|1/1|80k/120k|
|Q0校准帧|1|4096，main/tele各≥2048|
|Q1/Q2/Q3 Step|1/1/1|2k/48k/10k|
|设备|CPU|Auto，量产预期GPU训练|

执行命令：

```powershell
.\.venv\Scripts\python.exe -m ai_isp.pipeline `
  --config configs\train\v4_cpu_全流程.yaml
```

最近一次流水线墙钟时间约38.5秒。该时间只对应32×32 Patch和各阶段1 Step，不是模型训练工期或目标端推理时延。

## 3. 阶段结果

```text
双Block对称/非对称Offset前置微基准
→ Teacher监督训练 → Student监督训练 → Feature KD
→ P10剪枝/恢复 → P15剪枝/恢复
→ P10 LSQ → P10 LSQ+ → P15 LSQ → P15 LSQ+
→ 工程默认P10+对称LSQ → Q/DQ ONNX → ATC/OM失败闭锁
```

### 3.1 结构化剪枝

|候选|参数量|参数下降|Smoke Shape MAC下降|Feature Width|
|---|---:|---:|---:|---|
|Dense|660,836|—|—|`[16,32,64,128]`|
|P10|599,812|9.234%|8.070%|`[16,32,56,128]`|
|P15|565,420|14.439%|9.600%|`[16,32,56,120]`|

MAC 数值来自测试用小 Shape，只用于比较同一拓扑前后变化；固定发布 Shape 的绝对计算量和目标端收益必须另测。

### 3.2 QAT

- 前置微基准先生成对称/非对称双Block显式Q/DQ图；本机缺少目标结果，因此正式策略失败闭锁为Offset=0；

- P10/P15 都完成对称LSQ与非对称LSQ+的Q0/Q1/Q2/Q3；
- Q3只冻结量化参数，冻结前后最大绝对漂移为0；
- 权重使用signed symmetric per-output-channel，激活使用signed per-tensor；
- 没有目标DDK Offset微基准时，工程Manifest选择P10+对称LSQ，非对称LSQ+不被伪装成发布结论。

### 3.3 ONNX与OM

- ONNX包含显式`QuantizeLinear`和`DequantizeLinear`；
- PyTorch与ONNX Runtime最大绝对误差为0；
- 图中不含动态Split/Chunk，Slice参数通过常量审计；
- 本机找不到ATC，编译报告返回`available=false`且没有生成占位OM。

## 4. 自动化测试

完整测试覆盖：RYYB四相位与准入、Condition、数据防泄漏、Teacher/Student/KD反向、真实结构化剪枝、拓扑冻结、LSQ/LSQ+阶段、Q3零漂移、Q/DQ ONNX、单Profile和发布Manifest。

最终结果：`47 passed in 48.19s`。

若代码继续修改，必须重新运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 5. 当前环境未完成项

|项目|状态|原因/进入条件|
|---|---|---|
|真实RYYB完整训练|未执行|目标主摄/长焦Manifest与GPU训练资源未提供|
|C++ V4重新编译|通过|Visual Studio 2022 MSVC 19.42/C++17重新编译，CTest `1/1`通过|
|ATC/OM|阻断|缺少目标商用DDK、SOC版本和Q/DQ适配环境|
|麒麟9000 NPU|阻断|缺少目标设备与部署权限|
|AI 6/8/9/10ms|未验证|必须统计目标机完整AI节点Timeline|
|整条4K30|未验证|必须在真实Camera Pipeline联调|
|画质/功耗/热稳态|未验证|需要真实RYYB盲测、10000帧和10/30分钟稳态测试|

## 6. Go/No-Go

当前结论：`Engineering Pass / Release No-Go`。

只有真实RYYB数据、完整收敛训练、商用DDK编译、100% NPU、画质门槛、AI P95≤8ms、整条4K30及稳定性全部通过后，才能生成唯一的`dark_preview_ryyb_4x3_int8.om`并把`release_ready`改为`true`。
