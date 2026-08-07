# AI-ISP：麒麟 9000 暗光预览 AI RAW Denoise V6.1

本仓库实现面向麒麟 9000 的 RYYB 主摄/长焦共享 AI RAW 降噪工程。V6.1 已完成数据准入、Teacher/Student、KD 与时序训练、三个 16 通道剪枝候选、固定 LSQ+ 混合精度 QAT、唯一方案选优、Q/DQ ONNX、DMA-BUF Runtime 契约和 Manifest 的 CPU 工程闭环。

本次模拟选择 `P36-16=[16,32,48,96]`。当前状态固定为：

```text
development_selected=true
dynamic_affine_target_pending=true
target_validated=false
release_ready=false
```

## 冻结接口

|项目|定义|
|---|---|
|AI Camera|RYYB `main`、RYYB `tele`|
|Bypass Camera|RGGB `ultrawide`，传统 ISP NR|
|RAW Domain|Linear Post-BLC/LSC、Pre-Digital-Gain|
|模型输入|`packed_raw[1,4,768,1024] FP16`、`condition[1,24] FP32`|
|语义通道|`[R,Yr,Yb,B]`|
|模型输出|`noise_pred[1,4,768,1024] FP16`|
|后处理|NPU/ISP Vector `Subtract/Clamp → Semantic Unpack`|
|剪枝候选|P10-16、P18-16、P36-16；模拟后只保留 P36-16|
|量化|固定 FP16+INT8 Mixed Precision LSQ+|
|AI 节点预算|P50≤6ms、P95≤8ms、P99≤9ms、10ms硬超时（待真机）|

Unpack 必须根据 Sensor Profile 恢复二维物理 RYYB CFA，禁止按 Camera 名猜相位。Formatter 禁止二次减 Black Level。生产路径禁止逐像素 CPU 后处理、额外 CPU memcpy 和 CPU/GPU AI 回退。

## 当前状态

- 三个候选均完成剪枝恢复、Phase1、Q0/Q1 探针；P36-16 按冻结顺序获胜并继续 Q2/Q3。
- Q0 使用 FP32 Observer、跨帧 Reservoir 和 Main/Tele 平衡校准；Q1 后逐层饱和率门槛保持 `<0.1%`。
- SimpleGate 使用逐元素乘法，Dynamic Affine 完成 INT8/INT32 数学参考审计。
- 混合 ONNX 公共输入/输出为 FP16，Condition 保持 FP32且图内只 Cast 一次。
- Python 全量测试和 V6.1 CPU 全阶段流水线通过。
- C++17 Runtime 已用 MSVC 19.42 编译运行，覆盖四种 RYYB 相位、Stride、DMA-BUF/Fence 和失败闭锁。
- 本机无商用 ATC/DDK 和麒麟 9000，故不生成伪 OM，不宣称目标性能。

## 快速开始

```powershell
# 全部 Python 测试
.\.venv\Scripts\python.exe -m pytest -q

# V6.1 无 GPU 完整阶段冒烟
.\.venv\Scripts\python.exe -m ai_isp.pipeline_v6 `
  --config configs\train\v6_1_cpu_全流程.yaml

# 真实 RYYB 量产配置；需先准备 Manifest、GPU、目标 DDK 和设备
.\.venv\Scripts\python.exe -m ai_isp.pipeline_v6 `
  --config configs\train\v6_1_量产训练.yaml
```

## 主要入口

- [V6.1 详细开发设计](./AI%20ISP%20暗光拍照预览%20RAW%20降噪详细开发设计%20V6.1.md)
- [V6.1 项目开发指导与述职学习文件](./AI%20ISP%20V6.1%20项目开发指导与述职学习文件.md)
- [V6.0 设计基线](./AI%20ISP%20暗光拍照预览%20RAW%20降噪详细开发设计%20V6.0.md)
- [V6.1 CPU 全阶段验证报告](./docs/V6.1%20CPU全阶段验证报告.md)
- [SIDD 数据集数据卡](./docs/SIDD数据集数据卡.md)
本地工程制品位于 `artifacts/v6_1_cpu_smoke/`。大体积 ONNX、Safetensors 与 Checkpoint 默认不提交普通 Git；代码、配置、测试、Markdown、JSON 报告和 Hash Manifest 同步到代码仓。文档中的目标端画质、性能、功耗和内存数值均是准入门槛，不是当前电脑或麒麟 9000 的已验证结论。
