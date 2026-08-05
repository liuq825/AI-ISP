# AI-ISP：麒麟 9000 暗光预览 AI RAW Denoise

本仓库实现面向麒麟 9000 手机的条件式 MobileNAFNet RAW 域降噪算法与模型压缩工程。V4.0 已冻结为：仅 RYYB 广角主摄和 RYYB 潜望式长焦共用一个固定 Shape 模型；RGGB 超广角固定走传统 ISP NR；最终发布包只允许一个 OM。

当前算法工程链已经在无 GPU 电脑上用少量 SIDD 数据完整跑通：

```text
数据校验 → W32 Teacher → W16 Student → KD
→ Torch-Pruning P10/P15 → 分别恢复
→ P10/P15 × 对称 LSQ/非对称 LSQ+ 四套 QAT
→ 显式 Q/DQ ONNX → 单 OM 编译门禁 → Manifest
```

## 冻结接口

|项目|定义|
|---|---|
|AI Camera|RYYB `main`、RYYB `tele`|
|Bypass Camera|RGGB `ultrawide`，传统 ISP NR|
|模型输入|`packed_raw[1,4,768,1024]`、`condition[1,24]`|
|原始 RAW|`2048×1536`、4:3、通道 `[R,Yr,Yb,B]`|
|模型输出|同格点 `noise_pred`|
|后处理|`clamp(raw_in-noise_pred,0,1)`|
|最终制品|`dark_preview_ryyb_4x3_int8.om`|
|AI 节点预算|P50≤6ms、P95≤8ms、P99≤9ms、10ms硬超时|

Crop 的 `x/y/width/height` 必须全部为偶数。Camera、CFA、Shape、Stride、位深、Black/White Level 或各版本 Hash 任一不满足时，必须 Bypass AI，禁止 Resize、补行、运行时通道猜测和 CPU AI 回退。

## 当前状态

- 已完成真实结构化剪枝、拓扑冻结与可重建校验；CPU 冒烟 P10 参数下降 9.23%，P15 参数下降 14.44%。
- 已实现权重 Per-channel、激活 Per-tensor 的 LSQ/LSQ+ W8A8 QAT，包含 Q0/Q1/Q2/Q3、流式校准、梯度缩放、Q/DQ 导出及逐层审计。
- 已实现 Teacher `eval()+no_grad()`、Student AMP/梯度累积、Checkpoint/断点续训和 Stage3/Middle Feature KD。
- 已通过 Python 测试和 CPU 全阶段流水线；SIDD 结果仅代表工程冒烟。
- C++ Runtime 已由 MSVC 19.42/C++17 重新编译，CTest `1/1` 通过。
- 本机没有商用 CANN/ATC 和麒麟 9000 真机，真实 RYYB 完整训练、OM、100% NPU、4K30、时延、功耗与热稳态尚未验收，因此 `release_ready=false`。

## 快速开始

```powershell
# 全部 Python 测试
.\.venv\Scripts\python.exe -m pytest -q

# 无 GPU 电脑上的完整阶段冒烟
.\.venv\Scripts\python.exe -m ai_isp.pipeline `
  --config configs\train\v4_cpu_全流程.yaml

# 真实 RYYB 量产配置；需先准备 Manifest、GPU、目标 DDK 和设备
.\.venv\Scripts\python.exe -m ai_isp.pipeline `
  --config configs\train\v4_量产训练.yaml
```

## 主要入口

- [V4.0 详细开发设计](./AI%20ISP%20暗光拍照预览%20RAW%20降噪详细开发设计%20V4.0.md)
- [V4 项目总结与学习指南](./AI%20ISP%20V4%20项目总结与学习指南.md)
- [V4 开发关键步骤与注意事项](./docs/V4开发关键步骤与注意事项.md)
- [V4 CPU 全阶段验证报告](./docs/V4%20CPU全阶段验证报告.md)
- [SIDD 数据集数据卡](./docs/SIDD数据集数据卡.md)
- [V3.0 历史设计](./AI%20ISP%20暗光拍照预览%20RAW%20降噪详细开发设计%20V3.0.md)

文档中的目标端画质、性能、功耗和内存数值均是准入门槛，不是当前电脑或麒麟 9000 的已验证结论。
