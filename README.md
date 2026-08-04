# AI-ISP

面向手机 RAW 域的 AI 降噪与增强项目，目标平台为麒麟 9000。

当前开发基线：

- [AI ISP 暗光拍照预览 RAW 降噪详细开发设计 V3.0](./AI%20ISP%20暗光拍照预览%20RAW%20降噪详细开发设计%20V3.0.md)：30fps 暗光 Photo Preview，Full-Tensor Single Pass、三静态 OM 发布基线/多档 OM 条件候选、StaticSimpleGate、Torch-Pruning、PTQ 验链与 LSQ+ QAT。
- [AI ISP 量产设计基线 V18.0](./AI%20ISP量产设计基线%20V18.0.md)：项目总体量产设计基线。

历史设计：

- [AI ISP 暗光拍照预览 RAW 降噪详细开发设计 V2.0](./AI%20ISP%20暗光拍照预览%20RAW%20降噪详细开发设计%20V2.0.md)：上一版暗光 Preview 详细设计；三静态 OM、MobileNAFNet-W16 和通用 INT8 QAT 基线由 V3.0 细化。
- [AI ISP RAW 降噪增强详细开发设计 V1.0](./AI%20ISP%20RAW降噪增强详细开发设计%20V1.0.md)：面向 Single/HDR/Capture 与 12MP/50MP RAW 的 Tile 推理详细设计；在 Dark Photo Preview 场景由 V2.0/V3.0 取代。

## 当前实现状态

CPU/ONNX 工程闭环已经跑通：

- 四种 CFA 的 RAW Pack/Unpack、P1 540→544 Pad/Crop、ConditionSchemaV2；
- 660,836 参数 MobileNAFNet-W16 Student、W32 Teacher 骨架、StaticSimpleGate、RepDenseGate 折叠；
- SIDD HDF5 Patch 数据管线、RAW/Tone/Gradient Loss、物理噪声、按场景防泄漏；
- 少量真实 SIDD 数据 CPU 训练、safetensors 冻结、PTQ 前置验链；
- P0/P1/P2 三个真实发布 Shape ONNX 和 ONNX Runtime 数值对齐；
- Python Trigger/Profile 与 C++ Runtime 失败安全骨架。

本机发布分辨率 CPU 推理约为 1.0～1.6 秒/帧，明确不作为 30fps 回退。OM、NPU 100% 落点、完整 QAT、目标设备画质/功耗/温升仍需麒麟 9000、商用 DDK 和目标 Sensor 数据验证。

## 快速开始

```powershell
# 运行测试
.\.venv\Scripts\python.exe -m pytest -q

# 用少量真实 SIDD Patch 跑通 CPU 全流程
.\.venv\Scripts\python.exe -m ai_isp.cli `
  --dataset-root datasets\SIDD_Training_Subset `
  --output-dir artifacts\cpu_smoke `
  --steps 4 --patch-size 32 --max-pairs 2 --profile-mode smoke
```

开发与验证记录：

- [开发关键步骤与注意事项](./docs/开发关键步骤与注意事项.md)
- [CPU 全流程验证报告](./docs/CPU全流程验证报告.md)
- [SIDD 数据集数据卡](./docs/SIDD数据集数据卡.md)

文档中的性能、功耗、内存和画质数值均为目标设备准入门槛，正式量产结论以目标设备实测为准。
