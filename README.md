# AI-ISP

面向手机 RAW 域的 AI 降噪与增强项目，目标平台为麒麟 9000。

当前开发基线：

- [AI ISP 暗光拍照预览 RAW 降噪详细开发设计 V2.0](./AI%20ISP%20暗光拍照预览%20RAW%20降噪详细开发设计%20V2.0.md)：30fps 暗光 Photo Preview，Full-Tensor Single Pass、三静态 Profile、Tiny Conditional MobileNAFNet-W16。
- [AI ISP 量产设计基线 V18.0](./AI%20ISP量产设计基线%20V18.0.md)：项目总体量产设计基线。

历史设计：

- [AI ISP RAW 降噪增强详细开发设计 V1.0](./AI%20ISP%20RAW降噪增强详细开发设计%20V1.0.md)：面向 Single/HDR/Capture 与 12MP/50MP RAW 的 Tile 推理详细设计；在 Dark Photo Preview 场景由 V2.0 取代。

当前阶段为 CPU 工程验证、云端 GPU 训练准备和端侧可迁移设计。文档中的性能、功耗、内存和画质数值均为目标设备准入门槛，正式量产结论以目标设备实测为准。
