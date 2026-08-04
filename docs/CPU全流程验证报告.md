# AI ISP CPU 全流程验证报告

> 验证日期：2026-08-04  
> 结论：CPU 工程闭环通过；不满足 30fps，不能作为量产 AI 回退；目标设备发布仍被 DDK、OM、NPU 与目标数据验证阻断。

## 1. 验证环境

- 操作系统：Windows 11 10.0.26200 x86_64
- Python：3.12
- PyTorch：2.13.0+cpu，CUDA 不可用
- ONNX：1.22.0
- ONNX Runtime：1.28.0 CPUExecutionProvider
- h5py：3.16.0
- scipy：1.18.0
- Torch-Pruning：1.6.1
- PyTorch CPU 线程：4
- C++：MSVC 19.42.34435、C++17、CMake/Ninja

## 2. 真实数据小样本训练

配置：2 个 SIDD 场景、Packed 32×32 Patch、Batch 1、4 Steps、AdamW、固定随机种子 20260804。

|项目|结果|
|---|---:|
|Student 参数量|660,836|
|4 Step 总耗时|0.298 秒|
|平均每 Step|0.075 秒|
|Step 1 RAW PSNR|27.022 dB|
|Step 4 RAW PSNR|34.283 dB|
|快速验证 RAW PSNR|27.022 dB|
|快速验证全局 SSIM|0.5236|
|输出有限值|通过|

这些数值只说明反向传播、优化和评估可运行。训练和验证样本极少，不能据此判断模型收敛、泛化或量产画质。

## 3. 冻结与量化前置验链

- 冻结权重：safetensors；SHA-256 为 `a12d8138798909c4250cf066da6dbd847a30a3fdcf3e8aab37bcae89d24f38c5`。
- 拓扑：W16、Encoder `[2,2,4]`、Middle `2`、Decoder `[2,2,2]`、Condition 24、输出 `noise_pred`。
- PTQ 张量快速验链：最大绝对误差 `0.00195986`，平均绝对误差 `0.00097884`，饱和率 `0.0004883`。
- PTQ 结果不等于完整模型 W8A8；LSQ+ QAT、Q/DQ、ATC 和 OM 尚需目标工具链完成。

## 4. 三静态 ONNX 发布 Shape

|Profile|输入 Shape|PyTorch→ORT 最大误差|动态 Slice 输入|非白名单算子|ONNX SHA-256|
|---|---|---:|---:|---:|---|
|P0|1×4×768×1024|6.52e-9|0|0|`49d693c...57d9f`|
|P1|1×4×544×960|6.52e-9|0|0|`44520ee...c124f`|
|P2|1×4×640×960|6.52e-9|0|0|`3ec94a0...0a387`|

实际算子集合为 `Add/Constant/Conv/Gemm/Mul/Relu/Resize/Slice/Tanh/Unsqueeze`。三个图均满足误差 `≤1e-4`，StaticSimpleGate 的 Slice 切点全部由常量提供。

## 5. 发布分辨率 CPU 性能边界

每档预热 1 次、测量 3 次：

|Profile|P50|本次最大样本|30fps 预算 33.3ms 的倍数|
|---|---:|---:|---:|
|P0|1558.3 ms|1600.7 ms|约 46.8×|
|P1|987.3 ms|1073.9 ms|约 29.6×|
|P2|1163.2 ms|1166.5 ms|约 34.9×|

因此，本机适合小样本训练、图导出和数值验证，不适合 30fps Preview 推理。设计中的“禁止 CPU AI 回退”得到实测支持。

## 6. 测试结论与未覆盖项

已覆盖：四 CFA Pack/Unpack、Condition 24 维、P1 Pad/Crop、模型 Shape、强度 0 恒等、Rep 折叠、剪枝不变量、LSQ+ 基元、Trigger、真实 SIDD Patch、按场景防泄漏、ONNX 导出与对齐；C++ Runtime 以 MSVC/C++17 构建，1 项 CTest 通过。

未覆盖且仍为发布阻断项：

- 完整目标设备数据训练、Teacher/KD、P10/P15 恢复和画质盲测；
- 完整模型 PTQ、LSQ+ QAT、对称 LSQ A/B 和逐层 Q/DQ 审计；
- 麒麟 9000 ATC/OM 转换、NPU 100% 落点、端到端 30fps；
- 目标 Camera RAW Stream、P1 零冗余写入、多摄切换、功耗、温升和 10,000 帧稳定性；
- C++ Runtime 已完成本地编译和断言测试，但尚未接入目标 Camera HAL、DDK 与真实 NPU Executor。
