# AI ISP V6.1 项目开发指导与述职学习文件

> 用途：后续复盘、技术培训与述职答辩
> 项目：Kirin9000 Dark Photo Preview AI RAW Denoise
> 当前结论：工程流程完成，P36-16 为模拟最优；目标端尚未验证

## 1. 一句话介绍项目

本项目在 RYYB 主摄和长焦的暗光 Photo Preview 链路中，用一个带 Camera/Sensor 条件的 MobileNAFNet 在 Demosaic 前预测 RAW 噪声，通过固定混合精度、16 通道结构化剪枝和零拷贝运行时契约降低 NPU 成本，同时用严格的失败闭锁避免错误相位、错误 RAW 域和不可信性能结论进入量产。

## 2. 述职时建议先讲的三个价值

1. **算法价值**：在 RAW 域处理可利用线性信号和物理噪声模型，避免 RGB 域噪声与 ISP 非线性混合后更难分离。
2. **工程价值**：Main/Tele 共用一个模型，以 Condition 和 Sensor Profile 隔离差异，减少模型数量、内存和维护成本。
3. **量产价值**：把相位、RAW 域、量化、DMA-BUF、Fence、Manifest 和目标端门禁做成可执行契约，降低“模型能跑但链路不能量产”的风险。

## 3. 完整开发流程

```text
需求与硬件边界冻结
→ RAW Domain / Sensor Profile / Condition Schema
→ 数据 Manifest 与物理噪声
→ W32 Teacher
→ W16 Dense Student
→ Feature + Attention KD + Temporal
→ P10/P18/P36 三个16对齐候选
→ 各自剪枝恢复与全尺寸 Phase1
→ Q0校准与Q1量化探针
→ 硬门槛 + 确定性选优
→ 删除落选可执行制品
→ 获胜者 Q2/Q3
→ Dynamic Affine 整数审计
→ FP32 Q/DQ Reference + FP16 Mixed ONNX
→ ATC/OM 失败闭锁
→ DMA-BUF/C++ Runtime 验证
→ Manifest 和归档
```

每个阶段都必须有输入契约、输出制品、自动检查和失败处理，不能只留下训练脚本或单个权重。

## 4. 最关键的技术点

### 4.1 为什么 Pack 后还必须 Unpack

模型看到的是 `[R,Yr,Yb,B]` 四通道语义张量，后级 Demosaic 看到的是二维物理 CFA。不同 Sensor 的 RYYB 相位可能不同，所以模型输出在 Subtract/Clamp 后必须按 Sensor Profile 做 Pack 的位精确逆操作。若直接把四通道送给 Demosaic，数据维度和物理格点都错误。

可用于答辩的表述：**模型语义统一不等于传感器物理相位统一。**

### 4.2 为什么 Post-BLC/LSC 不能再减 Black Level

BLC 已消除黑电平，Formatter 如果再次执行 `(raw-black)/(white-black)` 会造成暗部负偏、剪零和噪声统计失真。正确公式只用 `white-black` 作为动态范围分母：`raw_post_lsc/(white-black)`。

### 4.3 为什么 Main/Tele 用一个模型

共享主干学习通用暗光去噪，Condition 提供 ISO、曝光、增益、温度、Camera、Sensor、CFA 等差异。这样减少模型切换和内存，但前提是 Condition Schema、Sensor Profile 和 Hash 准入严格一致。

### 4.4 为什么时序损失必须比较重建结果

两次独立噪声的真实残差本来就不同，直接约束 `N1=N2` 会鼓励零预测或过平滑。正确做法是分别重建 `y_i=clip(z_i-N_i)`，再在有效、非运动、非遮挡、非饱和区域比较 `y1/y2`。

### 4.5 为什么只保留 16 通道对齐候选

通道对齐直接影响 NPU Vector/Matrix 单元利用率、内存布局和算子融合。参数少不等于目标端快，因此先限制硬件友好的拓扑，再在画质等价候选中比较 MAC、Activation、Cast/QDQ 和延迟代理。

### 4.6 为什么最终选择 P36-16

P10-16、P18-16、P36-16 在本次 SIDD 工程模拟中均通过硬门槛；质量指标等价时，P36-16 的发布 Shape MAC 为 28.14G、参数 419,652，均为三者最低。因此按冻结的确定性顺序选择 P36-16，而不是凭主观偏好选择。

### 4.7 混合精度不是“部分层随便用 FP16”

本项目的精度映射是固定接口：公共输入输出、Intro/Ending、Condition Encoder 和前段 FiLM 为 FP16；主干 INT8；Stage3/Middle Dynamic Affine 具有整数语义。任何全 INT8、全 FP16 或另一套 FiLM 分支都不是发布候选。

## 5. 本轮最有价值的问题定位

### 5.1 Q0 临时量化污染下游统计

最初 Observer 在收集数据的同时执行尚未校准的 FakeQuant，导致后层看到的范围取决于初始化顺序。修复方式是 Q0 时 FP32 透传，只收集统计；校准完成后才进入 QAT。

### 5.2 校准样本只覆盖前几帧

原实现达到 65k 元素上限后停止收集，一个早期特征层只需几帧就填满，后面的 Camera/ISO 看不到。修复为每帧均匀抽样并对观察块做确定性 Reservoir。

### 5.3 LSQ+ Offset 学习率量纲错误

`log_scale` 是无量纲参数，Offset 位于真实反量化数值域。二者用相同绝对 LR 会让窄范围层的 Offset 一步移动多个量化格。修复为 Offset LR 乘当前量化 Scale。

### 5.4 饱和率不能用训练历史最大值

训练包含随机时序噪声和临时参数，不能代表发布分桶。最终门禁改为 Q1 后独立探针，Main/Tele 平衡，并排除以极小通道数为分母的 Per-channel 权重极值；权重由溢出和数值误差审计负责。

### 5.5 不能通过放宽门槛解决饱和

项目保留 `<0.1%` 硬门槛。先给所有激活 10% 稳健余量，再仅对失败层每轮扩张 5%，重新执行同一探针，直到通过或直接失败。这比全局大幅扩大 Range 更能保留 INT8 分辨率。

## 6. 代码阅读路线

建议按以下顺序学习：

1. `ai_isp/data/ryyb_contract.py`：先理解相位、准入和 Pack/Unpack；
2. `ai_isp/models/mobile_nafnet.py`：理解 Condition、FiLM 和 MobileNAFBlock；
3. `ai_isp/losses/dark_preview_losses.py`：理解完整损失；
4. `ai_isp/training_stages.py`：理解 Teacher/Student/KD；
5. `ai_isp/pruning/nafnet_pruning_validator.py`：理解真实结构化剪枝；
6. `ai_isp/quantization/lsqplus_qat.py` 与 `qat_training.py`：理解 Q0-Q3；
7. `ai_isp/pipeline_v6.py`：串联全流程和唯一候选收敛；
8. `runtime/include/dark_preview_denoise.h`：理解 HAL/NPU 生产边界；
9. `ai_isp/export/om_release.py`：理解 Manifest 和失败闭锁。

## 7. 如何运行与核对

```powershell
# Python 全量测试
.\.venv\Scripts\python.exe -m pytest -q

# V6.1 CPU 完整阶段冒烟
.\.venv\Scripts\python.exe -m ai_isp.pipeline_v6 `
  --config configs\train\v6_1_cpu_全流程.yaml

# 或使用工具入口
.\.venv\Scripts\python.exe tools\运行V6.1全流程.py `
  --config configs\train\v6_1_cpu_全流程.yaml

# 真实 RYYB 量产流程（需数据、GPU、ATC/DDK 和真机）
.\.venv\Scripts\python.exe -m ai_isp.pipeline_v6 `
  --config configs\train\v6_1_量产训练.yaml
```

C++ Runtime 使用 CMake 3.20+；本次开发机缺少 CMake 命令，因此用 VS2022 Developer Command Prompt 的 MSVC 19.42 直接编译并运行同一测试程序。正式 CI 应恢复 CMake/CTest。

## 8. 验收检查表

### 数据

- Main/Tele 是否各自满足训练/验证/盲测场景数；
- 是否全部是 Post-BLC/LSC Pre-DGain；
- LSC、RAW Domain、Sensor/Unpack Hash 是否一致；
- 是否存在 90°/270° Rotation 或奇数 Crop；
- 校准是否 ≥4096 且 Main/Tele 等量。

### 模型与量化

- 所有候选通道是否 16 对齐；
- SimpleGate 是否是 `⊙` 和 INT32 Multiply；
- Condition 是否只有一次 Cast；
- Stage3/Middle 是否无 `DQ→FP16 Mul/Add→Q`；
- 关键激活每 Camera/ISO 最大饱和率是否 `<0.1%`；
- 是否只剩唯一 Candidate ID 和唯一 Quant Policy。

### Runtime

- Pack→Unpack 四相位是否位精确；
- Stride、Crop、Offset 是否正确；
- FD 是否只导入一次；
- 每帧 CPU memcpy/map-unmap 是否为 0；
- Fence、Cache、超时回收是否通过；
- 任一异常是否立即 Bypass，而不是 CPU AI 回退。

### 发布

- 真实 RYYB 画质、时序和分桶是否通过；
- 商用 ATC 是否生成唯一 OM；
- 100% NPU、6/8/9/10ms、30fps 是否实测；
- 功耗、热稳态、10000 帧和回滚是否通过；
- Manifest 四个状态位是否与证据一致。

## 9. 常见答辩问题

**问：既然 CPU 流程完成，为什么还不能发布？**

答：CPU 只能证明代码、数学和制品流程；ATC Fusion、NPU 覆盖、DMA/Fence 延迟、功耗和真实 RYYB 画质必须在目标环境验证。

**问：为什么不直接选最小模型？**

答：先过画质、饱和、数值和图结构硬门槛，只有画质等价时才比较 MAC/内存/延迟。P36-16 是按这个顺序获胜。

**问：为什么不保留三个模型等真机再选？**

答：项目要求开发模拟后立即收敛唯一方案，减少分支维护和发布歧义；真机验证只验证已选方案，失败则 No-Go，不悄悄切换未归档分支。

**问：0.2ms 是否已经达成？**

答：没有。0.2ms 只是 Buffer 已导入后的 Fence/Cache 同步优化目标；当前只证明契约与 0 Byte 额外 memcpy 的模拟状态机。

**问：为什么 ONNX Q/DQ 不能证明 Dynamic Affine 融合？**

答：Q/DQ 表达量化边界，不保证 ATC 采用特定硬件内核。必须查看目标编译图和 Profiling，确认没有 FP16 回退。

## 10. 述职总结模板

> 我负责把暗光 RYYB RAW 降噪从算法设计推进到可执行工程闭环。项目统一了 Main/Tele 的语义通道，同时保留 Sensor 物理相位的位精确逆映射；完成 Teacher/Student、双层 KD、时序损失、三种 16 对齐剪枝、固定 LSQ+ 混合精度 QAT、ONNX/Manifest 和 DMA-BUF Runtime 契约。开发中通过逐层探针定位并修复 Q0 统计污染、校准样本偏置和 Offset 学习率量纲问题。三个候选通过同一门槛后，按画质等价、MAC 和内存顺序选择 P36-16，并立即删除落选可执行制品。当前工程状态真实标记为 development selected，但由于没有商用 ATC 和麒麟 9000，目标端性能与发布状态仍保持 No-Go，下一阶段重点是 RYYB 量产训练、ATC 图审计和真机全链验收。
