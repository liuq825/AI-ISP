# SIDD RAW 数据集数据卡

> 用途：AI ISP CPU 工程闭环、公开 RAW 预训练与泛化验证。  
> 限制：SIDD 不能替代目标麒麟手机的 Sensor 标定和 Noisy/Clean 数据。

## 1. 本地数据清单

|目录|场景/配对|文件数|体积|用途|
|---|---:|---:|---:|---|
|`SIDD_Training_Subset`|6 场景、6 对 RAW|19|约 398.6 MB|CPU 小样本训练闭环|
|`SIDD_Medium_Range`|160 场景、320 对 RAW|961|约 21.62 GB|后续公开数据训练/验证|
|`SIDD_Blocks`|40×32 对 256×256 RAW Block|26|约 594.2 MB|标准验证块、CFA/噪声元数据|

总数据量约 22.61 GB。数据目录不进入 Git。

## 2. 文件格式

- 完整 GT/Noisy RAW：MATLAB v7.3/HDF5，键为 `x`，`float32`，范围 `[0,1]`。
- h5py 返回的二维 Shape 是 `[W,H]`；项目读取器按 `[x,y]` 切 Patch 后转置为 `[H,W]`。
- 验证块：传统 MAT，变量 `ValidationNoisyBlocksRaw`/`ValidationGtBlocksRaw`，Shape 为 `40×32×256×256`。
- Camera CFA：GP=BGGR、IP=RGGB、S6=GRBG、N6=BGGR、G4=BGGR。
- Noise Level Function：从 `noise_level_functions.csv` 读取场景级 Shot/Read 参数，缺失时仅允许工程默认值并记录。

## 3. 场景命名解析

示例：`0013_001_S6_03200_01250_3200_L`。

- `0013`：场景编号；
- `001`：实例编号；
- `S6`：Camera ID；
- `03200`：ISO；
- `01250`：曝光分母，代码编码为约 `1/1250s`；
- 后续字段保留为 SIDD 场景属性，不擅自映射为目标设备 Sensor Profile。

## 4. 划分和泄漏控制

- 划分最小单位是完整 `scene_name`，同一场景的不同帧不得跨 Train/Validation/Test。
- `split_pairs_by_scene()` 以固定随机种子执行分组划分，并有单元测试验证集合不相交。
- 正式训练应另建独立盲测表；不能使用 SIDD Validation Block 同时调参和宣称盲测结果。

## 5. 已知偏差与禁止结论

- SIDD 来自公开手机，不是项目目标麒麟 9000 手机；Camera、Lens、CFA、噪声、Black/White 和上游 ISP 域均有差异。
- 当前数据已归一化，不能反推出可靠的目标设备 RAW10/RAW12 Black/White Level。
- SIDD 可用于验证代码、预训练和泛化，但在最终数据组成中应遵守设计的公开 RAW 占比 `≤10%`。
- 不允许用 SIDD 的 CPU PSNR/SSIM 宣称麒麟设备的最终画质、30fps、功耗或量产通过。

