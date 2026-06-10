# 软件操作说明

本文档记录当前软件的主要运行命令和常用操作流程。项目包含两个入口：

- 双摄/标定/测距主程序：`stereo_aruco_gui`
- DataMatrix 解码调试器：`decoder_debugger`

## 安装依赖

在项目根目录执行：

```powershell
cd C:\Users\ziylam\Desktop\cameraDebug
python -m pip install -r requirements.txt
```

如果使用虚拟环境：

```powershell
cd C:\Users\ziylam\Desktop\cameraDebug
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 启动双摄主程序

```powershell
cd C:\Users\ziylam\Desktop\cameraDebug
python -m stereo_aruco_gui.main
```

常用流程：

1. 点击 `Scan Cameras` 扫描摄像头。
2. 选择左右摄像头。
3. 选择分辨率，建议先用稳定分辨率，再逐步提高。
4. 点击 `Open Cameras` 打开摄像头。
5. 根据需要切换不同视图：
   - `Live Preview`
   - `Calibration`
   - `Rectified Preview`
   - `Depth / Distance`
   - `2D Measurement`
   - `Barcode Detection`

## 启动 DataMatrix 解码调试器

这个界面专门用于离线导入图片，调试 DataMatrix 找码、预处理和解码能力，不需要打开摄像头。

```powershell
cd C:\Users\ziylam\Desktop\cameraDebug
python -m decoder_debugger.main
```

## 解码调试器界面结构

顶部有三个页面：

- `Preprocess`：调试预处理参数和效果图。
- `Find / Decode`：找码、画 ROI、选择解码方式、执行解码。
- `Batch / Logs`：批量和日志预留页面。

## 导入图片

在 `Preprocess` 页面：

1. 点击 `Open Image` 导入单张图片。
2. 或点击 `Open Folder` 导入一个图片目录，软件会读取目录中的图片。
3. 图片路径会显示在左侧底部。

支持常见图片格式：

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`

## 预处理调试

在 `Preprocess` 页面：

1. `Preview View` 默认是 `Original`，显示原图。
2. 如果要查看预处理效果，把 `Preview View` 切到 `Preprocessed`。
3. 调整左侧预处理算子参数，右侧效果图会实时刷新。

当前可调参数包括：

- `Channel`
- `CLAHE`
- `Blur`
- `Sharpen`
- `Threshold`
- `Adaptive`
- `Invert threshold`
- `Morphology`
- `Scale`
- `Quiet zone padding`

## 内置预处理场景

`Scene` 下拉可以一键加载内置参数：

- `Manual`
- `Early decoder - gray 3x`
- `Early decoder - clahe 3x`
- `Early decoder - sharpen 3x`
- `Early decoder - otsu 3x`
- `Early decoder - adaptive 3x`
- `ROI finder - saturation source`

选择场景后，软件会自动把对应参数填入控件。之后如果继续手动调整参数，场景会回到 `Manual`。

## 保存和加载预处理参数

在 `Preprocess` 页面：

- 点击 `Save Preset` 保存当前预处理参数为 JSON。
- 点击 `Load Preset` 加载之前保存的 JSON。

这适合把某一批图片调好的预处理参数保存下来，下次继续使用。

## 找码流程

切换到 `Find / Decode` 页面。

1. 在 `Input View` 选择找码输入：
   - `Original`：用原图找码。
   - `Preprocessed`：用当前预处理结果找码。
2. 点击 `Find Code`。
3. 软件会在右侧图像中画出候选 ROI 框。
4. 右侧 `Decode Inspector` 会显示候选数量和耗时。

说明：

- 绿色框表示候选或成功结果。
- 红色框表示该 ROI 解码失败。
- 当前找码逻辑和解码逻辑是分开的：找码只负责找候选 ROI，解码负责读取内容。

## 选择解码方式

在 `Find / Decode` 页面使用 `Decoder Method`：

- `Auto`
  - 先尝试 `Native`。
  - 如果 `Native` 没有结果，再回退到 `ZXing`。
- `Native`
  - 只使用当前项目自研 DataMatrix 解码器。
  - 当前第一版支持标准方形 ECC200 小码、ASCII payload、单 data region。
  - 还不完整支持 Reed-Solomon 纠错、多 data region、C40/Text/Base256 和复杂透视。
- `ZXing`
  - 强制使用 `zxing-cpp`。
  - 用于和自研解码器做对比。

## 解码流程

找码之后有两种解码方式：

1. `Decode All Candidate ROIs`
   - 把所有候选 ROI 都送去解码。
   - 推荐优先使用这个按钮。
   - 即使某些芯片/焊盘被误框，也不会挡住其他真正 DataMatrix ROI 的结果。

2. `Decode Selected ROI`
   - 只解当前选中的 ROI。
   - 适合单独排查某个 ROI 为什么失败。

成功后，右侧 `Decode Inspector` 会显示：

```text
Decoded Text:
  ROI 3: JMB3404T
```

失败时会显示：

```text
ROI 3: failed - No DataMatrix decoded in ROI
```

如果选择 `Native`，失败诊断可能显示：

```text
Native DataMatrix decoder is not complete yet
```

这表示当前自研解码器还没有覆盖该图像情况，需要继续增强 native 解码算法。

## 当前 DataMatrix 解码能力说明

当前 DataMatrix 管线分为三层：

1. 找码：`industrial_code_reader/datamatrix/roi.py`
   - 从整图中找到可能是 DataMatrix 的 ROI。

2. 调度：`industrial_code_reader/core/engine.py`
   - 裁剪 ROI。
   - 调用指定解码器。
   - 合并结果和失败诊断。

3. 解码：`industrial_code_reader/datamatrix/decoder.py`
   - 根据 `method` 选择 `Auto`、`Native` 或 `ZXing`。

自研解码器入口：

```text
industrial_code_reader/datamatrix/native.py
```

当前 Native v1 已支持：

- 方形 ECC200 DataMatrix。
- 单 data region 小尺寸。
- Otsu / Adaptive 二值化尝试。
- quiet zone 裁剪。
- finder pattern 评分。
- 模块网格采样。
- Utah 码字读取。
- ASCII encodation payload 解码。

当前 Native v1 尚未完整支持：

- Reed-Solomon 纠错。
- 多 data region 大码。
- C40 / Text / X12 / Base256。
- 强透视、模糊、反光、缺 quiet zone 的工业现场图。

## 常见问题

### 预处理参数调了但图像没变化

确认 `Preview View` 是否切到了 `Preprocessed`。如果仍然是 `Original`，右侧会一直显示原图。

### 找到了很多红框

红框表示候选 ROI 解码失败。工业图里芯片、焊盘、丝印、边缘纹理都可能被误认为候选。优先看 `Decoded Text` 是否有成功结果。

### 某个中间码没有识别到

先确认：

1. `Find Code` 后是否画出了覆盖该码的 ROI。
2. 如果没有 ROI，问题在找码阶段。
3. 如果有 ROI 但解不出，问题在解码阶段或预处理阶段。
4. 可切换 `Input View = Preprocessed`，尝试不同 `Scene` 和参数。
5. 可切换 `Decoder Method = Native` / `ZXing` 做对比。

### Native 解不出但 ZXing 能解出

说明当前自研解码器还缺对应能力。优先记录失败图片、ROI、预处理参数，后续用于补 native 解码算法。

### ZXing 解不出但 Native 也解不出

优先检查：

- ROI 是否准确覆盖完整码。
- 模块是否清晰。
- 是否过曝或反光。
- 是否缺 quiet zone。
- 码是否旋转、透视或模糊严重。

## 推荐调试顺序

1. 用 `Open Image` 导入图片。
2. 在 `Preprocess` 页面先看原图。
3. 选择一个内置 `Scene`，比如 `Early decoder - adaptive 3x`。
4. 切到 `Preview View = Preprocessed` 查看效果。
5. 到 `Find / Decode` 页面。
6. `Input View` 先用 `Original`，找不到再试 `Preprocessed`。
7. 点击 `Find Code`。
8. 点击 `Decode All Candidate ROIs`。
9. 查看 `Decoded Text` 和失败 ROI。
10. 在 `Decoder Method` 中切换 `Auto`、`Native`、`ZXing` 做对比。
