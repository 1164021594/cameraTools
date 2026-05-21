# 摄像头排障记录

## Windows 后端选择

`Backend` 下拉框必须保留这些选项：

- `AUTO`
- `DSHOW`
- `MSMF`
- `DEFAULT`

不能把 `MSMF` 和 `DEFAULT` 从界面里删掉。不同 USB 摄像头、不同驱动、不同分辨率下，最流畅的 OpenCV 后端可能不一样，用户需要能手动切换测试。

当前策略：

- `AUTO` 默认只自动尝试 `DSHOW`。
- 手动选择 `DSHOW`、`MSMF`、`DEFAULT` 时，软件按用户指定的后端打开。
- `MSMF` 保留给手动测试，但不作为 `AUTO` 的自动 fallback。
- 本机这两只独立 USB 摄像头实测 `MSMF` 帧率和画面流畅度最好。如果 `AUTO` 或 `DSHOW` 下出现一侧卡顿，优先手动切到 `MSMF` 并保存配置。

这样做的原因是：前面遇到过 OpenCV `MSMF` 在双 USB 摄像头运行时反复报错：

```text
CvCapture_MSMF::grabFrame videoio(MSMF): can't grab frame. Error: -2147483638
```

如果 `AUTO` 在 DSHOW 临时读帧失败后自动切到 MSMF，可能把一次短暂失败变成长期卡顿或抓帧失败。所以 `AUTO` 保守使用 DSHOW；但界面仍允许手动选择 MSMF/DEFAULT，方便对具体摄像头做对比。

## 采集循环规则

双摄预览必须让左右摄像头各自独立线程采集，并且采集线程中不要做人为 FPS 节流。

这次排查到的关键问题：

- 旧逻辑在采集循环里使用 `grab()`，再按 FPS 间隔决定是否 `retrieve()`。
- 这种 `grab/retrieve` 分离加节流的方式，在部分 Windows 后端，尤其是本机实测更适合的 `MSMF` 下，会导致某一路摄像头隔一段时间才更新。
- 现象是 FPS 看起来不稳定，或者一侧画面明显比另一侧慢。
- 改为每个摄像头线程直接 `read()`，并且每读到一帧就立即保存为最新帧后，帧率和预览流畅度恢复正常。

当前原则：

- 采集线程：`read()` -> 保存最新帧。
- 不在采集线程里用 `grab/retrieve` 做跳帧。
- 不在采集线程里按 FPS 人为延时。
- UI 预览定时器只负责显示“最新帧”，不反向限制采集速度。

以后如果再次出现“一路摄像头过几秒才刷新一次”，优先检查是否有人把采集循环改回了 `grab/retrieve` 或增加了 sleep/节流逻辑。

## 打开成功不等于分辨率真正可用

有些摄像头在 `1280 x 960`、`1024 x 768` 等模式下会出现这种情况：

- `cap.isOpened()` 返回成功。
- OpenCV 显示实际分辨率也是请求的分辨率。
- 但后续 `read()`/`grab()` 持续失败，预览没有画面。

这说明驱动接受了参数，但该组合没有稳定输出视频流。软件需要按“持续拿到图像帧”判断成功，而不是只看 `isOpened()`。

当前打开逻辑：

- 打开阶段最多等待 10 秒。
- 只有连续拿到多帧有效图像，才认为摄像头打开成功。
- 如果 10 秒内没有稳定视频流，自动释放摄像头资源，按钮恢复为 `Open Cameras`。
- 打开过程中禁用按钮，避免正在打开时反复点 `Close` 导致线程还没退出就销毁。

## 分辨率选择

这批摄像头资料对应 GC5035 5MP 模组，原生比例更接近 4:3。双目标定和稳定预览优先使用：

- `640 x 480`
- `800 x 600`
- `1024 x 768`
- `1280 x 960`
- `2592 x 1944`

不建议用 `1280 x 720` 做双目标定预览。它是 16:9，部分 UVC 摄像头在这个模式下可能需要裁剪、缩放或切换输出格式，容易出现能打开但采集不到有效图像、黑帧、读帧失败或触发后端问题。

如果 `1280 x 960` 无画面，优先按这个顺序测试：

1. `640 x 480 @ 15 FPS`，`Format = MJPG`。
2. 单摄模式，只接一个摄像头测试。
3. 双摄模式下把两只摄像头插到不同 USB 主控或不同 USB3.0 根集线器。
4. 手动切换 `Backend`：`DSHOW`、`MSMF`、`DEFAULT` 分别测试。
5. 如果拔掉另一只摄像头后当前摄像头变流畅，优先怀疑 USB 带宽、主控共享或驱动调度问题。

## 重点看这些日志

GUI 状态栏里比较有用的日志：

- `left: opened index ... with DSHOW`
- `left: opened index ... with MSMF`
- `right: opened index ... with DSHOW`
- `right: opened index ... with MSMF`
- `failed to read frame`
- `reopening after repeated read failures`
- `failed to open camera ... error=open timeout`
- `closed (thread exited)`

`opened index` 只表示软件已经进入采集循环，不代表该分辨率长期稳定。如果马上连续出现 `failed to read frame`，说明该摄像头或该后端/分辨率组合不稳定。

## 预览无画面时

如果右摄能出图、左摄没有帧，软件应继续显示右摄画面，并在左侧显示 `Waiting for left camera`。不要因为左摄没有帧就阻塞整个预览刷新。

如果左摄反复失败并重开：

```text
left: failed to read frame
left: reopening after repeated read failures
```

优先降低分辨率和 FPS，再手动测试其他后端。

如果日志没有失败，但画面更新慢，重点看预览角落 overlay：

- `frames` 持续增加，`age` 很小：采集正常，问题更可能在 UI 显示或后处理。
- `frames` 增长很慢，`age` 周期性变大：对应摄像头的 `read()` 被后端、驱动或 USB 链路阻塞。
- 如果 `MSMF` 下 `frames` 和 `age` 正常，而 `DSHOW/AUTO` 下异常，就固定使用 `MSMF`。
