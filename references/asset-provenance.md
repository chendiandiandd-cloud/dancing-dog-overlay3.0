# 素材来源与维护

## 来源

- 原作者发布页：https://www.douyin.com/note/7698095369871264232
- 本次下载时间：2026-11-04。
- 来源集合：本次下载合集的 `01_跳舞小狗绿幕`，共16段动态MP4；原素材可能含其他卡通形象，选择时按实际画面判断，不作生物类别识别。
- 图片效果参考：https://www.douyin.com/note/7696539308497474176
- 视频效果参考：本次用户附件 `20261104-142732.mp4`，仅作为开发参考，不是运行必需文件，不纳入素材库。

## 包内资源

`assets/source/` 保存本次下载的原始绿幕 MP4，使用稳定的 dog_01 等名称。
`assets/dogs/` 保存离线处理后的 RGBA 动作帧缓存，`assets/dog_manifest.json` 记录路径、尺寸、帧率、周期、脚底锚点、主色、内容哈希。

清单中的 `file` 与 `source` 相对于 assets；不依赖当前 workspace 绝对路径、浏览器会话、登录状态或临时在线签名URL。RGBA缓存为 NumPy NPZ，使用 `allow_pickle=False` 加载。

## 维护原则

源视频本身不带透明通道。素材入库阶段完成抠绿和固定裁切，生产运行复用RGBA缓存。

本次版本保留16份源视频共828帧，运行缓存共827帧：dog_15原片第59帧（零基索引58）突然切为蓝色角色，缓存裁掉这一帧，使用前58帧；源视频不修改。其余素材保留原帧数。严格近背景色的封闭绿幕空洞已清理，dog_06的绿色/青色细节保留；少量压缩污染与细边仍需按场景检查。源动作并非全部无缝周期，尤其dog_15前缀循环不保证首尾姿态平滑，不使用生成式补帧或循环混合。
用全动作周期包围盒留出边距，避免耳朵与尾巴出界；脚底锚点是周期内的稳定站立基准，不逐帧跟随抬脚。
连通背景策略优先保留被轮廓围住的内部绿色区域；边缘颜色与抗锯齿可能仍需要人工检查，不能承诺所有绿色形状自动正确。

如素材或处理参数变更，更新哈希与清单并重新测试；不要把旧尺寸下的脚底坐标带入新缓存。
预处理预览、运行中间文件和样片只存 `temp_data/`，不把用户上传素材纳入共享动画库。

## 原生像素重制（overlay-1.2.0）

`prepare_assets.py` 默认 `--max-height 0`：从 `assets/source/` 解码原始帧，沿用离线抠绿、dog_15安全前缀和dog_06绿色细节保护，按整个动作周期的联合裁切加透明边距，缓存裁切像素不缩小、不放大。正数参数仅用于主动制作更小缓存（上限8192），不会升采样原片。缓存画布可因虚拟透明边距超过源画幅，这不表示有更多角色细节。

清单 `resolution_provenance` 记录源路径/SHA256、解码分辨率、原生裁切分辨率、实际缓存分辨率、是否保留原生像素、max_height和锐化量。本次锐化量为0，优先保留原始轮廓、黑线和青绿发光细节。运行时缩小时用面积采样，放大时用Lanczos4；颜色和Alpha在预乘域采样，并对滤波振铃进行有效Alpha约束，避免隐藏绿/黑RGB混入透明边缘。

这属于非生成式高保真重制：避免旧384高缓存先降再升，保留源素材已有信息。无法恢复源视频已经丢失的纹理、压缩损坏、被裁掉的耳尾或不存在的动作帧；仍可能有原始色溢、细边和循环接缝。禁止描述为AI高清恢复、超分辨率细节生成或全周期无缝修复。


## 完整功能无损缓存格式维护（1.3.6-lossless）

这只是包内 NPZ 的 ZIP 编码维护，不是像素重制。引擎仍为 `overlay-1.3.5`；全部16份缓存、原视频、实验模型、既有测试与文档保留。缓存从 ZIP_DEFLATED（方法8）改为 ZIP_LZMA（方法14），原有每个 ZIP 成员的解压字节（包括 NPY 头、数组数据与附加成员）保持完全相同；不重新抠绿、不改裁切、分辨率、帧数、RGB/Alpha、锚点、颜色、选狗、选位或合成逻辑。NumPy 继续使用 `np.load(..., allow_pickle=False)`；Python 环境须支持标准库 `lzma`。

### 固定维护命令

在 Skill 根目录执行：

```bash
python3 scripts/lossless_cache.py
python3 scripts/test_overlay.py --lossless-only --report temp_data/lossless/test_results.json
```

可通过 `--assets-dir PATH --work-dir PATH --report PATH` 指定独立测试目录；报告必须位于 `work-dir` 内，工作目录须与资产在同一文件系统，以支持 `os.replace`。维护脚本只处理清单列出的 `assets/dogs/*.npz`，绝不修改源 MP4 或实验模型，不触碰外部 old/new 基准目录。完整回归由维护者另行安排，`--lossless-only` 不生成媒体测试输入、不运行视频渲染。

### 校验与替换护栏

1. 全清单预检：核对当前 NPZ 的清单 SHA256、字节数、ZIP 成员唯一性与 CRC；用 `allow_pickle=False` 读取全部 NPY 数组，验证 RGBA 的 dtype/shape 与清单相符。损坏、容器哈希/尺寸不匹配、已有 decoded hash 不匹配或对象数组一律拒绝，不自动重建修复。
2. 候选与备份只存 `temp_data/lossless/transaction-*`。逐成员保持顺序与解压字节 SHA256 相同；逐数组 dtype、shape、值完全一致；全部 RGBA（含透明像素下的 RGB）逐像素比较，并生成全帧与每帧 pixels SHA256。全部候选验证完成且确实减小时才开始替换。
3. 替换前再次核对原清单/全部缓存未被并发改动；每个缓存和清单通过 `os.replace` 原子安装。普通异常执行回滚；失败事务保留候选、原清单与尚存的备份供诊断。成功事务删除临时候选/备份。
4. 已为 LZMA 的文件仍执行完整校验，但不重复压缩；清单字段已一致时，不改缓存或清单内容、mtime。重复验证另存报告以保留首轮压缩量化：`python3 scripts/lossless_cache.py --report temp_data/lossless/idempotency_report.json`。

### 清单与报告约定

只修改每只狗的 `sha256`、`npz_bytes`，增加以下字段；其余原有元数据（包括历史处理时间与 pipeline signature）保持不变：

- `compression`：`container=npz`、`zip_method=ZIP_LZMA`、`zip_method_id=14`、`lossless=true`。
- `decoded_frames_sha256`：全部 uint8 RGBA 数组按 N,H,W,4 的 C 顺序拼接后的 SHA256，包括零 Alpha 下的 RGB，不含 NPY 头。
- `decoded_frames_hash_convention`：上述哈希口径说明。容器 SHA256 会改变，但解码像素 SHA256 不变。

`temp_data/lossless/compression_report.json` 包含每只狗原/新缓存字节数与 SHA256、原/新成员字节 SHA256、全部数组 dtype/shape/值一致标记、全部 RGBA 一致标记、逐帧 pixels SHA256、两侧完整读取/CRC/NumPy 解码耗时与比较耗时、单狗耗时、总字节数、节省比例及总耗时。读取计时包括逐成员校验及 NumPy 加载，不等同于单独一次 loader 的延迟，也不是冷启动跨平台性能承诺。

后续 `prepare_assets.py` 新生成缓存默认使用同一个 LZMA writer，禁止 pickle，并在原子安装前比较内存 RGBA 与读取结果；图像处理参数及处理流程不变。其 pipeline signature 原有算法包含预处理脚本全文 SHA256，因此升级脚本后重新运行预处理可能触发重建：单纯压缩现有缓存应运行维护脚本，避免重新解码/抠绿。默认 NPZ 临时输出在 Skill 的 `temp_data/lossless`；自定义输出若位于另一文件系统，使用该输出目录下的 `temp_data/lossless` 以保持原子替换。

### 操作边界与故障恢复

维护期间停止渲染、预处理及其他资产写入任务；锁仅防止同一 `work-dir` 的维护脚本并发运行，不会锁住 loader 或其他工作目录的进程。多文件替换不是全包的一次原子切换：强制终止/断电可能留下部分新缓存与旧清单，禁止此时继续渲染或强行重算清单哈希。先对照保留的 `transaction-*/manifest-original.json`、按清单顺序编号的 `backup-N.npz` 及已验证原版基准恢复一致版本并复核；确认无运行进程后再移除残留 `maintenance.lock`、重新维护。磁盘 I/O 故障导致回滚失败时保留现场，人工恢复，不能声称自动恢复完成。

LZMA 提高压缩率但会增加压缩 CPU、内存及部分加载开销；维护过程同时暂存候选与原缓存备份，应预留磁盘空间。所有临时报告不纳入技能发布资产。当前改动仅在本地草稿生效，等待复测确认；不执行 GitHub push、AIME Skill upload 或本地提交。
