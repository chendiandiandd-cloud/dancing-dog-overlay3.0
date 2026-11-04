---
name: dancing-dog-overlay
description: 为用户上传的多张图片、多段视频或混合素材叠加预制跳舞卡通小狗，不调用图像或视频生成模型。支持离线透明动画缓存、尺寸与画面候选落点联合自适应、分镜检测与切镜换狗、种子复现、手动落点、保留原视频音轨和逐文件 MP4 输出。适用于“加一只跳舞小狗”“照片变小狗跳舞视频”“每个镜头换狗”“多素材批量套跳舞小狗特效”等需求。
author: 'chenyanqi.1008'
metadata:
  version: '1.3.5'
---

# 跳舞小狗叠加

当前引擎版本：`overlay-1.3.5`（草稿）。默认图片/视频auto完整动作画布占高24%–40%，不自动缩到24%以下。一镜头使用同一只狗。授权特殊模式优先整镜头固定落点，不用窗口瞬移掩盖失败；普通未授权视频仍保留既有0.5秒保护窗口重定位回退，可能瞬时换位，所有窗口同样遵守24%下限，无合法位置报错。

仅在用户授权特殊场景部分遮挡时添加 `--allow-special-fallback`：严格整镜头选位失败后，先放松身体/衣服保护，保留脸/文字；默认auto候选完整画布最小边距≥0.055（约6%）时采用该候选。若无合法位置或候选强贴边（边距<0.055），继续尝试放松所有自动保护以寻找内侧候选；两级均不满足内侧阈值时，在合法候选中优先较内侧者，再比较联合成本。保持24%–40%自动尺寸与内侧偏好，不强贴边、不退回15/12/9/6%。人工禁区与完整动作边界始终不可绕过。计划记录 `special_scene_fallback`、`partial_overlap_accepted`、`protection_policy` 和严格选位失败原因；授权允许遮挡不等于实际每帧都发生遮挡。显式height-ratio/anchor/corner保留旧行为例外。

v1.3.5完整回归71项：70通过、1跳过（缺真实8输入summary），无失败，64.466秒；CLI百分号帮助格式化问题已修复。最终两视频实测：猫5个检测镜头28%–40%、街拍4个检测镜头28%–36%；9段全部采用授权特殊保护策略且固定位置，无窗口重定位。最小完整画布边距：猫5.90%–6.60%，街拍6.57%–8.98%。逐帧解码/计划边界检查302+452帧通过；切点前后及中点共25张抽帧独立视觉核验无双狗、缺狗、明显绿边或画外裁切，街拍红狗贴边已改善。视频原音轨与时长保留。真实快剪仍有已知漏检，检测到的切点均换狗，不宣称所有真实切镜覆盖。历史试验与结果见 [使用与验收](references/usage-and-validation.md)。离线DeepLab-v3仅为显式 `--protection-backend semantic` 试验，不OR回旧bodymask；模型 `assets/deeplab_v3.tflite` 的来源/sha见 `assets/deeplab_v3.provenance.json`，依赖 `scripts/requirements-semantic.txt`，实现 [离线分割](scripts/semantic_protection.py)、证据 [分割诊断](scripts/diagnose_semantic.py)。既有误/漏分割及时间扫掠阻塞见 [分割可行性与准确阻塞](references/semantic-feasibility.md)，不宣称其已解决大号避让。


自动分镜采用最长边480低分辨率、有效源CFR全帧（1–60fps），联合RGB/HSV、空间亮度与边缘自适应跳变；候选持续/返回确认0.25秒，不调用生成模型。每个检测镜头换一只狗且相邻不同；低对比硬切、快速运动、渐变与闪光附近仍可能误/漏检。极亮/极暗帧及0.25秒恢复段保守合并，避免开头频繁换狗。具体采样率、切点精度及告警见本次计划；必要时人工`--cuts`覆盖。

## 效果约定

- 直接使用随包的 16 只动画小狗，合成成片，不调用生成模型，也不在运行时调用视觉大模型。
- 多文件默认独立输出；不把用户上传的素材擅自拼成一条视频。
- 每张图片默认独立生成 5 秒 MP4，视频保持原时长并保留原音轨，小狗素材静音。
- 每个分镜只新增一只小狗；相邻分镜使用不同狗，镜头内部不换狗。
- 不识别或清除输入素材中已经存在的小狗，参考效果视频不可当作没有贴纸的原片。
- 图片/视频默认auto联合评估尺寸与落点，完整动作画布高限定24%–40%；尺寸、留白成本与内侧程度软权衡，不强取最大号或中心。完整画布四边各至少约6%余量为内侧偏好；只有边缘合法时允许边缘fallback并warning，但不能因此缩到24%以下。宽动作受横向边界限制而达不到下限时拒绝。普通模式保留检测保护；授权特殊fallback按前述顺序放松自动保护，仍保留人工禁区/完整边界及内侧偏好。经典CV仍可能误/漏检，不承诺零遮挡。显式height-ratio/anchor/corner保留旧尺寸和选位例外。
- 自动落点补充肤色与相邻中性衣服保护：肤色闭运算前要求局部色度一致性，轮廓膨胀随连通组件尺度变化，中性衣服连接限定在肤色种子局部范围，减少暖色草叶/土壤桥接和全局连通误保护。视频仍最长边320全CFR帧mask并集，保留小臂/裙摆补漏；平滑暖色背景仍可能误保护，极暗/很小皮肤仍可能漏检，不承诺零遮挡。
- 自动落点使用全片代表帧Haar/HOG、局部GrabCut衣服保护、文字形态启发式及四角小保留区；大脸需二次眼部证据，避免宠物误检把全幅当人体。连续脚底跨度与时序支撑不足时选screen_sticker安全留白（默认优先适度内侧，必要边缘回退；非地面锚定、无落地阴影）。检测会漏检/误检，不承诺OCR理解、水面语义或真实遮挡。
- 素材从原始帧按原生像素重建，不再默认压到384高；预乘Alpha高质量采样减少缩放边缘污染，不生成缺失细节，也不宣称凭空高清恢复。
- 现阶段提供附件输入和脚本合成，没有独立的拖拽上传网页。

## 执行入口

固定入口为 [合成脚本](scripts/render_overlay.py)，其内部使用 [合成核心模块](scripts/overlay_core.py)、[流式多特征分镜检测](scripts/shot_detection.py)、[经典保护与光流](scripts/classical_vision.py) 和 [全片跟随规划](scripts/motion_tracking.py)。素材维护使用 [预处理脚本](scripts/prepare_assets.py)；回归测试由 [测试脚本](scripts/test_overlay.py) 调用 [测试输入生成器](scripts/make_test_inputs.py)，不需要在正常任务中运行测试输入生成器。

从本 Skill 根目录执行；附件使用本次请求真实存在的文件路径，不复用历史会话的临时路径。

1. 检查 `ffmpeg`、`ffprobe` 和 Python 包。缺 Python 包时执行：

```bash
python3 -m pip install -r scripts/requirements.txt
```

FFmpeg 需系统提供，缺失时明确报告，不把缺工具误报为渲染成功。

2. 确认 `assets/dog_manifest.json` 和其引用的透明缓存存在。完整包已包含处理后的缓存，常规运行不需要重新抠绿。

3. 对用户提供的真实附件直接调用固定入口：

```bash
python3 scripts/render_overlay.py --inputs '/path/photo.jpg' '/path/video.mp4' --output-dir temp_data/output --seed 42
```

指定明确的输出目录，返回命令生成的实际 MP4 链接。批次状态、镜头计划、告警、耗时从该次 JSON 结果读取；部分失败时仍交付成功文件并解释失败，不宣称全部成功。

## 常见调整

先用 `python3 scripts/render_overlay.py --help` 检查当前参数；不猜不存在的选项。

```bash
# 图片片长与高清模式
python3 scripts/render_overlay.py --inputs '/path/photo.png' --output-dir temp_data/hd --image-duration 5 --quality hd --seed 42

# 人工指定视频切点，单位为秒
python3 scripts/render_overlay.py --inputs '/path/video.mp4' --output-dir temp_data/cuts --cuts '1.5,3.2,5.0' --seed 42

# 指定脚底锚点与尺寸：坐标归一化，左上为(0,0)，右下为(1,1)
python3 scripts/render_overlay.py --inputs '/path/photo.jpg' --output-dir temp_data/manual --anchor 0.25 0.72 --height-ratio 0.12 --seed 42

# 人工区域辅助：可重复指定禁区；坐标按转正后的完整画面归一化
python3 scripts/render_overlay.py --inputs '/path/video.mp4' --output-dir temp_data/assisted --avoid-region 0 0.55 0.50 1 --seed 42

# 仅分析选位并输出计划，不渲染视频
python3 scripts/render_overlay.py --inputs '/path/video.mp4' --output-dir temp_data/plan --plan-only --seed 42
```

重新随机时改变 seed；复现时保留相同输入、参数、素材版本和 seed。`--avoid-region X1 Y1 X2 Y2`可重复：有限归一化坐标须满足0≤X1<X2≤1、0≤Y1<Y2≤1，严格避让完整动作画布及光流移动，不绕过真脸等检测保护；无安全位置报错。禁区来自人工确认时，成片标为人工区域辅助，不称纯自动；同批各文件共用禁区，区域不同需分别调用。手动切点只用于对应视频，不把一组切点默认套在长度不一致的批次。

`--dog-id` 只指定第一镜头的狗，后续镜头仍使用不相邻重复的洗牌袋。`--tracking auto|off|flow` 默认auto：全片保护采样保持；仅surface候选执行LK/RANSAC，且全片置信度/边界检查通过才二维跟随。screen_sticker固定并跳过光流，全片无候选时不增加光流解码；统计明确记录skipped_reason，未观测匹配率为null。flow显式尝试局部/相机光流，失败冻结在最后安全位置，切镜重置，不声称三维跟踪。手动anchor始终固定，可绕过四角小保留区但不能压检测人脸/文字；具体边界见使用与验收。

## 素材维护

运行时不下载线上素材。仅在主动维护素材库或缓存丢失时运行：

```bash
python3 scripts/prepare_assets.py --source-dir assets/source --output-dir assets
```

若重建缓存，检查棋盘预览和整个周期的黑轮廓、绿边、耳朵/尾巴裁切和脚底稳定性。透明效果是离线处理后的结果；输入源 MP4 本身没有 Alpha。素材版本与 sha256 来自清单，不因看起来相似而混用缓存。

## 交付检查

- 输出文件存在且可被 ffprobe 读取，画幅、时长、音轨符合本次参数。
- 镜头计划里每镜头一个 `dog_id`，相邻狗不同；切点附近没有额外叠加分支造成双狗。
- 尺寸约束针对完整动作周期，不只针对第一帧。
- 明确说明告警、退化选位、非生成式启发式的限制，不声称语义理解或真实三维跟踪。
- 只报告实测耗时，区分素材准备、分析、编码和文件传输。

运行回归：

```bash
python3 scripts/test_overlay.py
```

详细参数、测试基准与首版边界见 [使用与验收](references/usage-and-validation.md)。素材来源与处理约定见 [素材来源](references/asset-provenance.md)。
