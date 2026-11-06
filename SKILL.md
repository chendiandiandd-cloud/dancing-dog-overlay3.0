---
name: dancing-dog-overlay
description: 为上传的图片、视频或多个混合素材快速叠加预制跳舞卡通小狗，不调用生成模型。图片默认输出5秒MP4，视频保留原时长和原音轨；自动选择24%–35%大小与内侧落点，检测切镜后换狗，每镜头固定位置。适用于“加跳舞小狗”“照片变小狗跳舞视频”“多素材套小狗特效”“每镜头换狗”等需求。
author: 'chenyanqi.1008'
metadata:
  version: '1.4.1'
---

# 跳舞小狗叠加 · 快版

## 默认效果

- 多输入逐文件输出，不拼接；图片默认5秒，视频保留原时长和原音轨，小狗静音。
- 素材库保留15只透明动画狗，已删除 `dog_06`，其余编号不重排。每个检测镜头一只，相邻镜头不同，镜头内固定位置，不做运动跟随或瞬移。
- 自动尺寸候选为35%/30%/24%，完整动作画布占高范围24%–35%（允许1像素取整差异）；人工 `--height-ratio` 同样限定0.24–0.35。优先内侧留白，轻量避让检测到的人脸/文字；不保证漏检或人物运动后始终无重叠。
- 人工禁区和完整动作画布边界是硬约束，无可用位置或边界迫使尺寸低于24%时明确失败，不用缩小掩盖失败。不会删除原片已有的小狗。
- 保留剩余RGBA像素和高质量透明边缘缩放；不重新抠绿、不下载素材、不调用视觉大模型。

## 一次执行

从本Skill根目录运行[固定入口](scripts/render_overlay.py)，使用本轮真实附件路径：

```bash
python3 scripts/render_overlay.py --inputs '/path/photo.jpg' '/path/video.mp4' --output-dir temp_data/output --seed 42 --quality hd
```

仅当本次请求已授权部分遮挡时，追加 `--allow-special-fallback`；未授权不加该参数。

命令直接完成分析、选位、合成与基础验收。内部由[合成核心](scripts/overlay_core.py)、[分镜检测](scripts/shot_detection.py)、[轻量检测](scripts/lightweight_vision.py)执行，不需要先读取这些实现。视频一次低分辨率扫描完成切镜与代表帧采样，每镜头只用少量帧选位，再一次解码合成；不做全片身体/衣服保护并集、光流跟随、精细衣服分割或多级重试。

- 输出、失败原因、镜头计划、实测耗时从该次 `summary.json` 和逐文件JSON读取；交付实际MP4链接。部分失败仍交付成功项。
- 常规生成不跑全套回归、不复测历史素材、不维护缓存、不发布GitHub。只有明确要求优化/测试/发布时才做额外动作。
- 依赖为ffmpeg、ffprobe、Python标准库lzma与[Python依赖](scripts/requirements.txt)。环境已可用时直接运行，不每次重复安装；确实缺包时安装：`python3 -m pip install -r scripts/requirements.txt`。
- 15个缓存由[素材清单](assets/dog_manifest.json)索引，缺失或损坏不自动补造。

## 按需调整

保留 `--image-duration`、`--dog-id`、`--anchor X Y`、`--height-ratio`、`--avoid-region X1 Y1 X2 Y2`、`--cuts '1.5,3.2'`、`--plan-only`、`--quality preview|hd|source`。未知参数才查看 `--help`，不猜选项。人工切点只用于对应视频；同批人工禁区共用，区域不同则分别调用。更换seed重新随机。

`--dog-id dog_06` 或 `--dog-sequence` 包含已删除的编号会明确报错，不自动替换。删除素材后相同seed的随机序列可能变化，不能再用旧15/16只库的镜头序列直接复现。

`--allow-special-fallback` 表示允许部分遮挡的快路径策略，不再逐级放松保护；未授权时仍保留轻量严格避让。旧 `--tracking auto|flow` 和语义分割不属于快版，不可宣称继续支持。

素材来源、缓存画质和移出功能见[素材说明](references/asset-provenance.md)。版本迁移与验收边界见[使用与验收](references/usage-and-validation.md)。
