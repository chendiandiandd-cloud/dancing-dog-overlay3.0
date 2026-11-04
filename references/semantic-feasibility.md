# 离线分割可行性结论（2026-11-04，未验收）

## 结论

**MediaPipe DeepLab-v3可运行，但两个新视频不能在当前固定落点、全时主体避让约束下达到24%以上成功。停止盲调，不交付成功成片。**

模型2,780,176字节，SHA256 `ff36e24d40547fe9e645e2f4e8745d1876d6e38b332d39a82f0bf0f5d1d561b3`。实际模型内嵌标签和运行时labels一致，person=15、cat=8、dog=12。CPU测试首帧约0.025秒（仅单次环境观测，不是性能保证），无需另建venv。来源与Apache-2.0发布信息见 `../assets/deeplab_v3.provenance.json`；依赖见 `../scripts/requirements-semantic.txt`。

试验路径替代肤色/GrabCut衣服mask及HOG/脸扩展人体矩形，**不会OR回旧bodymask**。保留有person语义支持的人脸框、文字/四角保留区、完整动作画布与人工禁区约束。语义mask有小轮廓保护边距。视频约6fps+CFR切点前后邻帧，代表帧补采。仍做整镜头固定位置保护并集，不投票删除短时主体、不0.5秒瞬移、不生成画面、不运行时联网。

## 两个真实case

| 素材 | 自动镜头数 / 切点数 | 自动切点(s) | 语义路径首失败 | 该镜头采样语义并集 | ≥24%全片成功 |
|---|---:|---|---|---:|---|
| 9月29日(2).mp4 | 5 / 4 | 2.4, 4.1667, 6.0667, 8.1667 | 2.4–4.1667s | 99.6589% | 否，失败；首镜头仅约8.98% |
| 测试效果demo-gpt.mp4 | 4 / 3 | 2.6333, 4.8, 7.0 | 0–2.6333s | 99.5278% | 否，首镜头已失败 |

猫近景2.8/3.2/3.6/4.0秒单帧保护分别93.06/97.82/97.81/98.77%，并非只由整镜头OR造成。可视对照中猫、双手确实占据大部分画幅；模型将部分猫毛误分为person，但person与cat都应保护，不能靠删除person类获取假留白。边缘背景也有分割误差，不能称模型完全准确；即便修正边缘，未证明存在24%完整动作画布空间。

时尚0/0.3/0.6/0.9/1.2秒单帧保护63.34/72.10/79.39/60.48/64.61%，人物横移、缩放与鱼眼/拼贴使整镜头并集饱和。1.5秒有楼体误分person；9秒拼贴画面也有漏检。自动切点仍非逐真实转场真值。不能将密集特效全当hardcut换狗，亦不能以降低mask阈值/短时投票绕过真主体。

**准确阻塞：轻量分割解决了背景肤色判别范式问题，但无法解决近满幅主体及时间扫掠造成的固定大号落点不存在；模型在特效/近景也有类别和边缘误差。** 后续若要继续，需要明确改变产品策略（允许扩大/加边画布、遮挡隐藏或受安全约束平滑动态位置），不再只调肤色或默认保护阈值。本次未实施这些策略。

## 实际mask与plan证据

- [猫原帧/红色语义mask对照](../temp_data/v134/semantic_evidence/01_semantic_evidence.jpg)
- [时尚原帧/红色语义mask对照](../temp_data/v134/semantic_evidence/02_semantic_evidence.jpg)
- [逐帧类别/覆盖率](../temp_data/v134/semantic_evidence/feasibility.json)
- [猫失败plan](../temp_data/v134/semantic_probe/task_20261104T152920Z_44b7d89047bc/001_9月29日_2__b49be1f4f8.json)
- [时尚失败plan](../temp_data/v134/semantic_probe/task_20261104T152920Z_44b7d89047bc/002_测试效果demo-gpt_c509586cda.json)

上述plan生成时分割为试验默认，现已隔离为显式 `--protection-backend semantic`；算法未调整，默认classical保留此前草稿。不要将默认classical当作两个新视频已成功。

## 旧素材与回归

语义试验草地IMG_0233：27.96875%、最小画布边距6.484375%；白裙IMG_9828：30%、边缘fallback；ByteArtist首镜头固定保护无安全位置，退化失败，未验收其Shot1/5/8身体保护。分镜算法未在分割阶段再调整，原14镜头分析仍在缓存中。

隔离试验选项后，现有63项完整回归：62通过、1跳过（需要独立真实成片summary），0失败/错误，60.358秒；self-test 20项通过。**这些是默认classical及既有功能回归，不代表语义真实case成功。** [回归报告](../temp_data/v134/semantic_regression.json)。未渲染真实成片，未提交、未push、未上传，未改inner_skills。

## 复现试验

```bash
python3 -m pip install -r scripts/requirements-semantic.txt
python3 scripts/render_overlay.py --inputs '/path/9月29日(2).mp4' '/path/测试效果demo-gpt.mp4' --output-dir temp_data/semantic --seed 42 --plan-only --protection-backend semantic
```

缺模型/依赖或sha/schema不符时显式失败，不联网下载、不静默切回肤色mask。实验代码 `scripts/semantic_protection.py`，有限证据脚本 `scripts/diagnose_semantic.py`；不是可发布的已完成1.3.4修复。
