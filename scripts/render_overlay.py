#!/usr/bin/env python3
"""Dancing dog overlay CLI. Offline, single worker, one lossy video encode.

Examples (run from skill directory):
  python scripts/render_overlay.py --inputs photo.webp clip.mp4 --output-dir out --seed 42
  python scripts/render_overlay.py --inputs clip.mp4 --output-dir out --cuts "0.8,1.6" --plan-only
  python scripts/render_overlay.py --inputs photo.jpg --output-dir out --anchor .8 .9 --quality hd
  python scripts/render_overlay.py --self-test

Every invocation creates output-dir/task_<time>_<uuid>/ with per-input MP4 and
JSON, plus summary.json. output-dir/summary.json is an atomic latest-completed
summary copy (concurrent calls: last completed wins; isolated task reports stay).
JSON top level: input, output, status, shots, timings, warnings. Shot anchor is
normalized actual foot point; height_ratio is actual canvas height/frame height.
Only ONE added dog per output frame; existing dogs are not detected or removed.
Manifest loop_start_frame/loop_end_frame are zero-based [start,end), end EXCLUSIVE.
Default runtime stops before quality.suspicious_color_jump_frames; NPZ stays intact.
Shot JSON records effective/source dog frame counts and actual loop bounds.
--dog-id specifies the first shot; later shots still use non-repeating shuffle bag.
Images, including animated WebP/GIF, use the EXIF-transposed first frame. No
music means still-image output is silent, video preserves its first audio track.
Tracking uses offline sparse LK/RANSAC translation of a 2-D foot point, not 3-D.
Automatic cuts observe all effective CFR frames (1–60fps), with 0.25s confirmation;
reported frame precision is not original VFR PTS accuracy. No generated models.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from overlay_core import (ENGINE_VERSION, DogLibrary, PipeProcess, PreparedDog,
                          ShuffleBag, analyze, decode_image, image_rgb, media_info,
                          normalize_cuts, placement, probe, rate, run, save_json,
                          sha256, video_frames)

from motion_tracking import plan_tracking, tracking_offset

SKILL_ROOT = Path(__file__).resolve().parents[1]
BASE_WARNINGS = [
    '每帧仅新增一只狗；不检测或清除原画面已有狗，总可见狗数可能超过一只。',
    '位置为经典启发式；全片采样保护仍可能漏检，不是语义地面理解或三维跟踪。',
    '视频按有效源fps（1–60范围）转换CFR；VFR原始时间戳不保留，时长可能相差约一个输出帧。',
    '画幅按quality限制只缩小不放大；奇数宽高右侧/底部补1像素以兼容yuv420p。',
]


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--inputs', nargs='+', metavar='FILE', help='图片/视频，每文件独立输出；最多64个')
    p.add_argument('--output-dir', type=Path, help='结果目录，自动创建隔离任务子目录')
    p.add_argument('--seed', type=int, default=0, help='确定性随机种子（默认0）')
    p.add_argument('--image-duration', type=float, default=5., help='静态图片短片秒数（默认5）')
    p.add_argument('--quality', choices=['preview', 'hd', 'source'], default='preview')
    p.add_argument('--placement-mode', choices=['auto', 'corner'], default='auto')
    p.add_argument('--dog-id', help='指定首镜头的狗；后续镜头仍不重复换狗')
    p.add_argument('--height-ratio', type=float, help='手动请求狗画布高度比例；默认auto完整动作画布高0.24–0.40，不自动缩至24%%以下，优先安全大号/内侧；显式尺寸/anchor及corner保留原行为，必要缩小fit')
    p.add_argument('--anchor', nargs=2, type=float, metavar=('X', 'Y'), help='归一化脚底；不移动，必要时缩小，否则报错')
    p.add_argument('--avoid-region', nargs=4, type=float, action='append', default=[],
                   metavar=('X1','Y1','X2','Y2'), help='人工禁区归一化矩形，可重复；严格不交，不绕过检测保护')
    p.add_argument('--cuts', help='人工切点秒数逗号列表，覆盖自动检测；传空字符串表示整片一个镜头')
    p.add_argument('--plan-only', action='store_true', help='执行分析/资产验证/计划，不编码MP4')
    p.add_argument('--music', type=Path, help='替换原音轨的音频，循环/裁切至视频长度，编码AAC')
    p.add_argument('--tracking', choices=['auto', 'off', 'flow'], default='auto', help='auto可靠surface才跟随；flow尝试二维LK/RANSAC，失败静止回退；off固定')
    p.add_argument('--timeout', type=float, default=600., help='每个FFmpeg阶段的最大秒数（默认600）')
    p.add_argument('--protection-backend', choices=['classical', 'semantic'], default='classical',
                   help='classical保留草稿路径；semantic试验随包离线DeepLab，不生成画面；固定点无留白时失败，不瞬移')
    p.add_argument('--allow-special-fallback', action='store_true', help='用户允许时才启用：特殊镜头关闭自动保护并允许部分遮挡；24%%–40%%不缩小，保留人工禁区/边界')
    p.add_argument('--self-test', action='store_true', help='在系统临时目录创建mock assets并运行离线sanity测试')
    return p


def special_fallback(rgb, info, dog, rng, args, extra_frames):
    """Explicit user-authorized local fallback; never bypass manual keep-outs."""
    failures = []
    # Keep the normal joint size/inset search: explicit per-size calls would
    # silently disable inset preference and make the enlarged fallback hug edges.
    ratio = getattr(args, 'height_ratio', None)
    auto_inset = ratio is None and args.anchor is None and args.placement_mode == 'auto'
    choices = []
    for policy in ('body-relaxed', 'detected-relaxed'):
        try:
            pos = placement(rgb, info, dog, ratio, args.placement_mode, rng,
                            args.anchor, extra_frames=extra_frames,
                            avoid_regions=getattr(args, 'avoid_region', []),
                            protection_policy=policy)
            pos['special_scene_fallback'] = True
            pos['partial_overlap_accepted'] = True
            pos['requested_fallback_height'] = pos['evaluated_height_ratio']
            choices.append(pos)
            # Keep face/text protection when a roughly 6%-inset large dog fits.
            # If it forces a very edge-bound sticker, the user's authorization
            # permits relaxing automatic protections instead of shrinking.
            if not auto_inset or pos['minimum_canvas_edge_margin'] >= .055:
                return pos
        except ValueError as exc:
            failures.append(str(exc))
    if choices:
        return min(choices, key=lambda p: (
            max(0., .055-p['minimum_canvas_edge_margin']), p['joint_cost']))
    raise ValueError('特殊兜底仍无合法边界/人工禁区落点: ' + '; '.join(failures[-2:]))


def validate_args(args, p):
    from overlay_core import validate_avoid_regions
    try:
        args.avoid_region = validate_avoid_regions(args.avoid_region)
    except ValueError as exc:
        p.error(str(exc))
    if not math.isfinite(args.image_duration) or args.image_duration <= 0:
        p.error('--image-duration必须为有限正数')
    if args.height_ratio is not None and (not math.isfinite(args.height_ratio) or not 0 < args.height_ratio <= 1):
        p.error('--height-ratio必须在(0,1]')
    if args.anchor is not None and any(not math.isfinite(x) or not 0 <= x <= 1 for x in args.anchor):
        p.error('--anchor每个坐标须在[0,1]')
    if not math.isfinite(args.timeout) or args.timeout < 1:
        p.error('--timeout必须为至少1秒的有限数')
    if args.cuts is not None:
        try:
            args.cut_values = [] if not args.cuts.strip() else [float(c.strip()) for c in args.cuts.split(',')]
            if any(not math.isfinite(x) for x in args.cut_values):
                raise ValueError()
        except ValueError:
            p.error('--cuts须为逗号分隔的有限秒数，例如"0.8,1.6"')
    else:
        args.cut_values = None
    if not args.self_test:
        if not args.inputs or args.output_dir is None:
            p.error('须提供--inputs和--output-dir')
        if len(args.inputs) > 64:
            p.error('单worker有界批次最多64个输入；请分批调用')
        if args.music is not None and not args.music.is_file():
            p.error('--music文件不存在')


def create_plan(path, output, info, analysis, library, args, digest):
    rng_seed = int(hashlib.sha256(f'{ENGINE_VERSION}:{args.seed}:{digest}'.encode()).hexdigest(), 16)
    rng = random.Random(rng_seed)
    bag = ShuffleBag(library.entries, rng)
    if args.dog_id and args.dog_id not in library.entries:
        raise ValueError(f'未知dog-id: {args.dog_id}')
    ratio = args.height_ratio  # None means joint adaptive size/location search.
    plan = dict(schema_version=1, engine_version=ENGINE_VERSION, input=str(path), output=str(output),
                status='planned', shots=[], timings={}, warnings=list(BASE_WARNINGS),
                seed=args.seed, rng_input_key=digest, media=info, tracking=args.tracking,
                requested_height_ratio=ratio, requested_anchor=args.anchor,
                requested_cuts=args.cut_values, requested_avoid_regions=getattr(args,'avoid_region',[]),
                placement_assistance=('manual_anchor' if args.anchor is not None else
                                      'user_avoid_regions' if getattr(args,'avoid_region',[]) else 'none'),
                analysis={k: v for k, v in analysis.items() if k not in ('shots', 'signature')},
                audio_policy='music replaces first source audio' if args.music else
                             ('first source audio preserved; copy compatible codec else AAC' if info['audio'] else 'silent'),
                music=str(args.music.resolve()) if args.music else None,
                plan_only=args.plan_only)
    if info.get('animated_first_frame_only'):
        plan['warnings'].append('动画图片仅使用首帧，输出静态背景短片。')
    plan['warnings'].extend(analysis['limitations'])
    fps = info['fps']
    effective = []
    for shot in analysis['shots']:
        start = math.ceil(shot['start'] * fps - 1e-7) / fps
        if effective and (start - effective[-1]['start'] < .25 - 1e-6 or info['duration'] - start < .25 - 1e-6):
            plan['warnings'].append(f'切点{shot["start"]:.6f}s在CFR量化后镜头太短，已合并。')
            continue
        effective.append(dict(start=start, representative_png=shot['representative_png'],
                              temporal_samples=shot.get('temporal_samples', [])))
    if len(effective) > 1 and len(library.entries) < 2:
        raise ValueError('多镜头至少需要两只可用狗，才能保证相邻不同')
    body_windows = [[] for _ in effective]
    body_unions = [None] * len(effective)
    use_semantic = getattr(args, 'protection_backend', 'classical') == 'semantic'
    plan['protection_backend'] = 'semantic' if use_semantic else 'classical'
    if use_semantic:
        from semantic_protection import protector, MODEL_SHA256
        segmenter = protector()
        plan['semantic_protection'] = dict(model='DeepLab-v3', sha256=MODEL_SHA256,
                                          labels=['person','cat','dog'], offline=True,
                                          sampling_fps=min(6.,fps), temporal_relocation=False)
    if info['kind'] == 'video' and use_semantic:
        factor = min(1.,320/max(info['display_width'],info['display_height']))
        bw,bh = max(1,round(info['display_width']*factor)),max(1,round(info['display_height']*factor))
        body_unions = [np.zeros((bh,bw),np.uint8) for _ in effective]
        counts = [0] * len(effective)
        stride = max(1,round(fps/6))
        cut_frames = {round(s['start']*fps)+d for s in effective for d in (-1,0,1)}
        index = 0
        body_frames = video_frames(path,info,bw,bh,fps,args.timeout)
        try:
            for frame_index,frame in enumerate(body_frames):
                t = frame_index/fps
                while index+1<len(effective) and t>=effective[index+1]['start']:
                    index += 1
                if frame_index % stride and frame_index not in cut_frames:
                    continue
                body_unions[index] |= segmenter.mask(frame)
                counts[index] += 1
        finally:
            body_frames.close()
        plan['semantic_protection'].update(sample_counts=counts,
            shot_union_fractions=[float(m.mean()) for m in body_unions])
        plan['body_protection_fps'] = fps/stride
        plan['warnings'].append('离线DeepLab person/cat/dog分割约6fps与切点邻帧保护；不用宽泛肤色/衣服并集，不瞬移换位。分割与采样仍可能漏检；真实主体无留白时失败或明确缩小。')
    elif info['kind'] == 'video':
        from classical_vision import body_cue_mask
        factor = min(1.,320/max(info['display_width'],info['display_height']))
        bw,bh = max(1,round(info['display_width']*factor)),max(1,round(info['display_height']*factor))
        body_unions = [np.zeros((bh,bw),np.uint8) for _ in effective]
        index = 0
        body_frames = video_frames(path,info,bw,bh,info['fps'],args.timeout)
        try:
            for frame_index,frame in enumerate(body_frames):
                t = frame_index/info['fps']
                while index+1<len(effective) and t>=effective[index+1]['start']:
                    index += 1
                from overlay_core import face_boxes
                if frame_index % max(1,round(info['fps']*.5)) == 0 or t == effective[index]['start']:
                    body_faces, _ = face_boxes(frame)
                cue = body_cue_mask(frame, body_faces)
                body_unions[index] |= cue
                if info['duration'] <= 600:
                    windows=body_windows[index]
                    if not windows or t-windows[-1]['start'] >= .5-1e-7:
                        windows.append(dict(start=t,mask=np.zeros_like(cue),frames=[],end=t))
                    window=windows[-1]
                    window['mask'] |= cue
                    if frame_index % max(1,round(info['fps']/6)) == 0 or not window['frames']:
                        window['frames'].append(frame.copy())
                    window['last']=frame.copy()
                    window['end']=(frame_index+1)/info['fps']
        finally:
            body_frames.close()
        plan['body_protection_fps'] = info['fps']
        plan['warnings'].append('全CFR帧肤色/相邻中性衣服连通保护补充稀疏Haar/HOG；仍可能误检暖色背景或漏检深色皮肤/彩色衣服，不承诺零遮挡。')
    all_temporal_samples=[sample for original in analysis['shots'] for sample in original.get('temporal_samples',[])]
    for i, shot in enumerate(effective):
        end=effective[i+1]['start'] if i+1<len(effective) else info['duration']
        shot['temporal_samples']=[sample for sample in all_temporal_samples if shot['start']<=sample['time']<end]
        if i == 0 and args.dog_id:
            dog_id = args.dog_id
            # Fill bag excluding chosen first dog, without discarding a bag entry.
            bag.bag = [d for d in bag.ids if d != dog_id]
            rng.shuffle(bag.bag)
            bag.previous = dog_id
        else:
            dog_id = bag.next()
        dog = library.load(dog_id)
        representative = decode_image(shot['representative_png'])
        if info['kind'] == 'image':
            source_rgb = image_rgb(path)
            factor = min(1., 640 / max(source_rgb.shape[:2]))
            representative = cv2.resize(source_rgb, (max(1, round(source_rgb.shape[1] * factor)),
                                                      max(1, round(source_rgb.shape[0] * factor))),
                                         interpolation=cv2.INTER_AREA)
        extra_frames = [decode_image(s['png']) for s in shot['temporal_samples']] if info['kind']=='video' else []
        segments = []
        try:
            try:
                pos = placement(representative, info, dog, ratio,
                                args.placement_mode, rng, args.anchor, extra_frames=extra_frames,
                                avoid_regions=getattr(args,'avoid_region',[]), extra_mask=body_unions[i],
                                semantic_protection=use_semantic)
            except ValueError as whole_error:
                if getattr(args, 'allow_special_fallback', False) or args.anchor is not None or args.placement_mode != 'auto' or len(body_windows[i]) < 2:
                    # Authorized fallback keeps one fixed dog per shot instead
                    # of forcing half-second jumps between tiny safe windows.
                    raise
                for window in body_windows[i]:
                    try:
                        wp=placement(window['frames'][0],info,dog,ratio,args.placement_mode,rng,
                                     extra_frames=window['frames'][1:]+[window['last']],
                                     avoid_regions=getattr(args,'avoid_region',[]),extra_mask=window['mask'])
                    except ValueError as window_error:
                        window_error.diagnostics=dict(whole_shot=getattr(whole_error,'diagnostics',{}),
                                                      window_start=window['start'],window_end=window['end'],
                                                      window=getattr(window_error,'diagnostics',{}))
                        raise
                    wp.update(placement_mode='screen_sticker',ground_anchored=False,contact_shadow=False)
                    segments.append(dict(start=window['start'],end=min(end,window['end']),
                                         frame_start=round(window['start']*fps),placement=wp))
                pos=segments[0]['placement']
                plan['warnings'].append('整镜头保护并集无固定安全落点：使用0.5s局部全CFR保护窗口重定位同一只狗；不新增切镜、不丢弃短时身体保护，窗口边界瞬时换位（非光流/平滑跟随）。')
        except ValueError as exc:
            if getattr(args, 'allow_special_fallback', False):
                pos = special_fallback(representative, info, dog, rng, args, extra_frames)
                segments = []
                pos['strict_failure_reason'] = str(exc)
            else:
                failure=dict(start=shot['start'],end=end,dog_id=dog_id,status='failed',
                             protection_diagnostics=getattr(exc,'diagnostics',{}))
                plan['shots'].append(failure)
                plan['first_failed_shot']=failure
                exc.plan=plan
                raise
        phase = rng.randrange(dog['frame_count'])
        plan['shots'].append(dict(start=shot['start'],
                                  end=effective[i + 1]['start'] if i + 1 < len(effective) else info['duration'],
                                  dog_id=dog_id, anchor=pos['anchor_normalized'],
                                  height_ratio=pos['actual_height_ratio'], phase_frames=phase,
                                  frame_start=round(shot['start'] * fps),
                                  dog_fps=dog['fps'], dog_frame_count=dog['frame_count'],
                                  dog_source_frame_count=dog['source_frame_count'],
                                  dog_loop_start_frame=dog['loop_start_frame'], dog_loop_end_frame=dog['loop_end_frame'],
                                  dog_sha256=dog['manifest']['sha256'], placement=pos,
                                  placement_segments=segments, temporal_placement_fallback=bool(segments)))
        plan['warnings'].extend(dog['warnings'])
        plan['warnings'].extend(pos['warnings'])
    t = time.perf_counter()
    plan_tracking(path, plan, args.timeout)
    plan['tracking_analysis_seconds'] = time.perf_counter()-t
    plan['warnings'] = list(dict.fromkeys(plan['warnings']))
    return plan


def write_all(stream, data):
    view = memoryview(data)
    while len(view):
        n = stream.write(view)
        if not n:
            raise RuntimeError('FFmpeg编码器管道已关闭')
        view = view[n:]


def active_placement(shot, frame_number):
    segments=shot.get('placement_segments',[])
    if not segments:
        return shot['placement']
    active=segments[0]['placement']
    for segment in segments:
        if segment['frame_start'] > frame_number:
            break
        active=segment['placement']
    return active


def render_video(path, target, plan, library, timeout):
    info = plan['media']
    W, H, fps = info['output_width'], info['output_height'], info['fps']
    # Temporary encoded video, then stream-copy mux; only one lossy video encode.
    encoded = target.with_suffix('.video.part.mp4')
    muxed = target.with_suffix('.mux.part.mp4')
    encoder = None
    frames = None
    timings = dict(decode_overlay_encode=0., mux=0., verify=0.)
    started = time.perf_counter()
    n = 0
    try:
        encoder = PipeProcess(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '1',
                               '-filter_threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                               '-s:v', f'{W}x{H}', '-r', f'{fps:.10f}', '-i', 'pipe:0', '-an',
                               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf',
                               '23' if info['quality'] == 'preview' else '20',
                               '-pix_fmt', 'yuv420p', '-threads', '1', '-movflags', '+faststart', encoded],
                              timeout=timeout, writer=True)
        if info['kind'] == 'image':
            base = cv2.resize(image_rgb(path), (info['display_width'], info['display_height']), interpolation=cv2.INTER_AREA)
            base = cv2.copyMakeBorder(base, 0, H - base.shape[0], 0, W - base.shape[1], cv2.BORDER_CONSTANT)
            count = max(1, math.ceil(info['duration'] * fps - 1e-7))
            frames = (base.copy() for _ in range(count))
        else:
            frames = video_frames(path, info, info['display_width'], info['display_height'], fps, timeout, pad=True)
        shot_index, prepared = -1, None
        for n0, frame in enumerate(frames):
            current = max(0, shot_index)
            while current + 1 < len(plan['shots']) and n0 >= plan['shots'][current + 1]['frame_start']:
                current += 1
            if current != shot_index:
                shot_index = current
                shot = plan['shots'][current]
                prepared = PreparedDog(library.load(shot['dog_id']), shot['placement'])
            shot = plan['shots'][current]
            active=active_placement(shot,n0)
            if prepared.position is not active:
                prepared=PreparedDog(library.load(shot['dog_id']),active)
            index = (shot['phase_frames'] + math.floor((n0 - shot['frame_start']) * shot['dog_fps'] / fps + 1e-7)) % shot['dog_frame_count']
            dx,dy = tracking_offset(shot.get('motion',{}), n0/fps)
            prepared.overlay(frame, index, offset=(dx,dy))
            write_all(encoder.p.stdin, frame.tobytes())
            n = n0 + 1
        encoder.finish()
        if n == 0:
            raise RuntimeError('没有输出视频帧')
        timings['decode_overlay_encode'] = time.perf_counter() - started
        actual_duration = n / fps
        plan['encoded_frames'] = n
        plan['effective_duration'] = actual_duration
        if abs(actual_duration - info['duration']) > 1 / fps + .025:
            plan['warnings'].append('实际解码时长与容器声明时长差异超过一个输出帧；保留了所有有效解码帧，详见effective_duration。')
        plan['shots'] = [s for s in plan['shots'] if s['frame_start'] < n]
        for i, shot in enumerate(plan['shots']):
            shot['end'] = plan['shots'][i + 1]['start'] if i + 1 < len(plan['shots']) else actual_duration
        t = time.perf_counter()
        cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '1', '-filter_threads', '1', '-i', encoded]
        audio_source = plan['music'] or (str(path) if info['audio'] else None)
        if audio_source:
            if plan['music']:
                cmd += ['-stream_loop', '-1']
            cmd += ['-threads', '1', '-i', audio_source, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy']
            codec = info['audio']['codec_name'] if info['audio'] else None
            compatible = codec in {'aac', 'mp3', 'ac3', 'eac3', 'alac'} and not plan['music']
            audio_args = ['-c:a', 'copy'] if compatible else ['-c:a', 'aac', '-b:a', '192k', '-threads', '1']
            tail = ['-t', f'{actual_duration:.10f}', '-map_metadata', '-1', '-movflags', '+faststart', muxed]
            try:
                run(cmd + audio_args + tail, timeout)
                plan['audio_encoding'] = 'copy' if compatible else 'aac'
            except RuntimeError:
                if not compatible:
                    raise
                muxed.unlink(missing_ok=True)
                run(cmd + ['-c:a', 'aac', '-b:a', '192k', '-threads', '1'] + tail, timeout)
                plan['audio_encoding'] = 'aac_fallback'
                plan['warnings'].append('源音轨copy mux失败，已转AAC；视频未重新编码。')
        else:
            encoded.replace(muxed)
            plan['audio_encoding'] = 'none'
        timings['mux'] = time.perf_counter() - t
        t = time.perf_counter()
        verify = probe(muxed)
        v = next(s for s in verify['streams'] if s['codec_type'] == 'video')
        a = [s for s in verify['streams'] if s['codec_type'] == 'audio']
        if (int(v['width']), int(v['height'])) != (W, H):
            raise RuntimeError('ffprobe核验：输出尺寸不一致')
        vf = int(v.get('nb_frames', n))
        if vf != n:
            raise RuntimeError(f'ffprobe核验：输出帧数{vf} != 送入帧数{n}')
        vd = float(v.get('duration') or verify['format']['duration'])
        if abs(vd - actual_duration) > 1 / fps + .025:
            raise RuntimeError('ffprobe核验：视频时长不一致')
        if bool(a) != bool(audio_source):
            raise RuntimeError('ffprobe核验：音轨缺失或意外存在')
        plan['verification'] = dict(width=W, height=H, frame_count=vf, video_duration=vd,
                                    format_duration=float(verify['format']['duration']),
                                    fps=rate(v.get('avg_frame_rate')), video_codec=v['codec_name'],
                                    audio_codecs=[s['codec_name'] for s in a])
        timings['verify'] = time.perf_counter() - t
        muxed.replace(target)
        return timings
    finally:
        try:
            if frames is not None and hasattr(frames, 'close'):
                frames.close()
        finally:
            try:
                if encoder is not None:
                    encoder.close()
            finally:
                encoded.unlink(missing_ok=True)
                muxed.unlink(missing_ok=True)


def batch(args, assets=None, cache=None):
    root = args.output_dir.expanduser().resolve()
    protected = {Path(p).expanduser().resolve() for p in args.inputs}
    if args.music:
        protected.add(args.music.expanduser().resolve())
    if (root / 'summary.json').resolve() in protected:
        raise ValueError('output-dir/summary.json与输入源/音乐路径冲突；请选择不同output-dir，源文件不会被覆盖')
    root.mkdir(parents=True, exist_ok=True)
    task_id = 'task_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12]
    task = root / task_id
    task.mkdir(exist_ok=False)
    assets = Path(assets) if assets is not None else SKILL_ROOT / 'assets'
    cache = Path(cache) if cache is not None else SKILL_ROOT / 'temp_data' / 'cache'
    started = time.perf_counter()
    summary = dict(schema_version=1, engine_version=ENGINE_VERSION, task_id=task_id,
                   task_dir=str(task), summary_path=str(task / 'summary.json'),
                   status='running', seed=args.seed, plan_only=args.plan_only,
                   worker_count=1, max_inputs=64, results=[], counts={}, timings={})
    save_json(task / 'summary.json', summary)
    for i, name in enumerate(args.inputs):
        item_start = time.perf_counter()
        path = Path(name).expanduser().resolve()
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in path.stem)[:64] or 'input'
        stem = f'{i + 1:03d}_{safe}_{hashlib.sha256(str(path).encode()).hexdigest()[:10]}'
        target, plan_path = task / (stem + '.mp4'), task / (stem + '.json')
        plan = dict(schema_version=1, engine_version=ENGINE_VERSION, input=str(path), output=str(target),
                    status='failed', shots=[], timings={}, warnings=list(BASE_WARNINGS), plan_path=str(plan_path))
        try:
            if not path.is_file():
                raise ValueError('输入文件不存在或不是普通文件')
            t = time.perf_counter()
            digest = sha256(path)
            plan['timings']['hash'] = time.perf_counter() - t
            t = time.perf_counter()
            info = media_info(path, args.image_duration, args.quality)
            plan['timings']['probe'] = time.perf_counter() - t
            t = time.perf_counter()
            analysis, hit = analyze(path, info, digest, cache, args.cut_values, args.timeout)
            plan['timings']['analysis'] = time.perf_counter() - t
            t = time.perf_counter()
            library = DogLibrary(assets)
            timings = plan['timings']
            plan = create_plan(path, target, info, analysis, library, args, digest)
            plan.update(plan_path=str(plan_path), input_sha256=digest, analysis_cache_hit=hit)
            plan['timings'] = timings
            plan['timings']['planning_assets'] = time.perf_counter() - t
            # Music validation is performed even in plan-only mode.
            if args.music:
                if not any(s.get('codec_type') == 'audio' for s in probe(args.music)['streams']):
                    raise ValueError('--music没有可用音轨')
            save_json(plan_path, plan)
            if not args.plan_only:
                plan['timings'].update(render_video(path, target, plan, library, args.timeout))
                plan['status'] = 'success'
        except Exception as exc:
            if hasattr(exc,'plan'):
                timings=plan['timings']
                plan=exc.plan
                plan['timings']=timings
            plan['status'] = 'failed'
            plan['error'] = dict(type=type(exc).__name__, message=str(exc))
            target.unlink(missing_ok=True)
        finally:
            plan['timings']['total'] = time.perf_counter() - item_start
            save_json(plan_path, plan)
        summary['results'].append(dict(input=str(path), output=str(target), plan_path=str(plan_path),
                                       status=plan['status'], timings=plan['timings'], error=plan.get('error')))
        success = sum(r['status'] in ('success', 'planned') for r in summary['results'])
        summary['counts'] = dict(total=len(summary['results']), success=success,
                                 failed=len(summary['results']) - success, requested=len(args.inputs))
        save_json(task / 'summary.json', summary)
        print(json.dumps({'status': plan['status'], 'input': str(path), 'output': str(target),
                          'plan': str(plan_path), 'error': plan.get('error')}, ensure_ascii=False), flush=True)
    summary['status'] = ('success' if summary['counts']['failed'] == 0 else
                         'partial_failure' if summary['counts']['success'] else 'failed')
    summary['timings']['total'] = time.perf_counter() - started
    save_json(task / 'summary.json', summary)
    save_json(root / 'summary.json', summary)
    print(json.dumps({'summary': str(root / 'summary.json'), 'task_summary': str(task / 'summary.json'),
                      'status': summary['status'], 'counts': summary['counts']}, ensure_ascii=False), flush=True)
    return (0 if summary['counts']['failed'] == 0 else 1), summary


def self_test():
    """System-temp mock contract. Does not read/write real assets or skill cache."""
    from PIL import Image
    from overlay_core import difference
    checks = []
    (SKILL_ROOT/'temp_data').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dog_overlay_sanity_',dir=SKILL_ROOT/'temp_data') as td:
        base = Path(td)
        assets = base / 'assets'
        (assets / 'dogs').mkdir(parents=True)
        entries = []
        for i, color in enumerate([(230, 60, 30), (30, 210, 70), (50, 70, 230)]):
            f = np.zeros((6, 40, 24, 4), np.uint8)
            for j in range(6):
                f[j, 3:38, 3:21, :3] = color
                f[j, 3:38, 3:21, 3] = 220
                f[j, 12:20, 1 + j % 3:4 + j % 3, 3] = 128
            name = f'dog_{i + 1:02d}'
            dest = assets / 'dogs' / (name + '.npz')
            np.savez_compressed(dest, frames=f)
            entries.append(dict(id=name, file=f'dogs/{name}.npz', fps=30, duration=.2,
                                frame_count=6, width=24, height=40, anchor=[12, 38],
                                dominant_color=list(color), sha256=sha256(dest), source='mock system temp'))
        save_json(assets / 'dog_manifest.json', dict(version=1, dogs=entries))
        image = base / 'photo.webp'
        Image.new('RGB', (320, 180), '#b4b4b4').save(image)
        exif_file = base / 'rotated.jpg'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (80, 40), '#ababab').save(exif_file, exif=exif)
        assert image_rgb(exif_file).shape[:2] == (80, 40)
        checks.append('Pillow EXIF rotation and WebP input')
        rng = random.Random(99)
        bag = ShuffleBag([e['id'] for e in entries], rng)
        sequence = [bag.next() for _ in range(30)]
        assert all(a != b for a, b in zip(sequence, sequence[1:]))
        assert all(len(set(sequence[i:i + 3])) == 3 for i in range(0, 30, 3))
        checks.append('deterministic shuffle bag, no adjacent duplicates')
        common = ['--output-dir', str(base / 'out'), '--seed', '123', '--placement-mode', 'corner', '--image-duration', '.5']
        args = parser().parse_args(['--inputs', str(image)] + common)
        validate_args(args, parser())
        code, first = batch(args, assets, base / 'cache')
        assert code == 0
        p1 = json.loads(Path(first['results'][0]['plan_path']).read_text())
        assert p1['verification']['frame_count'] == 15 and not p1['verification']['audio_codecs']
        assert p1['status'] == 'success'
        checks.append('real raw decode/ROI alpha/ultrafast encode/ffprobe image output, silent 15 frames')
        args.plan_only = True
        code, second = batch(args, assets, base / 'cache')
        p2 = json.loads(Path(second['results'][0]['plan_path']).read_text())
        assert code == 0 and p2['analysis_cache_hit'] and p1['shots'] == p2['shots']
        assert not Path(p2['output']).exists()
        checks.append('plan-only no MP4, cache hit, identical seed/placements/phases')
        # Two-shot synthetic video with a source AAC audio track.
        clip = base / 'cuts.mp4'
        run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-filter_complex_threads', '1',
             '-f', 'lavfi', '-i', 'color=c=gray:s=320x180:r=30:d=1',
             '-f', 'lavfi', '-i', 'color=c=blue:s=320x180:r=30:d=1',
             '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100:duration=2',
             '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]', '-map', '2:a',
             '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1', '-c:a', 'aac', clip], 60)
        args.inputs, args.plan_only = [str(clip), str(base / 'missing.jpg')], False
        code, third = batch(args, assets, base / 'cache')
        p3 = json.loads(Path(third['results'][0]['plan_path']).read_text())
        assert code == 1 and third['counts']['success'] == 1 and third['counts']['failed'] == 1
        assert len(p3['shots']) == 2, p3
        assert p3['shots'][0]['dog_id'] != p3['shots'][1]['dog_id']
        assert p3['verification']['frame_count'] == 60 and p3['verification']['audio_codecs'] == ['aac']
        assert abs(p3['shots'][1]['start'] - 1) < .1
        checks.append('automatic hard cut, per-shot dog, original AAC copy, full 60 frames, mixed-batch failure retained')
        # Validate same-coordinate displaymatrix autorotation and manual override.
        rotated = base / 'rotation.mp4'
        run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', clip, '-c', 'copy', '-metadata:s:v:0', 'rotate=90', rotated], 60)
        args.inputs, args.cut_values, args.anchor = [str(rotated)], [.7, 1.3], [.75, .85]
        code, fourth = batch(args, assets, base / 'cache')
        p4 = json.loads(Path(fourth['results'][0]['plan_path']).read_text())
        assert code == 0, p4
        assert (p4['verification']['width'], p4['verification']['height']) == (180, 320)
        assert len(p4['shots']) == 3
        assert all(abs(s['anchor'][0] - .75) < 1 / 180 for s in p4['shots'])
        checks.append('displaymatrix autorotation, manual comma-cut equivalent, normalized anchor, three shots')
        # Verify frame-level no-black / one-added-ROI on the uniform two-shot source.
        inf = media_info(Path(p3['output']), .5, 'source')
        source_info = media_info(clip, .5, 'source')
        output_frames = video_frames(Path(p3['output']), inf, 320, 180, 30)
        source_frames = video_frames(clip, source_info, 320, 180, 30)
        seen = 0
        try:
            for k, (out, src) in enumerate(zip(output_frames, source_frames)):
                assert float(out.mean()) > 10
                shot = p3['shots'][0 if k < 30 else 1]
                p = shot['placement']
                delta = np.max(np.abs(out.astype(np.int16) - src.astype(np.int16)), axis=2)
                mask = np.ones(delta.shape, bool)
                # H.264 16px block changes can spread outside exact ROI.
                x0, y0 = max(0, p['x'] - 16), max(0, p['y'] - 16)
                x1, y1 = min(320, p['x'] + p['width'] + 16), min(180, p['y'] + p['height'] + 16)
                mask[y0:y1, x0:x1] = False
                assert delta[mask].max(initial=0) < 20
                assert delta[p['y']:p['y'] + p['height'], p['x']:p['x'] + p['width']].max() > 20
                seen += 1
        finally:
            output_frames.close()
            source_frames.close()
        assert seen == 60
        checks.append('all 60 output frames nonblack; one changed ROI including cut boundary, no doubled overlay')
        # Flash should not create a new shot, even though RGB/hist jumps strongly.
        flash = base / 'flash.mp4'
        run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-filter_threads', '1',
             '-f', 'lavfi', '-i', 'color=c=gray:s=320x180:r=30:d=2',
             '-vf', "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,1,1.07)'",
             '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1', flash], 60)
        args.inputs, args.cut_values, args.anchor, args.plan_only = [str(flash)], None, None, True
        code, fifth = batch(args, assets, base / 'cache')
        p5 = json.loads(Path(fifth['results'][0]['plan_path']).read_text())
        assert code == 0 and len(p5['shots']) == 1
        checks.append('short full-frame flash suppressed; no-hard-cut full clip is one shot')
        # Music loop on a still image; its duration must not shorten the clip.
        music = base / 'short music.wav'
        run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi', '-i',
             'sine=frequency=660:duration=0.1', music], 60)
        args.inputs, args.plan_only, args.music = [str(image)], False, music
        code, sixth = batch(args, assets, base / 'cache')
        p6 = json.loads(Path(sixth['results'][0]['plan_path']).read_text())
        assert code == 0 and p6['verification']['frame_count'] == 15
        assert p6['verification']['audio_codecs'] == ['aac'] and p6['audio_encoding'] == 'aac'
        checks.append('music loops/replaces source audio, AAC mux preserves image duration')
        args.inputs, args.music = [str(base / 'missing.jpg'), str(base / 'also_missing.mp4')], None
        code, seventh = batch(args, assets, base / 'cache')
        assert code == 1 and seventh['status'] == 'failed' and seventh['counts']['failed'] == 2
        checks.append('all-bad batch returns nonzero and still writes summary/per-input failure JSON')
        # Explicit first dog selection, repeated input paths cannot collide.
        args.inputs, args.plan_only, args.dog_id, args.cut_values = [str(clip), str(clip)], True, 'dog_02', [.5, 1., 1.5]
        code, eighth = batch(args, assets, base / 'cache')
        assert code == 0 and eighth['results'][0]['output'] != eighth['results'][1]['output']
        p8 = json.loads(Path(eighth['results'][0]['plan_path']).read_text())
        assert p8['shots'][0]['dog_id'] == 'dog_02'
        assert all(a['dog_id'] != b['dog_id'] for a, b in zip(p8['shots'], p8['shots'][1:]))
        checks.append('forced first dog plus non-repeating later shots, duplicate-input naming isolation')
        # Classic auto placement must really use a Hough support candidate.
        horizon = np.full((180, 320, 3), 180, np.uint8)
        horizon[110:] = 80
        im_info = media_info(image, .5, 'preview')
        dog = DogLibrary(assets).load('dog_01')
        pos = placement(horizon, im_info, dog, .12, 'auto', random.Random(123))
        assert pos['kind'] == 'hough_horizontal_support', pos
        assert 0 <= pos['x'] <= 320 - pos['width'] and 0 <= pos['y'] <= 180 - pos['height']
        checks.append('auto placement uses real Hough line support, full union/canvas fits')
        # Actually decode a VFR stream, then preserve its effective duration in CFR.
        vfr = base / 'vfr.mp4'
        run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-filter_threads', '1',
             '-f', 'lavfi', '-i', 'testsrc2=s=320x180:r=30:d=1.5',
             '-vf', "setpts='if(lt(N,15),N/(15*TB),(1+(N-15)/30)/TB)'", '-fps_mode', 'vfr',
             '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1', vfr], 60)
        args.inputs, args.plan_only, args.dog_id, args.cut_values = [str(vfr)], False, None, []
        code, ninth = batch(args, assets, base / 'cache')
        p9 = json.loads(Path(ninth['results'][0]['plan_path']).read_text())
        assert code == 0, p9
        assert abs(p9['effective_duration'] - p9['media']['duration']) <= 1 / p9['media']['fps'] + .025
        assert p9['verification']['audio_codecs'] == []
        checks.append('real VFR to CFR output within one frame of declared duration, video without audio stays silent')
        large = base / 'large.webp'
        Image.new('RGB', (2001, 1001), '#aaaaaa').save(large)
        widths = {q: media_info(large, .5, q)['output_width'] for q in ('preview', 'hd', 'source')}
        assert widths == {'preview': 1280, 'hd': 1920, 'source': 2002}, widths
        checks.append('preview/hd/source longest-edge caps and explicit odd-size padding')
        # Public flow is now implemented; validate actual full-clip motion stats,
        # not merely acceptance of an argument or a silent no-op.
        import subprocess
        flow_run = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   '--inputs', str(vfr), '--output-dir', str(base/'flow'),
                                   '--tracking', 'flow', '--plan-only', '--cuts', ''],
                                  capture_output=True, timeout=60)
        assert flow_run.returncode == 0, flow_run.stderr
        flow_summary = json.loads((base/'flow'/'summary.json').read_text())
        flow_plan = json.loads(Path(flow_summary['results'][0]['plan_path']).read_text())
        fm = flow_plan['shots'][0]['motion']
        assert fm['attempted_pairs'] > 5 and fm['trajectory'] and fm['accepted_pairs'] > 0
        checks.append('public --tracking flow executes full-clip LK/RANSAC with real motion statistics')
        # Encoder watchdog should kill a blocked child instead of hanging.
        sleeper = PipeProcess([sys.executable, '-c', 'import time; time.sleep(5)'], timeout=.15, reader=True)
        t0 = time.perf_counter()
        try:
            assert sleeper.p.stdout.read(1) == b''
            try:
                sleeper.finish()
                raise AssertionError('watchdog should fail')
            except RuntimeError as exc:
                assert 'timeout' in str(exc)
            assert time.perf_counter() - t0 < 3
        finally:
            sleeper.close()
        checks.append('pipe watchdog timeout terminates blocked child and cleans up')
        # Mild equal-layout color cuts below the old 0.13 mean threshold,
        # plus a short same-strength color impulse that must remain one shot.
        mild = base / 'mild_cuts_and_color_flash.mp4'
        colors = [(150, 150, 150), (174, 135, 138), (148, 172, 151)]
        colors_rgb = [np.full((180, 320, 3), c, dtype=np.uint8) for c in colors]
        d, hist = difference(colors_rgb[0], colors_rgb[1])
        assert .035 < d < .13 and hist > .65
        enc = PipeProcess(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo',
                           '-pix_fmt', 'rgb24', '-s', '320x180', '-r', '30', '-i', 'pipe:0',
                           '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1', mild],
                          timeout=30, writer=True)
        try:
            for i in range(90):
                which = 1 if i in (15, 16) else i // 30
                write_all(enc.p.stdin, colors_rgb[which].tobytes())
            enc.finish()
        finally:
            enc.close()
        minfo = media_info(mild, 3, 'source')
        detected, _ = analyze(mild, minfo, sha256(mild), base / 'cache')
        assert [s['start'] for s in detected['shots']] == [0., 1., 2.], detected
        assert all(e['rule'] == 'adaptive_color_jump' for e in detected['cut_events'])
        checks.append('adaptive mild-color cuts at 1/2s detected; equally strong short color flash suppressed')
        # Runtime loop bounds are metadata only; source integrity and N remain.
        mock_manifest = dict(version=1, dogs=[dict(e) for e in entries])
        mock_manifest['dogs'][0]['quality'] = dict(suspicious_color_jump_frames=[4])
        save_json(assets / 'dog_manifest.json', mock_manifest)
        limited = DogLibrary(assets).load('dog_01')
        assert limited['source_frame_count'] == 6 and limited['frame_count'] == 4
        assert limited['loop_end_frame'] == 4 and limited['warnings']
        assert sha256(assets / entries[0]['file']) == entries[0]['sha256']
        mock_manifest['dogs'][0]['loop_end_frame'] = 3
        save_json(assets / 'dog_manifest.json', mock_manifest)
        limited = DogLibrary(assets).load('dog_01')
        assert limited['frame_count'] == 3 and limited['loop_end_frame'] == 3
        checks.append('suspicious-frame safe prefix and exclusive manifest loop_end_frame honored, NPZ untouched')
        # Generic shelf vs smooth horizon: do not assert a fixed real-photo coordinate.
        shelf = np.full((240, 400, 3), 170, np.uint8)
        shelf[100:180] = 195
        shelf[180:202] = 35
        shelf[202:] = 190
        shelf[180:, 42:48] = 35
        shelf[180:, 345:351] = 35
        sinfo = dict(display_width=400, display_height=240)
        sp = placement(shelf, sinfo, dog, .12, 'auto', random.Random(123))
        assert .70 < sp['anchor_normalized'][1] < .83, sp
        assert sp['support_geometry']['material_below_bonus'] > 0
        checks.append('generic shelf/vertical structure preferred over smooth horizon and shelf underside')
        # Transparent RGB must not bleed into interpolated edges.
        frames = np.zeros((1, 2, 2, 4), np.uint8)
        frames[0, :, :, :3] = [0, 255, 0]
        frames[0, 0, 0] = [255, 0, 0, 255]
        prep = PreparedDog(dict(frames=frames, frame_count=1), dict(x=0, y=0, width=8, height=8))
        color, alpha = prep.frame(0)
        assert float(color[:, :, 1].max()) == 0
        checks.append('premultiplied resize prevents hidden-RGB edge contamination')
    print(json.dumps({'self_test': 'passed', 'checks': checks, 'real_assets_used': False,
                      'performance_claim': 'none; timings in generated plans only'}, ensure_ascii=False, indent=2))
    return 0


def main():
    p = parser()
    args = p.parse_args()
    validate_args(args, p)
    if args.self_test:
        return self_test()
    try:
        code, _ = batch(args)
        return code
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({'status': 'fatal', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
