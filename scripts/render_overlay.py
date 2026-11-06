#!/usr/bin/env python3
"""Offline one-dog-per-shot sticker compositor. Default HD; one lossy encode.

Each input has an isolated MP4/JSON, task summary and atomic latest summary.
Automatic cuts use longest-edge 480 effective source CFR (1-60fps). Placement
uses at most 3 frames from that SAME analysis pass; it does not track motion.
Detected face/text rectangles remain hard unless --allow-special-fallback is
explicitly authorized. Manual avoid regions and full cycle bounds stay hard.
No heuristic guarantees zero occlusion. Images default to 5 seconds and silent;
video keeps first source audio unless --music replaces it. VFR becomes CFR.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from overlay_core import (ENGINE_VERSION, RNG_SALT, DogLibrary, PipeProcess, PreparedDog,
                          ShuffleBag, analyze, decode_image, image_rgb, media_info,
                          placement, probe, rate, run, save_json, sha256,
                          validate_avoid_regions, validate_height_ratio, video_frames)

SKILL_ROOT = Path(__file__).resolve().parents[1]
BASE_WARNINGS = [
    '每帧仅新增一只狗；不检测或清除原画面已有狗。',
    '每镜头最多3帧轻量人脸/文字启发式；包含人体Proxy主体保护，防遮挡逻辑优化。',
    '地面锚定优先模式：检测水平支撑面（台阶/路面）并作为高优落点，无支撑时拒绝并返回诊断。',
    '有效源fps限制1–60并转换CFR，VFR逐帧PTS不保留；时长可相差约一个输出帧。',
    'quality只缩小不放大；奇数宽高在右/底补1像素。',
]


def parser():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--inputs',nargs='+',metavar='FILE',help='独立图片/视频，最多64个')
    p.add_argument('--output-dir',type=Path,help='结果目录，创建隔离任务子目录')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--image-duration',type=float,default=5.)
    p.add_argument('--quality',choices=['preview','hd','source'],default='hd')
    p.add_argument('--placement-mode',choices=['auto','corner'],default='auto')
    p.add_argument('--dog-id',help='指定第一镜头，后续镜头相邻不同')
    p.add_argument('--dog-sequence',help='逗号分隔每镜头dog id；复测可复用旧计划以确保相同狗；须与镜头数一致且相邻不同')
    p.add_argument('--height-ratio',type=float,help='完整狗画布请求高占比；默认35/30/24%%；仅接受[.24,.35]，边界fit低于24%%拒绝')
    p.add_argument('--anchor',nargs=2,type=float,metavar=('X','Y'),help='固定归一化脚底；边界必要时几何缩小，不移动；缩至24%%以下无合法位，拒绝')
    p.add_argument('--avoid-region',nargs=4,type=float,action='append',default=[],metavar=('X1','Y1','X2','Y2'),help='可重复人工禁区，完整动作画布严格不交')
    p.add_argument('--cuts',help='人工切点秒数逗号分隔，空字符串为一个镜头')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--music',type=Path,help='循环/裁切替换音频，编码AAC')
    p.add_argument('--timeout',type=float,default=600.)
    p.add_argument('--allow-special-fallback',action='store_true',help='默认关闭；用户显式授权后自动脸/文字改为一次软惩罚，可部分遮挡；人工禁区/完整画布仍hard')
    p.add_argument('--tracking',help='迁移兼容：仅接受off（deprecated）；auto/flow已移除')
    p.add_argument('--protection-backend',help='迁移兼容：仅接受lightweight（deprecated）；旧后端已移除')
    p.add_argument('--self-test',action='store_true',help='已移出运行包，传入报迁移错误')
    return p


def validate_args(args,p):
    if args.tracking not in (None,'off'):
        p.error('--tracking auto/flow已移除；移除此参数或使用off。新引擎整镜头固定screen-sticker。')
    if args.protection_backend not in (None,'lightweight'):
        p.error('--protection-backend旧后端已移除；移除此参数，使用默认lightweight。')
    if args.self_test:
        p.error('--self-test已移出轻量运行包；开发回归由独立测试入口执行。')
    try:
        args.avoid_region = validate_avoid_regions(args.avoid_region)
    except ValueError as exc:
        p.error(str(exc))
    if not math.isfinite(args.image_duration) or args.image_duration<=0:
        p.error('--image-duration必须为有限正数')
    try:
        validate_height_ratio(args.height_ratio)
    except ValueError as exc:
        p.error(str(exc))
    if args.anchor is not None and any(not math.isfinite(x) or not 0<=x<=1 for x in args.anchor):
        p.error('--anchor每个坐标须在[0,1]')
    if not math.isfinite(args.timeout) or args.timeout<1:
        p.error('--timeout必须为至少1秒的有限数')
    try:
        args.cut_values = (None if args.cuts is None else [] if not args.cuts.strip()
                           else [float(c.strip()) for c in args.cuts.split(',')])
        if args.cut_values is not None and any(not math.isfinite(x) for x in args.cut_values):
            raise ValueError()
    except ValueError:
        p.error('--cuts须为逗号分隔有限秒数')
    if not args.inputs or args.output_dir is None:
        p.error('须提供--inputs和--output-dir')
    if len(args.inputs)>64:
        p.error('单worker批次最多64个输入')
    if args.music is not None and not args.music.is_file():
        p.error('--music文件不存在')
    args.dog_sequence = [x.strip() for x in args.dog_sequence.split(',')] if args.dog_sequence else None
    if args.dog_sequence and (any(not x for x in args.dog_sequence) or
                              any(a==b for a,b in zip(args.dog_sequence,args.dog_sequence[1:]))):
        p.error('--dog-sequence须非空且相邻狗不同')


def create_plan(path,output,info,analysis,library,args,digest):
    rng_seed = int(hashlib.sha256(f'{RNG_SALT}:{args.seed}:{digest}'.encode()).hexdigest(),16)
    # Same legacy input salt / initial bag draw; later dog draws are now isolated
    # from size/candidate count and phase. Explicit replay allows apples-to-apples
    # comparison of ALL baseline dogs, not merely the first shot.
    dog_rng = random.Random(rng_seed)
    placement_rng = random.Random(rng_seed)
    phase_rng = random.Random(rng_seed ^ 0x5048415345)
    bag = ShuffleBag(library.entries,dog_rng)
    if args.dog_id and args.dog_id not in library.entries:
        raise ValueError(f'未知dog-id: {args.dog_id}')
    plan = dict(schema_version=2,engine_version=ENGINE_VERSION,input=str(path),output=str(output),
                status='planned',shots=[],timings={},warnings=list(BASE_WARNINGS),seed=args.seed,
                rng_input_key=digest,rng_salt=RNG_SALT,rng_schema='isolated-dog-placement-phase-v1',
                dog_selection_policy='explicit sequence replay' if args.dog_sequence else 'isolated shuffle bag',
                requested_dog_sequence=args.dog_sequence,media=info,tracking='off',
                requested_height_ratio=args.height_ratio,requested_anchor=args.anchor,
                requested_cuts=args.cut_values,requested_avoid_regions=args.avoid_region,
                placement_assistance='manual_anchor' if args.anchor is not None else 'user_avoid_regions' if args.avoid_region else 'none',
                analysis={k:v for k,v in analysis.items() if k not in ('shots','signature')},
                protection_backend='lightweight',allow_special_fallback=args.allow_special_fallback,
                protection_policy='detected-soft-authorized' if args.allow_special_fallback else 'detected-hard',
                planning_source_decode_passes=0,tracking_analysis_seconds=0.,
                audio_policy='music replaces first source audio' if args.music else 'first source audio preserved; copy compatible codec else AAC' if info['audio'] else 'silent',
                music=str(args.music.resolve()) if args.music else None,plan_only=args.plan_only)
    if args.tracking is not None or args.protection_backend is not None:
        plan['warnings'].append('兼容off/lightweight参数已弃用，可移除。')
    if info['kind'] == 'image' and args.cut_values is not None:
        plan['warnings'].append('图片忽略--cuts，始终一个镜头；人工切点只适用于视频。')
    if info.get('animated_first_frame_only'):
        plan['warnings'].append('动画图片仅使用转正首帧。')
    plan['warnings'].extend(analysis['limitations'])
    fps = info['fps']
    effective = []
    for shot in analysis['shots']:
        start = math.ceil(shot['start']*fps-1e-7)/fps
        if effective and (start-effective[-1]['start']<.25-1e-6 or info['duration']-start<.25-1e-6):
            plan['warnings'].append(f'切点{shot["start"]:.6f}s在CFR量化后过短，已合并。')
            continue
        effective.append(dict(shot,start=start))
    if len(effective)>1 and len(library.entries)<2:
        raise ValueError('多镜头至少需要两只狗')
    if args.dog_sequence:
        if len(args.dog_sequence)!=len(effective):
            raise ValueError('--dog-sequence长度必须等于本次有效镜头数；可用--cuts复用旧切点')
        if any(d not in library.entries for d in args.dog_sequence):
            raise ValueError('--dog-sequence包含未知dog id')
        if args.dog_id and args.dog_sequence[0]!=args.dog_id:
            raise ValueError('--dog-id与--dog-sequence首狗冲突')
    for i,shot in enumerate(effective):
        end = effective[i+1]['start'] if i+1<len(effective) else info['duration']
        if args.dog_sequence:
            dog_id = args.dog_sequence[i]
        elif i==0 and args.dog_id:
            dog_id = args.dog_id
            bag.bag = [d for d in bag.ids if d!=dog_id]
            dog_rng.shuffle(bag.bag)
            bag.previous = dog_id
        else:
            dog_id = bag.next()
        dog = library.load(dog_id)
        rgb = decode_image(shot['representative_png'])
        extra = [decode_image(s['png']) for s in shot.get('temporal_samples',[])[:2]]
        try:
            pos = placement(rgb,info,dog,args.height_ratio,args.placement_mode,placement_rng,
                            args.anchor,extra_frames=extra,avoid_regions=args.avoid_region,
                            allow_partial_overlap=args.allow_special_fallback)
        except ValueError as exc:
            failure = dict(start=shot['start'],end=end,dog_id=dog_id,status='failed',
                           protection_diagnostics=getattr(exc,'diagnostics',{}))
            plan['shots'].append(failure)
            plan['first_failed_shot'] = failure
            exc.plan = plan
            raise
        plan['shots'].append(dict(start=shot['start'],end=end,dog_id=dog_id,
                                  anchor=pos['anchor_normalized'],height_ratio=pos['actual_height_ratio'],
                                  phase_frames=phase_rng.randrange(dog['frame_count']),frame_start=round(shot['start']*fps),
                                  dog_fps=dog['fps'],dog_frame_count=dog['frame_count'],
                                  dog_source_frame_count=dog['source_frame_count'],dog_loop_start_frame=dog['loop_start_frame'],
                                  dog_loop_end_frame=dog['loop_end_frame'],dog_sha256=dog['manifest']['sha256'],
                                  placement=pos,placement_segments=[],temporal_placement_fallback=False))
        plan['warnings'].extend(dog['warnings'])
        plan['warnings'].extend(pos['warnings'])
    plan['warnings'] = list(dict.fromkeys(plan['warnings']))
    return plan


def write_all(stream,data):
    view = memoryview(data)
    while len(view):
        n = stream.write(view)
        if not n:
            raise RuntimeError('FFmpeg编码器管道已关闭')
        view = view[n:]


def render_video(path,target,plan,library,timeout):
    info = plan['media']
    W,H,fps = info['output_width'],info['output_height'],info['fps']
    encoded,muxed = target.with_suffix('.video.part.mp4'),target.with_suffix('.mux.part.mp4')
    encoder,frames,n = None,None,0
    timings = dict(decode_overlay_encode=0.,mux=0.,verify=0.)
    started = time.perf_counter()
    try:
        encoder = PipeProcess(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-filter_threads','1',
                               '-f','rawvideo','-pix_fmt','rgb24','-s:v',f'{W}x{H}','-r',f'{fps:.10f}',
                               '-i','pipe:0','-an','-c:v','libx264','-preset','ultrafast','-crf',
                               '23' if info['quality']=='preview' else '20','-pix_fmt','yuv420p','-threads','1',
                               '-movflags','+faststart',encoded],timeout=timeout,writer=True)
        if info['kind']=='image':
            base = cv2.resize(image_rgb(path),(info['display_width'],info['display_height']),interpolation=cv2.INTER_AREA)
            base = cv2.copyMakeBorder(base,0,H-base.shape[0],0,W-base.shape[1],cv2.BORDER_CONSTANT)
            frames = (base.copy() for _ in range(max(1,math.ceil(info['duration']*fps-1e-7))))
        else:
            frames = video_frames(path,info,info['display_width'],info['display_height'],fps,timeout,pad=True)
        shot_index,prepared = -1,None
        for n0,frame in enumerate(frames):
            current = max(0,shot_index)
            while current+1<len(plan['shots']) and n0>=plan['shots'][current+1]['frame_start']:
                current += 1
            if current!=shot_index:
                shot_index = current
                shot = plan['shots'][current]
                prepared = PreparedDog(library.load(shot['dog_id']),shot['placement'])
            shot = plan['shots'][current]
            index = (shot['phase_frames']+math.floor((n0-shot['frame_start'])*shot['dog_fps']/fps+1e-7))%shot['dog_frame_count']
            prepared.overlay(frame,index)
            write_all(encoder.p.stdin,frame.tobytes())
            n = n0+1
        encoder.finish()
        if n==0:
            raise RuntimeError('没有输出视频帧')
        timings['decode_overlay_encode'] = time.perf_counter()-started
        duration = n/fps
        plan.update(encoded_frames=n,effective_duration=duration,render_source_decode_passes=1 if info['kind']=='video' else 0)
        if abs(duration-info['duration'])>1/fps+.025:
            plan['warnings'].append('有效解码时长与容器声明差异超过一个输出帧；全部有效帧保留。')
        plan['shots'] = [s for s in plan['shots'] if s['frame_start']<n]
        for i,shot in enumerate(plan['shots']):
            shot['end'] = plan['shots'][i+1]['start'] if i+1<len(plan['shots']) else duration
        t = time.perf_counter()
        cmd = ['ffmpeg','-nostdin','-v','error','-y','-threads','1','-filter_threads','1','-i',encoded]
        audio_source = plan['music'] or (str(path) if info['audio'] else None)
        if audio_source:
            if plan['music']:
                cmd += ['-stream_loop','-1']
            cmd += ['-threads','1','-i',audio_source,'-map','0:v:0','-map','1:a:0','-c:v','copy']
            codec = info['audio']['codec_name'] if info['audio'] else None
            compatible = codec in {'aac','mp3','ac3','eac3','alac'} and not plan['music']
            audio_args = ['-c:a','copy'] if compatible else ['-c:a','aac','-b:a','192k','-threads','1']
            tail = ['-t',f'{duration:.10f}','-map_metadata','-1','-movflags','+faststart',muxed]
            try:
                run(cmd+audio_args+tail,timeout)
                plan['audio_encoding'] = 'copy' if compatible else 'aac'
            except RuntimeError:
                if not compatible:
                    raise
                muxed.unlink(missing_ok=True)
                run(cmd+['-c:a','aac','-b:a','192k','-threads','1']+tail,timeout)
                plan['audio_encoding'] = 'aac_fallback'
                plan['warnings'].append('音轨copy失败转AAC；视频未重编码。')
        else:
            encoded.replace(muxed)
            plan['audio_encoding'] = 'none'
        timings['mux'] = time.perf_counter()-t
        t = time.perf_counter()
        verify = probe(muxed)
        v = next(s for s in verify['streams'] if s['codec_type']=='video')
        a = [s for s in verify['streams'] if s['codec_type']=='audio']
        if (int(v['width']),int(v['height']))!=(W,H):
            raise RuntimeError('ffprobe核验：输出尺寸不一致')
        vf = int(v.get('nb_frames',n))
        vd = float(v.get('duration') or verify['format']['duration'])
        if vf!=n or abs(vd-duration)>1/fps+.025:
            raise RuntimeError('ffprobe核验：输出帧数/时长不一致')
        if bool(a)!=bool(audio_source):
            raise RuntimeError('ffprobe核验：音轨缺失或意外存在')
        plan['verification'] = dict(width=W,height=H,frame_count=vf,video_duration=vd,
                                    format_duration=float(verify['format']['duration']),fps=rate(v.get('avg_frame_rate')),
                                    video_codec=v['codec_name'],audio_codecs=[s['codec_name'] for s in a])
        timings['verify'] = time.perf_counter()-t
        muxed.replace(target)
        return timings
    finally:
        try:
            if frames is not None and hasattr(frames,'close'):
                frames.close()
        finally:
            try:
                if encoder is not None:
                    encoder.close()
            finally:
                encoded.unlink(missing_ok=True)
                muxed.unlink(missing_ok=True)


def batch(args,assets=None,cache=None):
    root = args.output_dir.expanduser().resolve()
    protected = {Path(p).expanduser().resolve() for p in args.inputs}
    if args.music:
        protected.add(args.music.expanduser().resolve())
    if (root/'summary.json').resolve() in protected:
        raise ValueError('output-dir/summary.json与输入/音乐路径冲突')
    root.mkdir(parents=True,exist_ok=True)
    task_id = 'task_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:12]
    task = root/task_id
    task.mkdir(exist_ok=False)
    assets = Path(assets) if assets is not None else SKILL_ROOT/'assets'
    cache = Path(cache) if cache is not None else SKILL_ROOT/'temp_data'/'cache'
    started = time.perf_counter()
    summary = dict(schema_version=2,engine_version=ENGINE_VERSION,task_id=task_id,task_dir=str(task),
                   summary_path=str(task/'summary.json'),status='running',seed=args.seed,plan_only=args.plan_only,
                   worker_count=1,max_inputs=64,results=[],counts={},timings={})
    save_json(task/'summary.json',summary)
    for i,name in enumerate(args.inputs):
        item_start = time.perf_counter()
        path = Path(name).expanduser().resolve()
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in path.stem)[:64] or 'input'
        stem = f'{i+1:03d}_{safe}_{hashlib.sha256(str(path).encode()).hexdigest()[:10]}'
        target,plan_path = task/(stem+'.mp4'),task/(stem+'.json')
        plan = dict(schema_version=2,engine_version=ENGINE_VERSION,input=str(path),output=str(target),
                    status='failed',shots=[],timings={},warnings=list(BASE_WARNINGS),plan_path=str(plan_path))
        try:
            if not path.is_file():
                raise ValueError('输入文件不存在或不是普通文件')
            t = time.perf_counter()
            digest = sha256(path)
            plan['timings']['hash'] = time.perf_counter()-t
            t = time.perf_counter()
            info = media_info(path,args.image_duration,args.quality)
            plan['timings']['probe'] = time.perf_counter()-t
            t = time.perf_counter()
            analysis,hit = analyze(path,info,digest,cache,args.cut_values,args.timeout)
            plan['timings']['analysis'] = time.perf_counter()-t
            t = time.perf_counter()
            library = DogLibrary(assets)
            timings = plan['timings']
            plan = create_plan(path,target,info,analysis,library,args,digest)
            plan.update(plan_path=str(plan_path),input_sha256=digest,analysis_cache_hit=hit)
            plan['timings'] = timings
            plan['timings']['planning_assets'] = time.perf_counter()-t
            if args.music and not any(s.get('codec_type')=='audio' for s in probe(args.music)['streams']):
                raise ValueError('--music没有可用音轨')
            save_json(plan_path,plan)
            if not args.plan_only:
                plan['timings'].update(render_video(path,target,plan,library,args.timeout))
                plan['status'] = 'success'
        except Exception as exc:
            if hasattr(exc,'plan'):
                timings = plan['timings']
                plan = exc.plan
                plan['timings'] = timings
                plan['plan_path'] = str(plan_path)
            plan['status'] = 'failed'
            plan['error'] = dict(type=type(exc).__name__,message=str(exc))
            target.unlink(missing_ok=True)
        finally:
            plan['timings']['total'] = time.perf_counter()-item_start
            save_json(plan_path,plan)
        summary['results'].append(dict(input=str(path),output=str(target),plan_path=str(plan_path),
                                       status=plan['status'],timings=plan['timings'],error=plan.get('error')))
        success = sum(r['status'] in ('success','planned') for r in summary['results'])
        summary['counts'] = dict(total=len(summary['results']),success=success,failed=len(summary['results'])-success,
                                 requested=len(args.inputs))
        save_json(task/'summary.json',summary)
        print(json.dumps({'status':plan['status'],'input':str(path),'output':str(target),
                          'plan':str(plan_path),'error':plan.get('error')},ensure_ascii=False),flush=True)
    summary['status'] = 'success' if summary['counts']['failed']==0 else 'partial_failure' if summary['counts']['success'] else 'failed'
    summary['timings']['total'] = time.perf_counter()-started
    save_json(task/'summary.json',summary)
    save_json(root/'summary.json',summary)
    print(json.dumps({'summary':str(root/'summary.json'),'task_summary':str(task/'summary.json'),
                      'status':summary['status'],'counts':summary['counts']},ensure_ascii=False),flush=True)
    return (0 if summary['counts']['failed']==0 else 1),summary


def main():
    p = parser()
    args = p.parse_args()
    validate_args(args,p)
    try:
        code,_ = batch(args)
        return code
    except (OSError,ValueError,RuntimeError) as exc:
        print(json.dumps({'status':'fatal','error':str(exc)},ensure_ascii=False),file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
