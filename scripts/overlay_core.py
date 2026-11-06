"""Deterministic offline overlay primitives; bounded analysis and one shader."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import subprocess
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from shot_detection import ShotDetector
from lightweight_vision import face_boxes, text_regions, head_regions, subject_proxy_zones, detect_horizontal_supports, subject_saliency_mask, detect_ground_regions

ENGINE_VERSION = 'overlay-1.4.2'
HEIGHT_RATIO_MIN = .24
HEIGHT_RATIO_MAX = .35


def validate_height_ratio(ratio):
    if ratio is not None and (not math.isfinite(ratio) or not HEIGHT_RATIO_MIN <= ratio <= HEIGHT_RATIO_MAX):
        raise ValueError('--height-ratio必须为[.24,.35]内的有限数（24%–35%）')
    return ratio
RNG_SALT = 'overlay-1.3.5'  # Stable input salt; dog stream is isolated from placement.
cv2.setNumThreads(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp', delete=False) as f:
        temp = Path(f.name)
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def run(args, timeout=120):
    result = subprocess.run([str(x) for x in args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exit {result.returncode}: ' +
                           result.stderr.decode('utf-8', 'replace')[-6000:])
    return result.stdout


class PipeProcess:
    """Deadline kills child even when pipe read/write blocks."""
    def __init__(self, args, timeout=600, reader=False, writer=False):
        self.err = tempfile.TemporaryFile()
        self.expired = False
        try:
            self.p = subprocess.Popen([str(x) for x in args],
                                      stdin=subprocess.PIPE if writer else subprocess.DEVNULL,
                                      stdout=subprocess.PIPE if reader else subprocess.DEVNULL,
                                      stderr=self.err, bufsize=0)
        except Exception:
            self.err.close()
            raise
        self.timer = threading.Timer(timeout, self._expire)
        self.timer.daemon = True
        self.timer.start()

    def _expire(self):
        self.expired = True
        if self.p.poll() is None:
            self.p.kill()

    def finish(self):
        if self.p.stdin and not self.p.stdin.closed:
            self.p.stdin.close()
        self.p.wait(timeout=15)
        self.timer.cancel()
        self.err.seek(0)
        error = self.err.read().decode('utf-8', 'replace')[-6000:]
        if self.expired or self.p.returncode:
            raise RuntimeError(f'FFmpeg {"timeout" if self.expired else "exit " + str(self.p.returncode)}: {error}')

    def close(self):
        self.timer.cancel()
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait(timeout=15)
        for stream in (self.p.stdin, self.p.stdout):
            if stream and not stream.closed:
                stream.close()
        self.err.close()


def probe(path):
    return json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                           '-of', 'json', path], timeout=60))


def rate(value, fallback=30.):
    try:
        a, *b = str(value).split('/')
        r = float(a) / (float(b[0]) if b else 1)
        return r if math.isfinite(r) and r > 0 else fallback
    except (ValueError, ZeroDivisionError):
        return fallback


def image_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return np.asarray(bg.convert('RGB')).copy()


def media_info(path, image_duration, quality):
    try:
        with Image.open(path) as im:
            original_format = im.format
            animated = bool(getattr(im, 'is_animated', False))
        rgb = image_rgb(path)
        h, w = rgb.shape[:2]
        info = dict(kind='image', width=w, height=h, duration=image_duration,
                    fps=30., audio=None, rotation=0, image_format=original_format,
                    animated_first_frame_only=animated, exif_transposed=True)
    except (OSError, ValueError, Image.DecompressionBombError):
        data = probe(path)
        stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'
                       and not s.get('disposition', {}).get('attached_pic')), None)
        if stream is None:
            raise ValueError('输入不是可解码的图片或视频')
        w, h = int(stream['width']), int(stream['height'])
        rotation = float(stream.get('tags', {}).get('rotate', 0) or 0)
        for side in stream.get('side_data_list', []):
            if 'rotation' in side:
                rotation = float(side['rotation'])
        if 45 < abs(rotation) % 180 < 135:
            w, h = h, w
        d = stream.get('duration') or data.get('format', {}).get('duration')
        duration = float(d) if d not in (None, 'N/A') else 0
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('无法确定有限的视频时长')
        source_fps = rate(stream.get('avg_frame_rate'), rate(stream.get('r_frame_rate')))
        fps = min(60., max(1., source_fps))
        audio = next((s for s in data['streams'] if s.get('codec_type') == 'audio'), None)
        info = dict(kind='video', width=w, height=h, duration=duration, fps=fps,
                    source_fps=source_fps, audio=audio, rotation=rotation,
                    video_stream_index=stream['index'],
                    timing_policy='FFmpeg fps filter to CFR; VFR timestamps approximated, not preserved',
                    rotation_policy='FFmpeg default autorotate before identical analysis/render scaling')
    limit = {'preview': 1280, 'hd': 1920, 'source': max(w, h)}[quality]
    factor = min(1., limit / max(w, h))
    dw, dh = max(1, round(w * factor)), max(1, round(h * factor))
    info.update(display_width=dw, display_height=dh, output_width=dw + dw % 2,
                output_height=dh + dh % 2, quality=quality, even_padding=[dw % 2, dh % 2])
    return info


def video_frames(path, info, width, height, fps, timeout=600, pad=False):
    vf = f'fps={fps:.10f},scale={width}:{height}:flags=bilinear'
    ow, oh = width, height
    if pad:
        ow, oh = width + width % 2, height + height % 2
        vf += f',pad={ow}:{oh}:0:0:color=black'
    proc = PipeProcess(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
                        '-filter_threads', '1', '-i', path, '-map',
                        f'0:{info["video_stream_index"]}', '-an', '-sn', '-dn',
                        '-vf', vf, '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-threads', '1', 'pipe:1'],
                       timeout=timeout, reader=True)
    size = ow * oh * 3
    try:
        while True:
            chunks, remaining = [], size
            while remaining:
                chunk = proc.p.stdout.read(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining == size:
                break
            if remaining:
                raise RuntimeError('FFmpeg解码返回不完整帧')
            yield np.frombuffer(b''.join(chunks), np.uint8).reshape(oh, ow, 3).copy()
        proc.finish()
    finally:
        proc.close()


def small_size(info):
    f = min(1., 480 / max(info['display_width'], info['display_height']))
    return max(1, round(info['display_width'] * f)), max(1, round(info['display_height'] * f))


def encode_image(image):
    ok, arr = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('缓存图像编码失败')
    return base64.b64encode(arr).decode('ascii')


def decode_image(text):
    arr = np.frombuffer(base64.b64decode(text), np.uint8)
    return cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def difference(a, b):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    diff = max(float(np.mean(cv2.absdiff(ga, gb))), float(np.mean(cv2.absdiff(a, b)))) / 255
    ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(ha, ha, 1, 0, cv2.NORM_L1)
    cv2.normalize(hb, hb, 1, 0, cv2.NORM_L1)
    return diff, float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))


def normalize_cuts(cuts, duration):
    result = []
    for t in sorted(set(float(c) for c in cuts)):
        if not math.isfinite(t) or not 0 < t < duration:
            raise ValueError('cuts必须为时长范围内的有限正数')
        if t - (result[-1] if result else 0) < .25 - 1e-6 or duration - t < .25 - 1e-6:
            raise ValueError('人工cuts产生小于0.25秒的镜头，请调整cuts')
        result.append(t)
    return result


def analyze(path, info, input_hash, cache_dir, manual_cuts=None, timeout=600):
    """One source decode; retain at most three frames/shot from that same pass.

    Representative is the exact detected cut frame. An isolated deterministic
    reservoir retains one later sample, and the last observed frame is retained.
    No frame is PNG-encoded until its shot is finalized; no per-frame body scan.
    """
    signature = dict(version=ENGINE_VERSION, opencv=cv2.__version__, detector='spatial-edge-context-3',
                     sampling='representative-reservoir-last-1', hash=input_hash, kind=info['kind'],
                     width=info['display_width'], height=info['display_height'],
                     duration=info['duration'], source_fps=info['fps'], cuts=manual_cuts)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_file = Path(cache_dir) / (key + '.json')
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding='utf-8'))
            if cached['signature'] == signature and cached['shots']:
                for shot in cached['shots']:
                    decode_image(shot['representative_png'])
                    for sample in shot['temporal_samples']:
                        decode_image(sample['png'])
                return cached, True
        except (ValueError, KeyError, cv2.error, TypeError, OSError):
            pass
    sw, sh = small_size(info)
    sampling_fps = info['fps'] if info['kind'] == 'video' else 1.
    detector = ShotDetector(sampling_fps, info['duration'])
    cuts = (normalize_cuts(manual_cuts, info['duration'])
            if manual_cuts is not None and info['kind'] == 'video' else None)
    shots, cut_events, state = [], [], None
    previous, count, manual_index = None, 0, 0
    import random
    sample_rng = random.Random(0)

    def start(t, frame):
        return dict(start=t, representative=frame.copy(), sample=None, last=None, seen=0)

    def finish(s, end):
        samples = {}
        for item in (s['sample'], s['last']):
            if item is not None and s['start'] < item[0] < end:
                samples[item[0]] = item[1]
        shots.append(dict(start=s['start'], end=end, representative_png=encode_image(s['representative']),
                          temporal_samples=[dict(time=t, png=encode_image(f)) for t, f in sorted(samples.items())]))

    frames = (video_frames(path, info, sw, sh, sampling_fps, timeout) if info['kind'] == 'video'
              else iter([cv2.resize(image_rgb(path), (sw, sh))]))
    try:
        for i, frame in enumerate(frames):
            count += 1
            t = i / sampling_fps
            if state is None:
                state = start(0., frame)
            if cuts is not None:
                events = []
                while manual_index < len(cuts) and t >= cuts[manual_index]:
                    events.append(dict(time=cuts[manual_index], frame=frame))
                    manual_index += 1
            else:
                events = detector.observe(frame, t, difference(previous, frame) if previous is not None else (0., 0.))
            for event in events:
                finish(state, event['time'])
                state = start(event['time'], event['frame'])
                if 'metrics' in event:
                    cut_events.append(event['metrics'])
            state['seen'] += 1
            if t > state['start']:
                state['last'] = (t, frame.copy())
                if sample_rng.randrange(state['seen']) == 0:
                    state['sample'] = (t, frame.copy())
            previous = frame
    finally:
        if hasattr(frames, 'close'):
            frames.close()
    if state is None:
        raise ValueError('分析未解码到任何帧')
    if cuts is None:
        for event in detector.observe(previous, min(info['duration'], count / sampling_fps), (0., 0.)):
            finish(state, event['time'])
            state = start(event['time'], event['frame'])
            cut_events.append(event['metrics'])
    else:
        while manual_index < len(cuts):
            finish(state, cuts[manual_index])
            state = start(cuts[manual_index], previous)
            manual_index += 1
    finish(state, info['duration'])
    result = dict(signature=signature, shots=shots, sampling_fps=sampling_fps, sampled_frames=count,
                  analysis_source_decode_passes=1 if info['kind']=='video' else 0,
                  protection_sample_counts=[1+len(s['temporal_samples']) for s in shots],
                  protection_sampling_policy='same-pass representative + reservoir + last; max 3 per shot',
                  cut_events=cut_events, cut_method='manual override' if cuts is not None else
                  'CFR all-frame RGB + HSV + spatial luminance + adaptive edge + persistence/return',
                  detector_stats=dict(rejected_returns=detector.rejected_returns, illumination_frames=detector.illumination_frames),
                  cut_time_resolution_seconds=1 / sampling_fps,
                  limitations=[f'最长边480、{sampling_fps:g}fps CFR多特征分镜；VFR逐帧PTS不保留。',
                               '0.25s持续/返回确认；极亮/暗恢复段、快运动、渐变仍可能误/漏切。',
                               '每镜头最多3帧人脸/文字启发式；没有全片主体保护，不能保证零遮挡。'])
    save_json(cache_file, result)
    return result, False


class DogLibrary:
    def __init__(self, assets):
        self.assets = Path(assets).resolve()
        manifest = json.loads((self.assets / 'dog_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('version') != 1 or not manifest.get('dogs'):
            raise ValueError('manifest须为version=1且dogs非空')
        self.entries = {}
        for dog in manifest['dogs']:
            if dog['id'] in self.entries:
                raise ValueError('manifest包含重复dog id')
            self.entries[dog['id']] = dog
        self.loaded_id, self.loaded = None, None

    def load(self, dog_id):
        if self.loaded_id == dog_id:
            return self.loaded
        e = self.entries[dog_id]
        path = (self.assets / e['file']).resolve()
        if not path.is_relative_to(self.assets) or not path.is_file():
            raise ValueError(f'{dog_id}: file必须在assets内且存在')
        if sha256(path) != e['sha256']:
            raise ValueError(f'{dog_id}: SHA256不匹配')
        with np.load(path, allow_pickle=False) as z:
            frames = z['frames']
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 4:
            raise ValueError(f'{dog_id}: frames必须uint8 N,H,W,4')
        n, h, w, _ = frames.shape
        if (n, w, h) != (int(e['frame_count']), int(e['width']), int(e['height'])) or n < 1:
            raise ValueError(f'{dog_id}: manifest尺寸/帧数不一致')
        fps = float(e['fps'])
        ax, ay = map(float, e['anchor'])
        if not math.isfinite(fps) or fps <= 0 or not all(map(math.isfinite, (ax, ay))):
            raise ValueError(f'{dog_id}: fps/anchor无效')
        if not 0 <= ax <= w or not 0 <= ay <= h:
            raise ValueError(f'{dog_id}: 脚底anchor不在裁切画布内')
        quality = e.get('quality', {})
        loop_start = int(e.get('loop_start_frame', quality.get('loop_start_frame', 0)))
        loop_end = int(e.get('loop_end_frame', quality.get('loop_end_frame', n)))
        if not 0 <= loop_start < loop_end <= n:
            raise ValueError(f'{dog_id}: loop区间须为0 <= start < end <= 原帧数（end不含）')
        jumps = [int(j) for j in quality.get('suspicious_color_jump_frames', [])]
        unsafe = [j for j in jumps if loop_start < j < loop_end]
        warnings = []
        if unsafe:
            loop_end = min(unsafe)
            warnings.append(f'{dog_id}: suspicious_color_jump_frames触发安全前缀[{loop_start},{loop_end})；源素材保留。')
        if loop_start in jumps:
            raise ValueError(f'{dog_id}: 循环起点已标记异常，不能默认安全循环')
        if loop_start != 0 or loop_end != n:
            warnings.append(f'{dog_id}: 仅循环原始帧[{loop_start},{loop_end})，原始{n}帧未删除；不宣称循环接缝已验证。')
        frames = frames[loop_start:loop_end]
        union = np.zeros((h, w), dtype=bool)
        for frame in frames:
            alpha_mask = frame[:, :, 3] > 0
            if not np.any(alpha_mask):
                raise ValueError(f'{dog_id}: 包含全透明帧，会违反每帧恰好新增一只狗')
            union |= alpha_mask
        yy, xx = np.nonzero(union)
        bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        data = dict(frames=frames, width=w, height=h, anchor=(ax, ay), fps=fps,
                    frame_count=len(frames), source_frame_count=n, loop_start_frame=loop_start,
                    loop_end_frame=loop_end, warnings=warnings, union_bbox=bbox, manifest=e)
        self.loaded_id, self.loaded = dog_id, data
        return data


class ShuffleBag:
    def __init__(self, ids, rng):
        self.ids, self.rng, self.bag, self.previous = sorted(ids), rng, [], None

    def next(self):
        if not self.bag:
            self.bag = list(self.ids)
            self.rng.shuffle(self.bag)
            if len(self.bag) > 1 and self.bag[-1] == self.previous:
                self.bag[-1], self.bag[0] = self.bag[0], self.bag[-1]
        self.previous = self.bag.pop()
        return self.previous


def intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def validate_avoid_regions(regions):
    result = []
    for region in regions or []:
        if len(region) != 4:
            raise ValueError('--avoid-region需要四个坐标X1 Y1 X2 Y2')
        x1, y1, x2, y2 = [float(v) for v in region]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (x1,y1,x2,y2)) or not (x1 < x2 and y1 < y2):
            raise ValueError('--avoid-region须满足有限坐标0≤X1<X2≤1、0≤Y1<Y2≤1')
        result.append([x1,y1,x2,y2])
    return result


class PlacementError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def placement(rgb, info, dog, ratio, mode, rng, explicit_anchor=None, extra_frames=None,
              avoid_regions=None, allow_partial_overlap=False):
    """Single bounded size/location comparison. Manual exclusions always hard.

    In auto mode only evidence-backed support candidates are searched.
    Detected face/text and body-proxy bounds are hard without authorization;
    explicit partial overlap softens automatic subject cues, never the manual
    exclusions or support requirement. Motion following is not implemented.
    """
    validate_height_ratio(ratio)
    W, H = info['display_width'], info['display_height']
    sw, sh = rgb.shape[1], rgb.shape[0]
    user_regions = validate_avoid_regions(avoid_regions)
    manual = [[x*W,y*H,x2*W,y2*H] for x,y,x2,y2 in user_regions]
    faces, texts, heads, haar = [], [], [], True
    frames = [rgb] + list(extra_frames or [])[:2]
    for frame in frames:
        if frame.shape != rgb.shape:
            frame = cv2.resize(frame, (sw, sh))
        found, available = face_boxes(frame)
        faces.extend(found)
        texts.extend(text_regions(frame))
        for head in head_regions(frame):
            x,y,x2,y2 = head['box']
            heads.append(dict(head,box=[x*W/sw,y*H/sh,x2*W/sw,y2*H/sh]))
        haar &= available
    def scaled(boxes):
        return [[x*W/sw,y*H/sh,x2*W/sw,y2*H/sh]
                for x,y,x2,y2 in dict.fromkeys(tuple(b) for b in boxes)]
    faces, texts = scaled(faces), scaled(texts)
    proxies = subject_proxy_zones(heads, H, W)
    supports = detect_horizontal_supports(rgb)
    ground_regions = detect_ground_regions(rgb)
    saliency_mask = subject_saliency_mask(rgb)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    grad = cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0), cv2.Sobel(gray,cv2.CV_32F,0,1))
    cost_map = grad*.25 + np.abs(gray-float(gray.mean()))*.45
    integral = cv2.integral(cost_map)
    sizes = [ratio] if ratio is not None else [.35, .30, .24]
    choices = []
    rejections = dict(bounds=0, user_region=0, face=0, text=0, below_size_floor=0, no_support=0, subject=0, saliency=0)
    evaluated = []
    for requested in sizes:
        scale = min(H*requested/dog['height'], W*.88/dog['width'], H*.88/dog['height'])
        rw, rh = max(1,round(dog['width']*scale)),max(1,round(dog['height']*scale))
        ax, ay = dog['anchor'][0]*rw/dog['width'],dog['anchor'][1]*rh/dog['height']
        if explicit_anchor is not None:
            px, py = explicit_anchor[0]*W,explicit_anchor[1]*H
            limits = [1.]
            for room, need in ((px,ax),(W-px,rw-ax),(py,ay),(H-py,rh-ay)):
                if need > 0:
                    limits.append(max(0.,room)/need)
            fit = min(limits)
            rw, rh = max(1,math.floor(rw*fit)),max(1,math.floor(rh*fit))
            ax, ay = dog['anchor'][0]*rw/dog['width'],dog['anchor'][1]*rh/dog['height']
            positions = [(round(px-ax),round(py-ay),'explicit_anchor', 0, 1.0)]
        else:
            # ONLY ground anchored candidates in auto mode
            positions = []
            for s in supports:
                x1, y1, x2, y2 = s['line']
                for k in (0.2, 0.5, 0.8):
                    px, py = x1 + k*(x2-x1), y1 + k*(y2-y1)
                    positions.append((round(px*W/sw - ax), round(py*H/sh - ay), 'ground_anchored', py, s['score']))
            for gr in ground_regions:
                px, py = gr['anchor']
                positions.append((round(px*W/sw - ax), round(py*H/sh - ay), 'ground_region', py, gr['score']))

            if not positions:
                rejections['no_support'] += 1
                continue

            # Corner preference cannot create unsupported sky/water anchors.

        if rh < H*HEIGHT_RATIO_MIN-1 or rh > H*HEIGHT_RATIO_MAX+1:
            rejections['below_size_floor'] += 1
            continue
        evaluated.append(requested)
        for x,y,kind,support_py,support_score in dict.fromkeys(positions):
            rect = [x,y,x+rw,y+rh]
            if x < 0 or y < 0 or rect[2] > W or rect[3] > H:
                rejections['bounds'] += 1
                continue
            if any(intersects(rect,b) for b in manual):
                rejections['user_region'] += 1
                continue

            face_hit, text_hit = any(intersects(rect,b) for b in faces),any(intersects(rect,b) for b in texts)
            subject_hit = any(intersects(rect, p) for p in proxies)

            # Saliency check (non-person subject)
            a_s,b_s,c_s,d_s = max(0,int(x*sw/W)),max(0,int(y*sh/H)),min(sw,math.ceil((x+rw)*sw/W)),min(sh,math.ceil((y+rh)*sh/H))
            saliency_overlap = float(np.mean(saliency_mask[b_s:d_s, a_s:c_s] > 0)) if (d_s > b_s and c_s > a_s) else 0.

            # HARD rejection for faces/text/subject/saliency unless authorized
            if not allow_partial_overlap:
                if face_hit or text_hit:
                    rejections['face' if face_hit else 'text'] += 1
                    continue
                if subject_hit:
                    rejections['subject'] += 1
                    continue
                if saliency_overlap > 0.4: # Soft limit for saliency to allow some background clutter
                    rejections['saliency'] += 1
                    continue

            density = float((integral[d_s,c_s]-integral[b_s,c_s]-integral[d_s,a_s]+integral[b_s,a_s])/max(1,(c_s-a_s)*(d_s-b_s)))
            margins = [x/W,y/H,(W-x-rw)/W,(H-y-rh)/H]
            edge = min(margins)

            # Extreme Sky/Water penalty for low-texture mid-upper regions
            sky_penalty = 0.0
            if y/H < 0.50 and density < 0.07:
                sky_penalty = 20.0

            head_overlap = 0.
            for head in heads:
                hx,hy,hx2,hy2 = head['box']
                overlap = max(0,min(x+rw,hx2)-max(x,hx))*max(0,min(y+rh,hy2)-max(y,hy))
                head_overlap = max(head_overlap,head['confidence']*overlap/max(1,(hx2-hx)*(hy2-hy)))

            head_penalty = 8.0 * head_overlap
            soft = float(face_hit)*3. + float(text_hit)*2.0 + head_penalty + float(subject_hit)*8.0 + saliency_overlap*4.0 + sky_penalty
            cost = density + soft + 1.0*max(0.,.35-rh/H) + .25*max(0.,.06-edge)/.06 - support_score

            choices.append(dict(x=x,y=y,width=rw,height=rh,anchor_pixel=[x+ax,y+ay],
                                anchor_normalized=[(x+ax)/W,(y+ay)/H], actual_height_ratio=rh/H,
                                evaluated_height_ratio=requested, cost=cost,joint_cost=cost,kind=kind,
                                ground_kind=kind, support_y=support_py/sh, support_score=support_score,
                                automatic_overlap_penalty=soft, head_cue_overlap_penalty=head_penalty,
                                head_cue_overlap_fraction=head_overlap, detected_overlap=bool(face_hit or text_hit),
                                canvas_edge_margins_normalized=margins,minimum_canvas_edge_margin=edge))
    if not choices:
        raise PlacementError('无法找到满足硬性地面支撑与主体避让要求的落点。',rejections)

    best = min(p['cost'] for p in choices)
    shortlist = [p for p in choices if p['cost'] <= best+.03]
    chosen = dict(rng.choice(shortlist))
    policy = 'detected-soft-authorized' if allow_partial_overlap else 'detected-hard'
    ground_anchored = (chosen['kind'] in ('ground_anchored', 'ground_region'))
    warnings = ['严格支撑与主体保护模式；' + ('地面锚定成功' if ground_anchored else '手动/角落落位')]
    if not haar:
        warnings.append('本地Haar不可用，未做人脸检测，不下载模型。')
    if allow_partial_overlap:
        warnings.append('已显式授权：自动脸/文字保护为软惩罚，可部分遮挡；人工禁区和画面边界仍为硬约束。')
    if chosen['minimum_canvas_edge_margin'] < .055:
        warnings.append('内侧候选受限：使用有限边缘候选，不裁切动作。')
    chosen.update(candidate_count=len(choices),best_candidate_count=len(shortlist),
                  ranked_candidates=sorted(choices,key=lambda p:p['cost'])[:6], evaluated_sizes=evaluated,
                  auto_size_range=[.24,.35] if ratio is None else None,size_mode='manual' if ratio else 'adaptive',
                  requested_height_ratio=ratio,size_fallback=False,placement_mode='fixed_ground_anchor' if ground_anchored else 'explicit_anchor',
                  ground_anchored=ground_anchored,contact_shadow=False,whole_canvas_bounds_checked=True,
                  union_bbox_source=dog['union_bbox'], detected_faces=faces,detected_text_like_regions=texts,
                  detected_head_like_regions=heads,head_cue_policy='bounded compact/dark-band-cheek proposals; always soft',
                  user_avoid_regions_normalized=user_regions,user_avoid_regions_pixels=manual,
                  protection_policy=policy,partial_overlap_accepted=allow_partial_overlap,
                  special_scene_fallback=allow_partial_overlap,protection_sample_count=len(frames),
                  automatic_detection_overlap=chosen['detected_overlap'],
                  protection_search_passes=1,temporal_relocation=False,haar_available=haar,
                  edge_fallback=chosen['minimum_canvas_edge_margin']<.055,
                  source_asset_resolution=[dog['width'],dog['height']],
                  output_asset_resolution=[chosen['width'],chosen['height']],analysis_resolution=[sw,sh],
                  selection_reason='一次有限35/30/24尺寸与内侧候选成本比较；人工禁区/完整画布硬约束。',warnings=warnings)
    return chosen


class PreparedDog:
    """Current shot's pre-multiplied, resized frames; bounded LRU for larger assets."""
    def __init__(self, dog, position, byte_budget=64 * 1024 * 1024):
        from collections import OrderedDict
        self.dog, self.position, self.cache = dog, position, OrderedDict()
        self.limit = max(1, min(dog['frame_count'], byte_budget // max(1, position['width'] * position['height'] * 16)))

    def frame(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        src = self.dog['frames'][index].astype(np.float32) / 255
        rgb = src[:, :, :3]
        alpha = src[:, :, 3:4]
        from edge_cleanup import cleanup_edge_rgb
        rgb, alpha = cleanup_edge_rgb(rgb, alpha)
        premul = np.concatenate([rgb * alpha, alpha], axis=2)
        size = (self.position['width'], self.position['height'])
        # High-quality sampling in premultiplied space. Bound Lanczos ringing to
        # valid associated-alpha colors, so invisible RGB/negative lobes cannot fringe.
        interp = cv2.INTER_AREA if size[0] <= src.shape[1] and size[1] <= src.shape[0] else cv2.INTER_LANCZOS4
        premul = cv2.resize(premul, size, interpolation=interp)
        raw_alpha = premul[:, :, 3:4].copy()
        straight = np.clip(premul[:, :, :3] / np.maximum(raw_alpha, 1e-8), 0, 1)
        straight[raw_alpha[:, :, 0] <= 0] = 0
        premul[:, :, 3:4] = np.clip(raw_alpha, 0, 1)
        premul[:, :, :3] = straight * premul[:, :, 3:4]
        value = (premul[:, :, :3] * 255, premul[:, :, 3:4])
        self.cache[index] = value
        if len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return value

    def overlay(self, rgb, index, offset=(0, 0)):
        p = self.position
        x, y = p['x'] + round(offset[0]), p['y'] + round(offset[1])
        if x < 0 or y < 0 or x+p['width'] > rgb.shape[1] or y+p['height'] > rgb.shape[0]:
            raise ValueError('motion offset exceeds full RGBA canvas bounds')
        roi = rgb[y:y + p['height'], x:x + p['width']]
        color, alpha = self.frame(index)
        roi[:] = np.clip(np.rint(color + roi * (1 - alpha)), 0, 255).astype(np.uint8)
