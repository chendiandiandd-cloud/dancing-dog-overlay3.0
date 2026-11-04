"""Deterministic, bounded-memory overlay primitives; no network/model downloads."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from classical_vision import garment_mask, body_cue_mask, text_regions, reserved_regions, mask_box_fraction

from shot_detection import ShotDetector

ENGINE_VERSION = 'overlay-1.3.5'
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
    # Unique sibling temp prevents races between independent invocations.
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
    """Deadline kills child even when pipe read/write blocks; stderr cannot fill a pipe."""
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
        # First frame for animated image formats, explicitly documented in plan.
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
        if abs(rotation) % 180 > 45 and abs(rotation) % 180 < 135:
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
                    source_start_time=data.get('format', {}).get('start_time'),
                    video_stream_index=stream['index'],
                    timing_policy='FFmpeg fps filter to CFR; VFR timestamps approximated, not preserved',
                    rotation_policy='FFmpeg default autorotate before identical analysis/render scaling')
    limit = {'preview': 1280, 'hd': 1920, 'source': max(w, h)}[quality]
    factor = min(1., limit / max(w, h))
    dw, dh = max(1, round(w * factor)), max(1, round(h * factor))
    info.update(display_width=dw, display_height=dh,
                output_width=dw + dw % 2, output_height=dh + dh % 2,
                quality=quality, even_padding=[dw % 2, dh % 2])
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
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    # Include RGB difference so equal-luminance color cuts are not invisible.
    diff = max(float(np.mean(cv2.absdiff(ga, gb))),
               float(np.mean(cv2.absdiff(a, b)))) / 255
    ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(ha, ha, 1, 0, cv2.NORM_L1)
    cv2.normalize(hb, hb, 1, 0, cv2.NORM_L1)
    hist = float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))
    return diff, hist


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
    signature = dict(version=ENGINE_VERSION, opencv=cv2.__version__, detector='spatial-edge-context-3', hash=input_hash, kind=info['kind'],
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
                return cached, True
        except (ValueError, KeyError, cv2.error, TypeError, OSError):
            pass
    sw, sh = small_size(info)
    if info['kind'] == 'video':
        sampling_fps = info['fps'] if manual_cuts is None else min(12., info['fps'])
    else:
        sampling_fps = 1.
    shots, previous, count = [], None, 0
    detector = ShotDetector(sampling_fps, info['duration'])
    cut_events = []
    temporal_samples = []
    protection_interval = max(.5, info['duration'] / 240)
    next_sample = 0.
    cuts = normalize_cuts(manual_cuts, info['duration']) if manual_cuts is not None else None
    manual_index = 0
    frames = (video_frames(path, info, sw, sh, sampling_fps, timeout)
              if info['kind'] == 'video' else iter([cv2.resize(image_rgb(path), (sw, sh))]))
    try:
        for i, frame in enumerate(frames):
            count += 1
            t = i / sampling_fps
            if t + 1e-6 >= next_sample:
                temporal_samples.append(dict(time=t, png=encode_image(frame)))
                next_sample = t + protection_interval
            if not shots:
                shots.append(dict(start=0., representative_png=encode_image(frame)))
            if cuts is not None:
                while manual_index < len(cuts) and t >= cuts[manual_index]:
                    shots.append(dict(start=cuts[manual_index], representative_png=encode_image(frame)))
                    manual_index += 1
            else:
                rgb_metrics = difference(previous, frame) if previous is not None else (0., 0.)
                for event in detector.observe(frame, t, rgb_metrics):
                    shots.append(dict(start=event['time'], representative_png=encode_image(event['frame'])))
                    cut_events.append(event['metrics'])
            previous = frame
    finally:
        if hasattr(frames, 'close'):
            frames.close()
    if not shots:
        raise ValueError('分析未解码到任何帧')
    if cuts is not None:
        # Near EOF sampling can miss a short last interval; use last available frame.
        while manual_index < len(cuts):
            shots.append(dict(start=cuts[manual_index], representative_png=encode_image(previous)))
            manual_index += 1
    elif previous is not None:
        # Last decoded CFR frame covers its frame interval; do not extrapolate
        # persistence through a missing/truncated tail of the container duration.
        for event in detector.observe(previous, min(info['duration'], count / sampling_fps), (0., 0.)):
            shots.append(dict(start=event['time'], representative_png=encode_image(event['frame'])))
            cut_events.append(event['metrics'])
    if previous is not None and (not temporal_samples or temporal_samples[-1]['time'] < (count-1)/sampling_fps):
        temporal_samples.append(dict(time=(count-1)/sampling_fps, png=encode_image(previous)))
    for i, shot in enumerate(shots):
        shot['end'] = shots[i + 1]['start'] if i + 1 < len(shots) else info['duration']
        shot['temporal_samples'] = [s for s in temporal_samples if shot['start'] <= s['time'] < shot['end']]
    result = dict(signature=signature, shots=shots, sampling_fps=sampling_fps,
                  protection_sampling_interval=protection_interval,
                  sampled_frames=count, cut_events=cut_events,
                  cut_method='manual override' if cuts is not None else 'CFR all-frame RGB + HSV + spatial luminance + adaptive edge + persistence/return',
                  detector_stats=dict(rejected_returns=detector.rejected_returns,
                                      illumination_frames=detector.illumination_frames),
                  cut_time_resolution_seconds=1 / sampling_fps,
                  limitations=[f'低分辨率最长边480分析；本次{sampling_fps:g}fps CFR采样，自动硬切时间约±{1/sampling_fps:.6f}s；VFR/超过60fps源不保留原始逐帧PTS。',
                               '0.25s持续性/返回抑制短闪光；极亮/极暗帧及其0.25s恢复段不独立切镜，附近真实硬切可能被合并。长闪光、快速运动、渐变转场仍可能漏检或误检。',
                               '小于0.25s的自动镜头合并；无硬切时整片一个镜头。',
                               '全片时间采样并集保护；采样间隙仍可能漏掉短时进入的主体/文字。'])
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
        # End is an EXCLUSIVE zero-based bound, matching frames[start:end].
        # Keep original integrity checks above; do not alter the source NPZ.
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
        # Union over every effective phase; alpha>0, not just representative frame.
        union = np.zeros((h, w), dtype=bool)
        for frame in frames:
            alpha_mask = frame[:, :, 3] > 0
            if not np.any(alpha_mask):
                raise ValueError(f'{dog_id}: 包含全透明帧，会违反每帧恰好新增一只狗')
            union |= alpha_mask
        yy, xx = np.nonzero(union)
        if not len(xx):
            raise ValueError(f'{dog_id}: 所有帧透明')
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


def face_boxes(rgb):
    cascade_dir = getattr(getattr(cv2, 'data', None), 'haarcascades', '')
    if not cascade_dir:
        return [], False
    cascade = Path(cascade_dir) / 'haarcascade_frontalface_default.xml'
    if not cascade.is_file():
        return [], False
    model = cv2.CascadeClassifier(str(cascade))
    if model.empty():
        return [], False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = model.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
    # Large false boxes (pets/texture) must never extrapolate to full-frame bodies.
    # Large close-up faces require eye evidence from a second classical cascade.
    eye = cv2.CascadeClassifier(str(Path(cascade_dir) / 'haarcascade_eye.xml'))
    result = []
    H, W = gray.shape
    for x, y, w, h in boxes:
        area = w*h / (W*H)
        if area > .30 or w > W*.65 or h > H*.60:
            continue
        if area > .075:
            eyes = eye.detectMultiScale(gray[y:y+int(h*.65), x:x+w], 1.1, 3,
                                        minSize=(max(5,w//10),max(5,h//10))) if not eye.empty() else []
            if len(eyes) < 1:
                continue
        result.append([int(x), int(y), int(x+w), int(y+h)])
    return result, True


def intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def person_boxes(rgb, faces):
    """Bundled classical HOG plus conservative face-derived body regions; no downloads."""
    h, w = rgb.shape[:2]
    boxes = []
    for x0, y0, x1, y1 in faces:
        fw, fh = x1 - x0, y1 - y0
        # Widen toward clothing/long hems; cap lateral close-up extrapolation.
        spread = min(1.25 * fw, w * .18)
        if fw/w > .28:
            # Close-up head is not reliable evidence of full-width standing body.
            # Preserve the actual detected face; constrain lateral inferred body.
            spread = min(spread,w*.045)
        boxes.append([max(0, x0 - spread), max(0, y0 - .2 * fh),
                      min(w, x1 + spread), min(h, y1 + 8.5 * fh)])
    hog_available = callable(getattr(cv2, 'HOGDescriptor_getDefaultPeopleDetector', None))
    if hog_available and w >= 64 and h >= 128:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        found, weights = hog.detectMultiScale(rgb, winStride=(8, 8), padding=(8, 8), scale=1.08)
        for (x, y, bw, bh), weight in zip(found, np.asarray(weights).reshape(-1)):
            if weight >= .5 and bw*bh/(w*h) < .60 and bw/w < .80:
                boxes.append([max(0, x - .08 * bw), max(0, y - .04 * bh),
                              min(w, x + 1.08 * bw), min(h, y + 1.04 * bh)])
    return boxes, hog_available


def validate_avoid_regions(regions):
    """Normalize explicit rectangles; reject malformed/nonfinite/degenerate input."""
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


def placement(rgb, info, dog, ratio, mode, rng, explicit_anchor=None, extra_frames=None, avoid_regions=None,
              extra_mask=None, semantic_protection=False, protection_policy='strict'):
    """Bounded search; protection relaxation requires an explicit caller policy."""
    if protection_policy not in ('strict', 'body-relaxed', 'detected-relaxed'):
        raise ValueError('未知保护策略')
    relaxed = protection_policy != 'strict'
    user_regions = validate_avoid_regions(avoid_regions)
    frames = [rgb] + list(extra_frames or [])
    faces, people, texts = [], [], []
    masks = np.zeros(rgb.shape[:2], np.uint8)
    if extra_mask is not None:
        masks |= cv2.resize(extra_mask, (rgb.shape[1],rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    haar, hog = True, True
    for frame in frames:
        if frame.shape != rgb.shape:
            frame = cv2.resize(frame, (rgb.shape[1], rgb.shape[0]))
        f, ha = face_boxes(frame)
        if semantic_protection:
            from semantic_protection import protector
            segmenter = protector()
            categories = segmenter.categories(frame)
            semantic = segmenter.mask_from_categories(categories)
            masks |= semantic
            # Face rectangles must have actual person evidence, not cat Haar hits.
            from classical_vision import mask_box_fraction
            person_mask = (categories == segmenter.labels.index('person')).astype(np.uint8)
            f = [box for box in f if mask_box_fraction(person_mask, box) > .5]
            # Semantic silhouette replaces HOG/face-derived body rectangles and
            # broad skin/GrabCut expansions. Never OR old bodymask back in.
            p, ho = [], True
        else:
            p, ho = person_boxes(frame, f)
            masks |= garment_mask(frame, f)
            masks |= body_cue_mask(frame, f)
        faces.extend(f); people.extend(p); texts.extend(text_regions(frame))
        haar &= ha; hog &= ho
    # De-duplicate repeated static boxes, keeping all temporal movement extents.
    faces = [list(v) for v in dict.fromkeys(tuple(v) for v in faces)]
    people = [list(v) for v in dict.fromkeys(tuple(v) for v in people)]
    texts = [list(v) for v in dict.fromkeys(tuple(v) for v in texts)]
    if relaxed:
        # Explicit special-scene fallback only. Manual keep-out regions and
        # complete-cycle bounds remain hard constraints in _fixed_placement.
        people = []
        masks[:] = 0
        if protection_policy == 'detected-relaxed':
            faces, texts = [], []
    protection = (faces, haar, people, hog, texts, masks, frames)
    prefer_inset = mode == 'auto' and explicit_anchor is None and ratio is None
    sizes = ([ratio] if ratio is not None else [.40, .36, .32, .28, .24] if prefer_inset
             else [.24, .22, .20, .18])
    minimum_preferred = .24 if prefer_inset else .18
    fallback_sizes = [] if prefer_inset else [.15, .12, .09, .06]
    choices, failures = [], []
    # Safety first, then larger, then inset: do not shrink merely to move inward.
    for requested in sizes + ([] if ratio is not None else fallback_sizes):
        if ratio is None and requested < minimum_preferred and choices:
            break
        try:
            pos = _fixed_placement(rgb, info, dog, requested, mode, rng, explicit_anchor, protection, user_regions,
                                   prefer_inset=prefer_inset)
        except ValueError as exc:
            failures.append(dict(requested_height_ratio=requested, message=str(exc),
                                 rejection_counts=getattr(exc,'diagnostics',{})))
            continue
        # Pixel rounding + full-cycle fit determine actual size (never stale rh).
        actual = pos['height'] / info['display_height']
        if prefer_inset and actual < .24 - .5 / info['display_height']:
            failures.append(dict(requested_height_ratio=requested, message='动作宽度约束会缩至24%以下，拒绝自动缩小'))
            continue
        pos['actual_height_ratio'] = actual
        pos['size_preference_cost'] = (0. if ratio is not None else
                                       1.5 * max(0., .40 - actual) if prefer_inset else
                                       .9 * max(0., .24 - actual))
        pos['joint_cost'] = pos['cost'] + pos['size_preference_cost'] + (
            .20 * max(0., .06-pos['minimum_canvas_edge_margin']) / .06 if prefer_inset else 0.)
        pos['evaluated_height_ratio'] = requested
        choices.append(pos)
    if not choices:
        raise PlacementError('所有尺寸/落点均无法满足完整周期边界和分层检测保护',
                             dict(evaluated_sizes=failures, face_count=len(faces), person_count=len(people),
                                  text_count=len(texts), combined_mask_fraction=float((masks>0).mean()),
                                  extra_body_mask_fraction=float((extra_mask>0).mean()) if extra_mask is not None else None))
    # Deterministic tie-break: prefer larger; random seed selects spatial near-ties.
    # Authorized relaxed scenes should not trade an available inset 24%-40%
    # sticker for a larger one hugging the edge. Never lower the size floor.
    inset_choices = ([p for p in choices if p['minimum_canvas_edge_margin'] >= .055]
                     if relaxed and prefer_inset else [])
    chosen = dict(min(inset_choices or choices,
                      key=lambda p: (p['joint_cost'], -p['actual_height_ratio'])))
    chosen.update(size_mode='manual' if ratio is not None else 'adaptive', requested_height_ratio=ratio,
                  evaluated_sizes=[dict(height_ratio=p['actual_height_ratio'],
                                        requested=p['evaluated_height_ratio'], joint_cost=p['joint_cost'],
                                        anchor=p['anchor_normalized'], kind=p['kind']) for p in choices],
                  selection_reason=('检测保护/完整周期边界优先；肤色连通与相邻中性衣服补充全片mask，尺寸在24%–40%内与内侧软权衡，不自动缩小至24%以下，不强制中心。' if prefer_inset else
                                    '全片采样人物/衣服/文字并集与四角小保留区硬约束；连续支撑面证据不足时安全边角screen_sticker（非地面锚定、无阴影）'),
                  garment_mask_png=encode_image(np.repeat((masks*255)[:,:,None],3,axis=2)),
                  protection_sample_count=len(frames), garment_protected_fraction=float(np.mean(masks > 0)),
                  source_asset_resolution=[dog['width'], dog['height']],
                  output_asset_resolution=[chosen['width'], chosen['height']],
                  scale_xy=[chosen['width'] / dog['width'], chosen['height'] / dog['height']],
                  native_asset_provenance=dog.get('manifest', {}).get('resolution_provenance'),
                  analysis_resolution=[rgb.shape[1], rgb.shape[0]])
    chosen['auto_size_range'] = [.24, .40] if prefer_inset else None
    chosen['size_fallback'] = False  # default auto never falls below the 24% floor
    if ratio is None and not prefer_inset and chosen['actual_height_ratio'] < .175:
        chosen['warnings'].append('常规0.18–0.24尺寸无安全落点；已缩小退化，不保证审美。')
    chosen['protection_policy'] = protection_policy
    if relaxed:
        chosen['placement_mode'] = 'screen_sticker'
        chosen['ground_anchored'] = False
        chosen['contact_shadow'] = False
        chosen['warnings'].append('用户允许的特殊场景兜底：' + ('关闭身体/衣服保护，保留检测脸与文字。' if protection_policy == 'body-relaxed' else '关闭自动主体/脸/文字保护；可能遮挡，保留人工禁区和完整动作边界。'))
    return chosen


def _fixed_placement(rgb, info, dog, ratio, mode, rng, explicit_anchor=None, protection=None, user_regions=None,
                     prefer_inset=False):
    W, H = info['display_width'], info['display_height']
    sw, sh = rgb.shape[1], rgb.shape[0]
    scale = min(H * ratio / dog['height'], W * .8 / dog['width'], H * .8 / dog['height'])
    rw, rh = max(1, round(dog['width'] * scale)), max(1, round(dog['height'] * scale))
    ax = dog['anchor'][0] * rw / dog['width']
    ay = dog['anchor'][1] * rh / dog['height']
    faces, haar, people, hog, texts, garment, sampled = protection
    corner_reserves = reserved_regions(sh, sw)
    avoid = texts + ([] if explicit_anchor is not None else corner_reserves)
    avoid = [[x * W / sw, y * H / sh, x2 * W / sw, y2 * H / sh] for x,y,x2,y2 in avoid]
    user_avoid = [[x*W,y*H,x2*W,y2*H] for x,y,x2,y2 in (user_regions or [])]
    faces = [[x * W / sw, y * H / sh, x2 * W / sw, y2 * H / sh] for x, y, x2, y2 in faces]
    people = [[x * W / sw, y * H / sh, x2 * W / sw, y2 * H / sh] for x, y, x2, y2 in people]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)) / 1024
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    contrast = np.linalg.norm(lab - np.mean(lab, axis=(0, 1)), axis=2) / 255
    yy, xx = np.mgrid[:sh, :sw]
    center = np.exp(-(((xx / sw - .5) / .3) ** 2 + ((yy / sh - .5) / .35) ** 2))
    saliency = contrast * .6 + grad * .4 + center * .18
    warnings = ['显著性proxy不是可靠语义识别/主体避让；Hough水平线不等于理解地面。']
    if not haar:
        warnings.append('本地Haar级联不可用，未做人脸检测；不下载模型。')
    if not hog:
        warnings.append('本地HOG人物检测不可用；仅保留可用的人脸扩展近似框，不保证人物避让。')

    rejections = dict(bounds=0, face=0, person=0, text_or_corner=0, user_region=0, body_or_garment=0)

    def evaluate(x, y, support, kind, geometry=None):
        left, top = round(x - ax), round(y - ay)
        rect = [left, top, left + rw, top + rh]
        # Full RGBA canvas bound is stronger than union-alpha bound.
        if left < 0 or top < 0 or rect[2] > W or rect[3] > H:
            rejections['bounds'] += 1
            return None
        for layer,boxes in [('face',faces),('person',people),('text_or_corner',avoid),('user_region',user_avoid)]:
            if any(intersects(rect, f) for f in boxes):
                rejections[layer] += 1
                return None
        scaled_rect = [rect[0]*sw/W, rect[1]*sh/H, rect[2]*sw/W, rect[3]*sh/H]
        if mask_box_fraction(garment, scaled_rect) > .015:
            rejections['body_or_garment'] += 1
            return None
        x0, y0 = max(0, int(left * sw / W)), max(0, int(top * sh / H))
        x1, y1 = min(sw, max(x0 + 1, math.ceil(rect[2] * sw / W))), min(sh, max(y0 + 1, math.ceil(rect[3] * sh / H)))
        lower_region_cost = .55 * max(0., (.72 - y / H) / .72)
        cost = (float(np.mean(saliency[y0:y1, x0:x1]))
                + .35 * float(np.mean(grad[y0:y1, x0:x1])) - support * .15
                + lower_region_cost)
        if geometry:
            cost += geometry['cost_adjustment']
        return dict(x=left, y=top, width=rw, height=rh, anchor_pixel=[left + ax, top + ay],
                    anchor_normalized=[(left + ax) / W, (top + ay) / H], cost=cost,
                    kind=kind, support_score=support, support_geometry=geometry,
                    lower_region_cost=lower_region_cost,
                    canvas_edge_margins_normalized=[left/W, top/H, (W-left-rw)/W, (H-top-rh)/H],
                    minimum_canvas_edge_margin=min(left/W, top/H, (W-left-rw)/W, (H-top-rh)/H))

    candidates = []
    if explicit_anchor is not None:
        # Shrink, do not silently move the requested normalized foot point.
        x, y = explicit_anchor[0] * W, explicit_anchor[1] * H
        for _ in range(80):
            p = evaluate(x, y, 0, 'explicit_anchor')
            if p:
                candidates = [p]
                break
            if rw <= 1 and rh <= 1:
                break
            rw, rh = max(1, int(rw * .9)), max(1, int(rh * .9))
            ax, ay = dog['anchor'][0] * rw / dog['width'], dog['anchor'][1] * rh / dog['height']
        if not candidates:
            raise ValueError('指定anchor不能在不越界/不覆盖检测人脸的条件下放置狗')
    else:
        if mode == 'auto':
            edges = cv2.Canny(gray, 60, 150)
            temporal_edges = [cv2.Canny(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),60,150) for f in sampled]
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                                    minLineLength=max(15, int(sw * .12)), maxLineGap=12)
            # OpenCV bindings return either N,1,4 or N,4 depending on version.
            line_rows = [] if lines is None else np.asarray(lines).reshape(-1, 4)[:150].tolist()
            # Unrestricted probabilistic Hough can consume fragmented rail edges
            # into many diagonals in water/foliage. A near-horizontal accumulator
            # retains those real long supports; demand observed edge coverage.
            aligned = cv2.HoughLines(edges, 1, np.pi / 360,
                                     threshold=max(18, int(sw * .35)),
                                     min_theta=np.pi / 2 - .10, max_theta=np.pi / 2 + .10)
            if aligned is not None:
                for rho, theta in np.asarray(aligned).reshape(-1, 2)[:40]:
                    cols = np.arange(sw)
                    ys = (float(rho) - cols * math.cos(float(theta))) / math.sin(float(theta))
                    observed = np.zeros(sw, dtype=bool)
                    for offset in (-1, 0, 1):
                        rows = np.rint(ys).astype(int) + offset
                        valid = (rows >= 0) & (rows < sh)
                        observed[valid] |= edges[rows[valid], cols[valid]] > 0
                    support_x = np.flatnonzero(observed)
                    if len(support_x) < max(12, sw * .3):
                        continue
                    lx, rx = int(support_x[0]), int(support_x[-1])
                    if len(support_x) / (rx - lx + 1) < .5:
                        continue
                    line_rows.append([lx, float(ys[lx]), rx, float(ys[rx])])
            verticals = [v for v in line_rows if abs(float(v[2] - v[0])) <= abs(float(v[3] - v[1])) * .25
                         and abs(float(v[3] - v[1])) >= sh * .06]
            for line in line_rows:
                x1, y1, x2, y2 = map(float, line)
                if min(y1, y2) / sh < .52:
                    continue  # Exclude upper-frame ceiling/distant structures, not semantic detection.
                if abs(y2 - y1) > max(1, abs(x2 - x1)) * .22:
                    continue
                xa, xb = max(0, int(min(x1, x2))), min(sw, int(max(x1, x2)) + 1)
                ly = max(0, min(sh - 1, round((y1 + y2) / 2)))
                band, gap = max(2, round(sh * .035)), max(1, round(sh * .006))
                above = (max(0, ly - band - gap), max(0, ly - gap))
                below = (min(sh, ly + gap + 1), min(sh, ly + band + gap + 1))
                def strip_mean(arr, bounds):
                    region = arr[bounds[0]:bounds[1], xa:xb]
                    return float(np.mean(region)) if region.size else 0.
                texture_above = strip_mean(grad, above)
                texture_below = strip_mean(grad, below)
                smooth_both = max(0., 1 - max(texture_above, texture_below) / .10)
                length_fraction = abs(x2 - x1) / sw
                longness = min(1., max(0., (length_fraction - .5) / .35))
                normalized_y = ly / sh
                mid_upper = min(1., max(0., (.72 - normalized_y) / .12))
                horizon_penalty = .26 * longness * smooth_both * mid_upper
                lower_support_bonus = .075 * min(1., max(0., (normalized_y - .45) / .45))
                # Darker material below a bright edge is a weak top-surface cue;
                # the opposite polarity is often the underside of a beam/shelf.
                polarity = (strip_mean(gray, above) - strip_mean(gray, below)) / 255
                top_edge_bonus = .06 * max(-1., min(1., polarity))
                # A coherent, low-texture material strip immediately BELOW an
                # edge is a useful beam/shelf cue, unlike random water edge votes.
                material_bonus = .22 * max(0., 1 - texture_below / .08) * min(1., max(0., polarity / .35))
                # Thin image-bottom borders should not outrank interior supports;
                # this is a generic framing penalty, not a target anchor coordinate.
                border_penalty = .20 * min(1., max(0., (normalized_y - .90) / .10))
                for k in (.25, .5, .75):
                    px, py = x1 + k * (x2 - x1), y1 + k * (y2 - y1)
                    # Demand edge observations across the actual foot span, not a
                    # long Hough segment assembled from disconnected ripple votes.
                    half = max(5, round(rw * sw / W * .30))
                    cx0, cx1 = max(0,round(px)-half), min(sw,round(px)+half+1)
                    cols = np.arange(cx0,cx1)
                    if len(cols) < 2*half*.9:
                        continue
                    slope = (y2-y1) / max(1.,x2-x1)
                    rows = np.rint(py+(cols-px)*slope).astype(int)
                    coverage = []
                    for edge in temporal_edges:
                        votes = np.zeros(len(cols),bool)
                        for offset in (-2,-1,0,1,2):
                            rr = np.clip(rows+offset,0,sh-1)
                            votes |= edge[rr,cols] > 0
                        coverage.append(float(votes.mean()))
                    continuity = min(coverage) if coverage else 0.
                    # Real-world support is still heuristic: reject ambiguous blue
                    # smooth expanses and thin/discontinuous distant lines.
                    patch = hsv[min(sh-1,round(py)+3):min(sh,round(py)+band*2+3),cx0:cx1]
                    blue_fraction = float(np.mean((patch[...,0]>85)&(patch[...,0]<125)&(patch[...,1]>35))) if patch.size else 0.
                    if continuity < .72:
                        continue
                    foot_patch = gray[min(sh-1,round(py)+3):min(sh,round(py)+round(sh*.12)),cx0:cx1]
                    gx = float(np.mean(np.abs(cv2.Sobel(foot_patch,cv2.CV_32F,1,0)))) if foot_patch.size else 0.
                    gy = float(np.mean(np.abs(cv2.Sobel(foot_patch,cv2.CV_32F,0,1)))) if foot_patch.size else 0.
                    ripple_ratio = gy / max(1.,gx)
                    if blue_fraction > .45 and texture_below < .12:
                        continue
                    if material_bonus < .035 and (texture_below > .09 or (ripple_ratio > 1.6 and normalized_y < .80)):
                        continue
                    vertical_reward = 0.
                    for vx1, vy1, vx2, vy2 in verticals:
                        vx, vtop, vbottom = float(vx1 + vx2) / 2, float(min(vy1, vy2)), float(max(vy1, vy2))
                        # Reward only a nearby perpendicular structure that reaches
                        # the support line; not every vertical elsewhere in the shot.
                        distance_y = max(vtop - py, py - vbottom, 0.)
                        if distance_y <= sh * .08:
                            proximity = max(0., 1 - abs(vx - px) / max(sw * .3, 1.))
                            reach = max(0., 1 - distance_y / max(sh * .08, 1.))
                            vertical_reward = max(vertical_reward, proximity * reach)
                    geometry = dict(line_normalized=[x1 / sw, y1 / sh, x2 / sw, y2 / sh],
                                    local_edge_continuity=continuity, temporal_support_samples=len(coverage),
                                    blue_smooth_expanses_fraction=blue_fraction, horizontal_ripple_ratio=ripple_ratio,
                                    texture_above=texture_above, texture_below=texture_below,
                                    horizon_penalty=horizon_penalty, lower_support_bonus=lower_support_bonus,
                                    nearby_vertical_score=vertical_reward, top_edge_bonus=top_edge_bonus,
                                    material_below_bonus=material_bonus, image_border_penalty=border_penalty,
                                    cost_adjustment=horizon_penalty + border_penalty - lower_support_bonus
                                                    - .10 * vertical_reward - top_edge_bonus - material_bonus)
                    p = evaluate(px * W / sw, py * H / sh,
                                 min(1., length_fraction), 'hough_horizontal_support', geometry)
                    if p:
                        candidates.append(p)
        if mode == 'auto':
            if prefer_inset:
                # Twelve bounded, meaningful interior canvas centers; no extra
                # detector pass and no unbounded dense center search.
                for cx in (.22, .40, .60, .78):
                    for foot_y in (.60, .74, .88):
                        left = cx * W - rw / 2
                        top = min(foot_y * H - ay, H - rh - H * .08)
                        p = evaluate(left + ax, top + ay, 0, 'screen_sticker_inset')
                        if p:
                            candidates.append(p)
            # Always compete with supports: a line must beat usable lower-frame
            # whitespace instead of forcing a remote ceiling/horizon placement.
            for fx in (.035, .12, .88, .965):
                for fy in (.70, .80, .89, .94):
                    p = evaluate(ax + fx * (W - rw), min(H - rh + ay - H*.065, H * fy),
                                 0, 'screen_sticker')
                    if p:
                        candidates.append(p)
        if not candidates:
            if mode == 'auto':
                warnings.append('无可用可靠支撑候选：退回角落/边缘留白，不声称理解地面。')
            # Corners first; if faces block every corner use conservative perimeter grid.
            corners = [(0, 1), (1, 1), (0, 0), (1, 0)]
            for fx, fy in corners:
                p = evaluate(ax + fx * (W - rw), ay + (H*.065 if fy == 0 else H-rh-H*.065), 0, 'screen_sticker')
                if p:
                    candidates.append(p)
            if not candidates:
                for fx in np.linspace(0, 1, 9):
                    for fy in ((.25,.4,.55,.65,.8,1.) if user_regions and mode == 'auto' else
                               (0.,.15,.3,.45,.65,.8,1.) if mode == 'auto' else (0.,.5,1.)):
                        p = evaluate(ax + fx * (W - rw), ay + H*.065 + fy * max(0,H-rh-H*.13), 0, 'screen_sticker')
                        if p:
                            candidates.append(p)
            if not candidates:
                raise PlacementError('找不到不越界且避开人脸/人物/文字/身体衣服/人工禁区的放置位置',rejections)
    best = min(p['cost'] for p in candidates)
    inset_candidates = ([p for p in candidates if p['minimum_canvas_edge_margin'] >= .06 - 1e-6
                         and p['cost'] <= best + .15] if prefer_inset else [])
    pool = inset_candidates or candidates
    best_pool = min(p['cost'] for p in pool)
    shortlist = [p for p in pool if p['cost'] <= best_pool + .035]
    chosen = dict(rng.choice(shortlist))
    chosen['inset_preference_applied'] = prefer_inset
    chosen['inset_candidate_count'] = len(inset_candidates)
    chosen['edge_fallback'] = prefer_inset and chosen['minimum_canvas_edge_margin'] < .06 - 1e-6
    if chosen['edge_fallback']:
        warnings.append('当前最大安全尺寸无合适内侧留白候选；已边缘fallback，不强制中心或覆盖检测保护。')
    chosen.update(candidate_count=len(candidates), best_candidate_count=len(shortlist),
                  ranked_candidates=sorted(candidates, key=lambda p: p['cost'])[:12],
                  detected_faces=faces, detected_person_regions=people,
                  user_avoid_regions_normalized=list(user_regions or []), user_avoid_regions_pixels=user_avoid,
                  detected_text_like_regions=avoid, corner_reserve_policy='four corners 17% width x 5.5% height; manual anchor bypasses reserves only',
                  placement_mode='surface' if chosen['kind']=='hough_horizontal_support' else 'screen_sticker',
                  ground_anchored=False, contact_shadow=False,
                  haar_available=haar, hog_available=hog, warnings=warnings,
                  union_bbox_source=dog['union_bbox'], whole_canvas_bounds_checked=True,
                  actual_height_ratio=rh / H)
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
        alpha = src[:, :, 3:4]
        premul = np.concatenate([src[:, :, :3] * alpha, alpha], axis=2)
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
