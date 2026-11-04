"""Full-clip sparse LK/RANSAC follow, with conservative freeze-on-failure.
No 3-D geometry, optical occlusion, scale/rotation rendering or neural inference.
"""
from __future__ import annotations
import bisect
import cv2
import numpy as np
from classical_vision import sparse_motion, mask_box_fraction
from overlay_core import video_frames, intersects, decode_image


def tracking_offset(motion, time):
    points = motion.get('trajectory', [])
    if not points or motion.get('mode') == 'off':
        return 0., 0.
    i = bisect.bisect_right([p['time'] for p in points], time)-1
    if i < 0:
        return 0., 0.
    a = points[i]
    if i+1 == len(points):
        return a['dx'], a['dy']
    b = points[i+1]
    # Failure freezes at last safe offset, without interpolating a rejected jump.
    f = np.clip((time-a['time'])/max(1e-8,b['time']-a['time']),0,1)
    return float(a['dx']+(b['dx']-a['dx'])*f), float(a['dy']+(b['dy']-a['dy'])*f)


def safe_offset(position, offset, W, H):
    dx,dy = np.rint(offset).astype(int)
    rect = [position['x']+dx, position['y']+dy,
            position['x']+dx+position['width'], position['y']+dy+position['height']]
    if rect[0]<0 or rect[1]<0 or rect[2]>W or rect[3]>H:
        return False, 'boundary'
    boxes = position.get('detected_faces',[]) + position.get('detected_person_regions',[]) + position.get('detected_text_like_regions',[])
    if any(intersects(rect,b) for b in boxes):
        return False, 'sampled_protection_conflict'
    if any(intersects(rect,b) for b in position.get('user_avoid_regions_pixels',[])):
        return False, 'user_avoid_region_conflict'
    if position.get('garment_mask_png'):
        mask = decode_image(position['garment_mask_png'])[:,:,0]
        r = [rect[0]*mask.shape[1]/W,rect[1]*mask.shape[0]/H,
             rect[2]*mask.shape[1]/W,rect[3]*mask.shape[0]/H]
        if mask_box_fraction(mask,r) > .015:
            return False, 'sampled_garment_conflict'
    return True, 'safe'


def follow_frames(frames, plan, sampling_fps, width, height):
    """Shared by actual video analysis and synthetic tests. Streaming O(1) frames."""
    info = plan['media']; W,H = info['display_width'],info['display_height']
    previous, previous_shot = None, -1
    frozen = False; offset = np.zeros(2)
    starts = [s['start'] for s in plan['shots']]
    for k, frame in enumerate(frames):
        t = k/sampling_fps
        si = max(0,bisect.bisect_right(starts,t)-1)
        shot = plan['shots'][si]; p = shot['placement']; motion = shot['motion']
        if si != previous_shot:
            offset = np.zeros(2); frozen = False
            previous = frame; previous_shot = si
            motion['trajectory'] = [dict(time=shot['start'],dx=0.,dy=0.,state='cut_reset')]
            continue
        if motion['mode'] == 'off':
            previous = frame; continue
        mask = np.full((height,width),255,np.uint8)
        for b in p.get('detected_faces',[])+p.get('detected_person_regions',[])+p.get('detected_text_like_regions',[]):
            x0,y0,x1,y1 = b
            mask[max(0,int(y0*height/H)):min(height,int(np.ceil(y1*height/H))),
                 max(0,int(x0*width/W)):min(width,int(np.ceil(x1*width/W)))] = 0
        if p.get('garment_mask_png'):
            g = cv2.resize(decode_image(p['garment_mask_png'])[:,:,0],(width,height),interpolation=cv2.INTER_NEAREST)
            mask[g>0]=0
        # Try local background around the CURRENT foot; do not follow person legs.
        ax,ay = p['anchor_pixel'] + offset
        cx,cy = ax*width/W, ay*height/H
        local = np.zeros_like(mask)
        rx,ry = max(width*.18,p['width']*width/W),max(height*.16,p['height']*height/H)
        local[max(0,int(cy-ry)):min(height,int(cy+ry)),max(0,int(cx-rx)):min(width,int(cx+rx))] = 255
        matrix, stats = sparse_motion(previous,frame,cv2.bitwise_and(mask,local))
        source = 'local'
        if matrix is None:
            matrix,stats = sparse_motion(previous,frame,mask); source='camera'
        motion['attempted_pairs'] += 1
        if stats['ok']:
            motion['accepted_pairs'] += 1
            motion['confidence_sum'] += stats['confidence']
            motion[source+'_pairs'] += 1
        else:
            reason = stats['reason']
            motion['rejection_reasons'][reason] = motion['rejection_reasons'].get(reason,0)+1
        state = 'fixed_safe_sticker' if motion['mode']=='off' else 'following'
        if motion['mode'] != 'off' and not frozen:
            if matrix is None:
                frozen = True; state='confidence_freeze'
            else:
                foot = np.array([cx,cy])
                proposed = (matrix[:,:2]@foot+matrix[:,2]-foot)*[W/width,H/height]+offset
                good,reason = safe_offset(p,proposed,W,H)
                # Check swept full canvas, not only sample endpoints: interpolated
                # render frames cannot cut across a protected face/text region.
                swept = dict(p)
                lower=np.floor(np.minimum(offset,proposed)).astype(int)
                upper=np.ceil(np.maximum(offset,proposed)).astype(int)
                swept.update(x=p['x']+int(lower[0]),y=p['y']+int(lower[1]),
                             width=p['width']+int(upper[0]-lower[0]),
                             height=p['height']+int(upper[1]-lower[1]))
                sweep_good,sweep_reason=safe_offset(swept,[0,0],W,H)
                if good and not sweep_good:
                    good=False; reason='swept_'+sweep_reason
                if not good or np.linalg.norm((proposed-offset)/[W,H]) > .05:
                    frozen=True; state=reason if not good else 'step_limit'
                    motion['rejection_reasons'][state]=motion['rejection_reasons'].get(state,0)+1
                else:
                    offset=proposed
            if frozen:
                motion['freeze_at'] = t
        elif frozen:
            state='frozen'
        if motion['mode'] != 'off':
            motion['trajectory'].append(dict(time=t,dx=float(offset[0]),dy=float(offset[1]),state=state))
        previous = frame
    for shot in plan['shots']:
        m=shot['motion']; n=m['attempted_pairs']
        # No observations are not a zero match rate: report unavailable explicitly.
        m['accepted_ratio']=m['accepted_pairs']/n if n else None
        confidence_sum=m.pop('confidence_sum')
        m['mean_confidence']=confidence_sum/m['accepted_pairs'] if m['accepted_pairs'] else None
        if not n and m['mode'] != 'off':
            m['skipped_reason']='insufficient_frame_pairs'
        if plan['tracking']=='auto' and m['mode']!='off' and (not n or m['accepted_ratio']<.85 or 'freeze_at' in m):
            m.update(mode='off',reason='full_clip_confidence_or_safety_fallback')
            m['trajectory']=[dict(time=shot['start'],dx=0.,dy=0.,state='auto_static_fallback')]
        m['max_displacement_pixels']=max((float(np.hypot(p['dx'],p['dy'])) for p in m['trajectory']),default=0.)
        m['limitation']='2-D foot translation only; no 3-D, scale/rotation warp, occlusion or semantic ground'


def plan_tracking(path, plan, timeout):
    info=plan['media']; requested=plan['tracking']
    for shot in plan['shots']:
        reason=('image' if info['kind']=='image' else 'explicit_anchor' if plan['requested_anchor'] is not None
                else 'temporal_safe_windows' if shot.get('placement_segments') else 'requested_off' if requested=='off' else 'safe_screen_sticker' if requested=='auto' and shot['placement']['placement_mode']!='surface'
                else 'lk_ransac_candidate')
        shot['motion']=dict(mode='flow' if reason=='lk_ransac_candidate' else 'off',reason=reason,
                            attempted_pairs=0,accepted_pairs=0,local_pairs=0,camera_pairs=0,
                            confidence_sum=0.,rejection_reasons={},trajectory=[])
        if reason != 'lk_ransac_candidate':
            shot['motion']['skipped_reason']=reason
    # Fixed stickers cannot use optical flow. Avoid the extra decode entirely
    # when every shot is fixed; mixed clips still skip LK on fixed shots.
    if not any(s['motion']['mode']=='flow' for s in plan['shots']):
        follow_frames([],plan,1,1,1)
        plan['tracking_sampling_fps']=None
        plan['tracking_skipped_reason']='no_flow_eligible_shots'
        return
    factor=min(1,320/max(info['display_width'],info['display_height']))
    w,h=max(1,round(info['display_width']*factor)),max(1,round(info['display_height']*factor))
    sampling=min(6.,info['fps'])
    frames=video_frames(path,info,w,h,sampling,timeout)
    try:
        follow_frames(frames,plan,sampling,w,h)
    finally:
        frames.close()
    plan['tracking_sampling_fps']=sampling
    plan['warnings'].append('全片稀疏LK/RANSAC二维跟随；auto非可靠支撑面固定screen_sticker，低置信度/边界/保护冲突静止回退，无3D或遮挡承诺。')
