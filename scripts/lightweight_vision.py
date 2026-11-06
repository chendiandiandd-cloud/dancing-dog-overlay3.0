"""Lightweight offline face and text-like heuristics, not OCR or semantics."""
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np


@lru_cache(maxsize=1)
def cascades():
    root = Path(getattr(getattr(cv2,'data',None),'haarcascades',''))
    models = []
    for name in ('haarcascade_frontalface_default.xml','haarcascade_eye.xml'):
        path = root/name
        model = cv2.CascadeClassifier(str(path)) if path.is_file() else None
        models.append(model if model is not None and not model.empty() else None)
    return models


def face_boxes(rgb):
    model, eye = cascades()
    if model is None:
        return [], False
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    H,W = gray.shape
    result = []
    for x,y,w,h in model.detectMultiScale(gray,1.1,5,minSize=(20,20)):
        area = w*h/(W*H)
        if area > .30 or w > W*.65 or h > H*.60:
            continue
        if area > .075:
            eyes = eye.detectMultiScale(gray[y:y+int(h*.65),x:x+w],1.1,3,
                                       minSize=(max(5,w//10),max(5,h//10))) if eye is not None else []
            if len(eyes) < 1:
                continue
        result.append([int(x),int(y),int(x+w),int(y+h)])
    return result, True


def text_regions(rgb):
    """Small high-contrast strokes grouped into horizontal text-like runs."""
    h,w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    response = cv2.morphologyEx(gray,cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    binary = (response>55).astype(np.uint8)*255
    n,labels,stats,_ = cv2.connectedComponentsWithStats(binary)
    keep = np.zeros(n,np.uint8)
    for label,(x,y,bw,bh,area) in enumerate(stats[1:],1):
        letter = 2<=bh<=max(8,h*.045) and 1<=bw<=max(12,w*.055)
        joined = 3<=bh<=max(8,h*.04) and bw/bh>5 and w*.08<bw<w*.80
        if (letter or joined) and area>=3:
            keep[label] = 255
    strokes = keep[labels]
    grouped = cv2.morphologyEx(strokes,cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT,(max(5,w//50),3)))
    contours,_ = cv2.findContours(grouped,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        x,y,bw,bh = cv2.boundingRect(contour)
        density = float(np.mean(strokes[y:y+bh,x:x+bw]>0))
        if bw>=max(14,w*.045) and 3<=bh<=h*.07 and bw/bh>2.2 and .10<density<.8:
            result.append([max(0,x-3),max(0,y-3),min(w,x+bw+3),min(h,y+bh+3)])
    return result


def head_regions(rgb):
    """At most four bounded head-like cues for hats/shades missed by Haar.

    Color is only evidence inside compact head-shaped proposals; it is NOT a
    full-frame skin/body protection mask. A bounded integral-window search uses
    lower-cheek color plus an upper dark band and lateral contrast. Compact
    hat/head proposals additionally need dark-brim or neutral-clothing context.
    Warm scenery/pets/hands can still false-positive: every cue remains SOFT.
    """
    H,W = rgb.shape[:2]
    ycc = cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    r,g,b = [rgb[...,k].astype(np.int16) for k in range(3)]
    skin = ((ycc[...,0]>45)&(ycc[...,1]>130)&(ycc[...,1]<185)&
            (ycc[...,2]>75)&(ycc[...,2]<140)&(r-g>5)&(r-b>5)&(hsv[...,1]<170)).astype(np.uint8)
    smooth = cv2.morphologyEx(skin,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    smooth = cv2.morphologyEx(smooth,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    integral = {k:cv2.integral(v) for k,v in
                [('skin',skin),('dark',(gray<100).astype(np.uint8)),
                 ('neutral',((hsv[...,1]<65)&(gray>80)).astype(np.uint8))]}

    def mean(k,rect):
        x,y,x2,y2 = [int(v) for v in rect]
        x,y,x2,y2 = max(0,x),max(0,y),min(W,x2),min(H,y2)
        if x2<=x or y2<=y:
            return 0.
        a = integral[k]
        return float((a[y2,x2]-a[y,x2]-a[y2,x]+a[y,x])/((x2-x)*(y2-y)))

    proposals = []
    _,_,stats,_ = cv2.connectedComponentsWithStats(smooth)
    for x,y,w,h,area in stats[1:]:
        if not (.02*H<=h<=.16*H and .025*W<=w<=.23*W and
                .6<w/h<2.1 and .3<area/(w*h)<.96):
            continue
        if not (.15<(x+w/2)/W<.85 and .08<y/H<.65):
            continue
        cloth = mean('neutral',[x,y+h,x+w,y+2*h])
        brim = mean('dark',[x,y+h*.25,x+w,y+h*.65])
        if cloth>.30 or brim>.15:
            proposals.append((float(.6+cloth*.2+brim*.2),
                              [int(x-w*.15),int(y-h*.1),int(x+w*1.15),int(y+h*1.25)],
                              'compact_hat_head'))
    # Four sizes x eleven x eleven positions; O(1) integral reads per window.
    for h in (H*.08,H*.12,H*.18,H*.25):
        w = h*.8
        for cx in np.linspace(.22*W,.78*W,11):
            for cy in np.linspace(.15*H,.55*H,11):
                x,y = cx-w/2,cy-h/2
                # Broad face windows belong to the upper portrait region;
                # lower arm/hand patches are not promoted to large head cues.
                if y < H*.05 or y > H*.32:
                    continue
                cheek = mean('skin',[x+w*.12,y+h*.45,x+w*.88,y+h*.95])
                eye = mean('dark',[x,y+h*.05,x+w,y+h*.45])
                side = max(mean('skin',[x-w*.4,y+h*.45,x,y+h*.95]),
                           mean('skin',[x+w,y+h*.45,x+w*1.4,y+h*.95]))
                bottom = mean('skin',[x,y+h,x+w,y+h*1.45])
                if cheek<.50 or eye<.16 or cheek-side<.10:
                    continue
                score = cheek*.6+eye*.2+(cheek-side)*.3+max(0,cheek-bottom)*.2
                proposals.append((score,[round(x),round(y),round(x+w),round(y+h)],
                                  'dark_band_cheek_head'))
    proposals.sort(reverse=True)
    picked = []
    for score,box,kind in proposals:
        if score<.6:
            continue
        x,y,x2,y2 = box
        duplicate = False
        for found in picked:
            a,b,c,d = found['box']
            intersection = max(0,min(x2,c)-max(x,a))*max(0,min(y2,d)-max(y,b))
            if intersection/max(1,min((x2-x)*(y2-y),(c-a)*(d-b)))>.3:
                duplicate = True
                break
        if not duplicate:
            picked.append(dict(confidence=min(1.,float(score)),
                               box=[max(0,x),max(0,y),min(W,x2),min(H,y2)],cue=kind))
        if len(picked)==4:
            break
    return picked
