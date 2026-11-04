"""Offline classical protection and sparse 2-D motion; no learned neural weights.
All coordinates in the supplied RGB image. Conservative cues, NOT OCR/semantics.
"""
from __future__ import annotations
import math
import cv2
import numpy as np


def text_regions(rgb):
    """Small high-contrast connected strokes grouped into horizontal text-like runs."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    response = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    binary = (response > 55).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    keep = np.zeros(n, np.uint8)
    for label, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        letter = 2 <= bh <= max(8,h*.045) and 1 <= bw <= max(12,w*.055)
        joined_text = 3 <= bh <= max(8,h*.04) and bw/bh > 5 and w*.08 < bw < w*.80
        if (letter or joined_text) and area >= 3:
            keep[label] = 255
    strokes = keep[labels]
    grouped = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w // 50), 3)))
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        density = float(np.mean(strokes[y:y+bh, x:x+bw] > 0))
        # Text-like only: group geometry and stroke density, not character recognition.
        if bw >= max(14, w * .045) and 3 <= bh <= h * .07 and bw / bh > 2.2 and .10 < density < .8:
            result.append([max(0, x-3), max(0, y-3), min(w, x+bw+3), min(h, y+bh+3)])
    return result


def reserved_regions(h, w):
    # Small unconditional reserves at FOUR corners; do not claim detected watermarks.
    cw, ch = w * .17, h * .055
    return [[0, 0, cw, ch], [w-cw, 0, w, ch],
            [0, h-ch, cw, h], [w-cw, h-ch, w, h]]


def garment_mask(rgb, faces):
    """Face-seeded local GrabCut + conservative widening toward hem/feet.
    Local crop and area gates prevent one false cascade box masking the entire frame.
    Face-derived core is provided separately by the caller even when segmentation fails.
    """
    h, w = rgb.shape[:2]
    protected = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in faces:
        fw, fh = x1-x0, y1-y0
        lx, rx = max(0, int(x0-1.35*fw)), min(w, int(x1+1.35*fw))
        ty, by = max(0, int(y0-.4*fh)), min(h, int(y1+9*fh))
        if rx-lx < 8 or by-ty < 8:
            continue
        crop = rgb[ty:by, lx:rx]
        mask = np.full(crop.shape[:2], cv2.GC_PR_BGD, np.uint8)
        mask[:2] = cv2.GC_BGD; mask[-2:] = cv2.GC_BGD
        mask[:, :2] = cv2.GC_BGD; mask[:, -2:] = cv2.GC_BGD
        a, b = max(2, int(x0-lx)), min(rx-lx-2, int(x1-lx))
        t, u = max(2, int(y0-ty)), min(by-ty-2, int(y1-ty))
        if b <= a or u <= t:
            continue
        mask[t:u, a:b] = cv2.GC_FGD
        # Clothing seed beneath neck, probable not definite foreground.
        mask[u:min(by-ty-2, u+int(4*fh)), a:b] = cv2.GC_PR_FGD
        # Clothing can differ completely from skin and be disconnected at the
        # neck in a pixel mask. Add a tiny definite local clothing seed only when
        # its color differs from BOTH crop-side background samples.
        cy0,cy1=min(by-ty-3,u+round(.6*fh)),min(by-ty-2,u+round(1.8*fh))
        ca,cb=max(a,(a+b)//2-max(1,round(fw*.20))),min(b,(a+b)//2+max(1,round(fw*.20)))
        clothing_seed=None
        if cy1>cy0 and cb>ca:
            lab=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB).astype(np.float32)
            core=np.median(lab[cy0:cy1,ca:cb].reshape(-1,3),axis=0)
            sides=[np.median(lab[cy0:cy1,:max(2,round(fw*.3))].reshape(-1,3),axis=0),
                   np.median(lab[cy0:cy1,-max(2,round(fw*.3)):].reshape(-1,3),axis=0)]
            if min(np.linalg.norm(core-s) for s in sides)>22:
                mask[cy0:cy1,ca:cb]=cv2.GC_FGD
                clothing_seed=(cy0,cy1,ca,cb)
        try:
            # Bound graph size: classical max-flow on a close-up must not dominate
            # runtime. Restore a conservative nearest-neighbour mask afterwards.
            factor=min(1.,256/max(crop.shape[:2]))
            size=(max(4,round(crop.shape[1]*factor)),max(4,round(crop.shape[0]*factor)))
            local_rgb=cv2.resize(crop,size,interpolation=cv2.INTER_AREA)
            local_mask=cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
            cv2.setRNGSeed(0)
            cv2.grabCut(local_rgb, local_mask, None, np.zeros((1,65),np.float64),
                        np.zeros((1,65),np.float64), 2, cv2.GC_INIT_WITH_MASK)
            mask=cv2.resize(local_mask,(crop.shape[1],crop.shape[0]),interpolation=cv2.INTER_NEAREST)
        except cv2.error:
            continue
        fg = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
        seeds = set(np.unique(labels[t:u, a:b])) - {0}
        if clothing_seed:
            sy0,sy1,sx0,sx1=clothing_seed
            seeds |= set(np.unique(labels[sy0:sy1,sx0:sx1]))-{0}
        fg = np.isin(labels, list(seeds)).astype(np.uint8)
        if clothing_seed:
            # GrabCut can label a differently colored skirt as background despite
            # the tiny clothing seed. Bounded color-connected growth keeps that
            # garment without flooding through unrelated background colors.
            color_match=(np.linalg.norm(lab-core,axis=2)<28).astype(np.uint8)
            _,color_labels,_,_=cv2.connectedComponentsWithStats(color_match)
            sy0,sy1,sx0,sx1=clothing_seed
            color_seeds=set(np.unique(color_labels[sy0:sy1,sx0:sx1]))-{0}
            grown=np.isin(color_labels,list(color_seeds)).astype(np.uint8)
            if .01<float(grown.mean())<.75:
                fg |= grown
        fraction = float(fg.mean())
        if not .015 < fraction < .85:
            continue
        radius = max(3, round(fw*.2))
        fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1)))
        protected[ty:by, lx:rx] |= fg
    return protected


def mask_box_fraction(mask, rect):
    x0, y0, x1, y1 = rect
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(mask.shape[1], math.ceil(x1)), min(mask.shape[0], math.ceil(y1))
    roi = mask[y0:y1, x0:x1]
    return float(np.mean(roi > 0)) if roi.size else 1.


def sparse_motion(before, after, mask=None):
    """Fresh Shi-Tomasi, bidirectional LK, RANSAC partial affine with strict gates."""
    a = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(a, 180, .015, 6, mask=mask, blockSize=5)
    result = dict(ok=False, reason='low_texture', features=0, inliers=0, confidence=0.)
    if points is None or len(points) < 8:
        return None, result
    result['features'] = len(points)
    q, s, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, winSize=(21,21), maxLevel=3)
    if q is None:
        result['reason'] = 'lk_failed'; return None, result
    back, sb, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(21,21), maxLevel=3)
    if back is None:
        result['reason'] = 'backward_lk_failed'; return None, result
    good = (s[:,0] > 0) & (sb[:,0] > 0) & (np.linalg.norm(back[:,0]-points[:,0],axis=1) < 1.2)
    p, q = points[:,0][good], q[:,0][good]
    if len(p) < 8:
        result['reason'] = 'few_bidirectional_points'; return None, result
    cv2.setRNGSeed(0)
    matrix, inlier = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
                                              ransacReprojThreshold=1.5, maxIters=500, confidence=.99)
    if matrix is None:
        result['reason'] = 'ransac_failed'; return None, result
    valid = inlier[:,0] > 0
    confidence = float(valid.mean())
    residual = float(np.median(np.linalg.norm(p[valid] @ matrix[:,:2].T + matrix[:,2]-q[valid],axis=1)))
    scale = float(np.hypot(matrix[0,0],matrix[1,0]))
    angle = float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
    result.update(inliers=int(valid.sum()), confidence=confidence, residual=residual,
                  scale=scale, rotation_degrees=angle)
    h,w = a.shape
    if valid.sum() < 8 or confidence < .65 or residual > 1.2:
        result['reason'] = 'low_consensus'; return None, result
    if not .96 < scale < 1.04 or abs(angle) > 3 or np.linalg.norm(matrix[:,2]/[w,h]) > .065:
        result['reason'] = 'excess_motion'; return None, result
    result.update(ok=True, reason='lk_ransac_2d')
    return matrix, result


def body_cue_mask(rgb, faces=None):
    """Fast conservative skin + skin-connected neutral garment cue, even without a face.

    Chroma coherence precedes closing; contour padding follows component scale.
    Neutral growth is bounded by local seed geometry, not global connectivity.
    Not skin/person semantics: smooth warm scenery can still false-positive;
    tiny/dark/occluded skin or unrelated colored garments can still be missed.
    """
    h, w = rgb.shape[:2]
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    r, g, b = [rgb[..., k].astype(np.int16) for k in range(3)]
    skin = ((ycc[..., 0] > 35) & (ycc[..., 1] > 132) & (ycc[..., 1] < 185)
            & (ycc[..., 2] > 75) & (ycc[..., 2] < 140)
            & (r-g > 8) & (r-b > 8)).astype(np.uint8)
    # Require local coherent chroma BEFORE closing: isolated warm blades, soil
    # and sunlit foliage must not become a body through morphological bridges.
    support = cv2.boxFilter(skin.astype(np.float32), -1, (5,5))
    skin &= (support >= .68).astype(np.uint8)
    # Ambiguous warm scenes need independent human evidence, not a wider RGB
    # threshold. Recover skin matching observed face chroma inside local body
    # context; without faces retain only compact, locally smooth components.
    if float(skin.mean()) > .35:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        variance = cv2.boxFilter(gray*gray,-1,(5,5)) - cv2.boxFilter(gray,-1,(5,5))**2
        smooth = variance < 144
        if faces:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            supported = np.zeros_like(skin)
            for x0,y0,x1,y1 in faces:
                fw,fh=x1-x0,y1-y0
                crop=lab[y0:y1,x0:x1]
                seed=skin[y0:y1,x0:x1]>0
                if seed.sum()<5:
                    continue
                palette=np.median(crop[seed],axis=0)
                match=(np.linalg.norm(lab[...,1:]-palette[1:],axis=2)<18)
                match &= np.abs(lab[...,0]-palette[0])<65
                context=np.zeros_like(skin)
                context[max(0,int(y0-.5*fh)):min(h,int(y1+9*fh)),
                        max(0,int(x0-2*fw)):min(w,int(x1+2*fw))]=1
                supported |= (match & smooth & (context>0)).astype(np.uint8)
            skin &= supported
        else:
            skin &= smooth.astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
    valid = [i for i, (_,_,bw,bh,area) in enumerate(stats) if i and
             area >= max(20,h*w*.0015) and area < h*w*.55 and max(bw,bh) >= min(h,w)*.04
             and not (bw > w*.85 and bh > h*.60)
             and not (sum(map(int, (stats[i,0]==0,stats[i,1]==0,
                                   stats[i,0]+bw==w,stats[i,1]+bh==h))) >= 3)]
    skin = np.isin(labels, valid).astype(np.uint8)
    if not np.any(skin):
        return skin
    radius = max(3, round(h*.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    contact = np.zeros_like(skin)
    reach = np.zeros_like(skin)
    for i in valid:
        x, y, bw, bh, area = stats[i]
        component = (labels == i).astype(np.uint8)
        # Distant small bodies need contour padding, not a frame-sized halo.
        pad = max(3, min(radius, round(min(bw,bh)*.20)))
        contact |= cv2.dilate(component, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*pad+1,2*pad+1)))
        # Garment growth is local to each seed's scale. A nearby neutral ground
        # component may be connected but is not promoted in its entirety.
        extent = max(2*pad, round(max(bw,bh)*.85))
        reach[max(0,y-extent):min(h,y+bh+extent),
              max(0,x-extent):min(w,x+bw+extent)] = 1
    # Grey/white clothing next to observed skin, including a skirt below the
    # torso; no face crop that truncates a bent arm or the outside of a hip.
    neutral = ((hsv[...,1] < 85) & (hsv[...,2] > 45)).astype(np.uint8)
    neutral = cv2.morphologyEx(neutral, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    _, regions, nstats, _ = cv2.connectedComponentsWithStats(neutral)
    connected = set(np.unique(regions[contact > 0])) - {0}
    connected = [i for i in connected if h*w*.002 < nstats[i,4] < h*w*.35
                 and nstats[i,1] > 0
                 and not (nstats[i,0] == 0 and nstats[i,0]+nstats[i,2] == w)]
    garment = np.isin(regions, connected).astype(np.uint8) & reach
    # Skin contour padding is already scale-aware; garment padding stays local.
    return contact | cv2.dilate(garment, kernel)
