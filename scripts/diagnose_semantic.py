"""Finite real-frame feasibility check; writes masks/evidence, never video."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from semantic_protection import protector, MODEL_SHA256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    s=protector(); reports=[]
    for n,path in enumerate(a.inputs):
        cap=cv2.VideoCapture(path)
        duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
        times=([0,.5,1.,1.5,2.0,2.4,2.8,3.2,3.6,4.] if n==0 else
               [0,.3,.6,.9,1.2,1.5,1.8,2.1,2.4,9.])
        records=[]; tiles=[]; union=None
        for t in times:
            if t>=duration: continue
            cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,bgr=cap.read()
            if not ok: continue
            scale=min(1.,320/max(bgr.shape[:2])); bgr=cv2.resize(bgr,(round(bgr.shape[1]*scale),round(bgr.shape[0]*scale)))
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            categories=s.categories(rgb); mask=s.mask_from_categories(categories)
            if union is None: union=np.zeros_like(mask)
            union |= mask
            fractions={label:float((categories==s.labels.index(label)).mean()) for label in ('person','cat','dog','background')}
            records.append(dict(time=t,protected_fraction=float(mask.mean()),categories=fractions))
            tint=np.zeros_like(bgr); tint[:,:,2]=255
            evidence=bgr.copy(); evidence[mask>0]=(bgr[mask>0]*.55+tint[mask>0]*.45).astype(np.uint8)
            pair=np.concatenate([bgr,evidence],axis=1)
            canvas=np.full((pair.shape[0]+25,pair.shape[1],3),245,np.uint8); canvas[25:]=pair
            cv2.putText(canvas,f'{t:.2f}s protected={mask.mean():.3f}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
            tiles.append(canvas)
        cap.release()
        sheet=np.concatenate(tiles,axis=0)
        evidence_path=a.output_dir/f'{n+1:02d}_semantic_evidence.jpg'
        cv2.imwrite(str(evidence_path),sheet)
        reports.append(dict(input=path,samples=records,union_fraction=float(union.mean()),evidence=str(evidence_path)))
    result=dict(model_sha256=MODEL_SHA256,reports=reports)
    (a.output_dir/'feasibility.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
