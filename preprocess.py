"""Reproducible Kvasir-SEG preparation; originals are never modified."""
import json, hashlib, random, shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Data' / 'processed'
SEED = 42

def crop_box(box, size, margin=.15):
    import math
    x1,y1,x2,y2 = box
    dx,dy = (x2-x1)*margin,(y2-y1)*margin
    return [max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),
            min(size[0],math.ceil(x2+dx)),min(size[1],math.ceil(y2+dy))]

def letterbox(image, mask=None, size=352):
    w,h=image.size
    ratio=min(size/w,size/h)
    nw,nh=max(1,round(w*ratio)),max(1,round(h*ratio))
    left,top=(size-nw)//2,(size-nh)//2
    canvas=Image.new('RGB',(size,size),(0,0,0))
    canvas.paste(image.resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
    target=None
    if mask is not None:
        target=Image.new('L',(size,size),0)
        target.paste(mask.resize((nw,nh),Image.Resampling.NEAREST),(left,top))
    return canvas,target,dict(resized_size=[nw,nh],padding=[left,top,size-nw-left,size-nh-top])

def main():
    if OUT.exists():
        raise RuntimeError('Output already exists; preserve it or choose a new output path before rerunning.')
    boxes=json.loads((ROOT/'Data/kavsir_bboxes.json').read_text())
    images={p.stem:p for p in (ROOT/'Data/images').iterdir() if p.is_file()}
    masks={p.stem:p for p in (ROOT/'Data/masks').iterdir() if p.is_file()}
    assert images.keys()==masks.keys()==boxes.keys(), 'Unpaired filenames'
    ids=sorted(images)
    hashes=[]; exact={}; sizes={}; foreground={}
    for key in ids:
        with Image.open(images[key]) as raw, Image.open(masks[key]) as m:
            raw.load(); m.load()
            assert raw.size==m.size==(boxes[key]['width'],boxes[key]['height'])
            rgb=raw.convert('RGB'); binary=np.array(m.convert('L'))>127
            assert binary.any(), f'Empty mask: {key}'
            foreground[key]=int(binary.sum()); sizes[key]=list(raw.size)
            for b in boxes[key]['bbox']:
                assert b['label']=='polyp'
                assert 0<=b['xmin']<b['xmax']<=raw.width and 0<=b['ymin']<b['ymax']<=raw.height
            exact.setdefault(hashlib.sha256(rgb.tobytes()+str(rgb.size).encode()).hexdigest(),[]).append(key)
            # Difference hash: candidates are grouped conservatively before splitting.
            small=np.array(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS))
            bits=(small[:,1:]>small[:,:-1]).ravel()
            hashes.append(int.from_bytes(np.packbits(bits).tobytes(),'big'))
    parent=list(range(len(ids)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        parent[find(j)]=find(i)
    near=[]
    for i in range(len(ids)):
        for j in range(i):
            dist=(hashes[i]^hashes[j]).bit_count()
            if dist<=5:
                union(i,j); near.append([ids[j],ids[i],dist])
    index={k:i for i,k in enumerate(ids)}
    for group in exact.values():
        for k in group[1:]: union(index[group[0]],index[k])
    groups={}
    for i,key in enumerate(ids): groups.setdefault(find(i),[]).append(key)
    grouped=list(groups.values()); random.Random(SEED).shuffle(grouped)
    grouped.sort(key=len,reverse=True)
    split={s:[] for s in ['train','val','test']}
    targets=dict(train=round(len(ids)*.8),val=round(len(ids)*.1))
    targets['test']=len(ids)-sum(targets.values())
    for group in grouped:
        fitting=[s for s in split if targets[s]-len(split[s])>=len(group)]
        assert fitting, 'Similarity groups cannot fit exact split sizes'
        selected=max(fitting,key=lambda s:targets[s]-len(split[s]))
        split[selected].extend(group)
    OUT.mkdir()
    (OUT/'splits.json').write_text(json.dumps(split,indent=2))
    rows=[]
    for s,keys in split.items():
        for folder in [f'yolo/images/{s}',f'yolo/labels/{s}',f'unet/images/{s}',f'unet/masks/{s}']:
            (OUT/folder).mkdir(parents=True)
        for key in sorted(keys):
            shutil.copy2(images[key],OUT/f'yolo/images/{s}'/images[key].name)
            w,h=sizes[key]; lines=[]
            with Image.open(images[key]) as raw, Image.open(masks[key]) as mask:
                raw=raw.convert('RGB'); mask=Image.fromarray((np.array(mask.convert('L'))>127).astype('uint8'))
                for i,b in enumerate(boxes[key]['bbox']):
                    x1,y1,x2,y2=[b[n] for n in ['xmin','ymin','xmax','ymax']]
                    lines.append(f'0 {(x1+x2)/(2*w):.10f} {(y1+y2)/(2*h):.10f} {(x2-x1)/w:.10f} {(y2-y1)/h:.10f}')
                    crop=crop_box([x1,y1,x2,y2],(w,h))
                    image,target,meta=letterbox(raw.crop(crop),mask.crop(crop))
                    name=f'{key}__{i:02d}.png'
                    image.save(OUT/f'unet/images/{s}'/name)
                    target.save(OUT/f'unet/masks/{s}'/name)
                    rows.append(dict(source_id=key,split=s,box_index=i,box=[x1,y1,x2,y2],crop=crop,original_size=[w,h],file=name,**meta))
            (OUT/f'yolo/labels/{s}/{key}.txt').write_text('\n'.join(lines)+'\n')
    (OUT/'roi_manifest.json').write_text(json.dumps(rows,indent=2))
    report=dict(seed=SEED,images=len(ids),boxes=sum(len(b['bbox']) for b in boxes.values()),
                split_images={s:len(v) for s,v in split.items()},split_rois=dict(Counter(r['split'] for r in rows)),
                box_count_histogram=dict(Counter(len(b['bbox']) for b in boxes.values())),
                exact_duplicate_groups=[g for g in exact.values() if len(g)>1],similarity_candidates=near,
                similarity_groups=[g for g in groups.values() if len(g)>1],
                similarity_method='64-bit dHash Hamming <=5; transitive groups stay in one split. Heuristic, not patient-level assurance.',
                patient_split_limitation='Patient IDs unavailable; patient-level separation cannot be guaranteed.')
    (OUT/'validation_report.json').write_text(json.dumps(report,indent=2))
    (OUT/'yolo/data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: polyp\n')
    (OUT/'yolo/train_args.yaml').write_text('imgsz: 640\nmosaic: 0.0\nmixup: 0.0\ncopy_paste: 0.0\ndegrees: 10.0\nfliplr: 0.5\nflipud: 0.5\nhsv_h: 0.0\nhsv_s: 0.0\nhsv_v: 0.15\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['similarity_candidates','similarity_groups']},indent=2))

if __name__=='__main__': main()
