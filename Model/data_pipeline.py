"""Online train-only augmentation and ROI restoration (numpy/Pillow/torch)."""
import json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from preprocess import crop_box, letterbox

def yolo_train_photometric(image):
    """Optional train-only PIL RGB transform before YOLO's own letterbox/scaling.

    Contrast and mild blur preserve boxes. YOLO YAML already supplies brightness.
    Call only on training images in the detector's dataset transform hook.
    """
    image=ImageEnhance.Contrast(image).enhance(random.uniform(.85,1.15))
    if random.random()<.15:
        image=image.filter(ImageFilter.GaussianBlur(random.uniform(.1,.7)))
    return image

class PolypROIDataset(Dataset):
    def __init__(self, split='train', augment=True, jitter=.10, size=352, fold=None):
        if split not in ('train','val','test'): raise ValueError(split)
        self.rows=[r for r in json.loads((ROOT/'Data/processed/roi_manifest.json').read_text()) if r['split']==split]
        if fold is not None:
            if split == 'test':
                raise ValueError('Test is held out; use fold=None for final test evaluation')
            manifest=ROOT/'Data/processed/kfold'/f'fold_{fold}'/f'unet_{split}.json'
            self.rows=json.loads(manifest.read_text())
        self.augment=augment and split=='train'
        self.jitter=jitter if self.augment else 0
        self.size=size
        self.images={p.stem:p for p in (ROOT/'Data/images').iterdir()}
        self.masks={p.stem:p for p in (ROOT/'Data/masks').iterdir()}

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]; key=row['source_id']
        with Image.open(self.images[key]) as f: image=f.convert('RGB')
        with Image.open(self.masks[key]) as f: mask=Image.fromarray((np.array(f.convert('L'))>127).astype('uint8'))
        box=row['box']
        if self.jitter:
            x1,y1,x2,y2=box; w,h=x2-x1,y2-y1
            # Retry to retain >=95% of annotated box area after margin/clipping.
            for _ in range(20):
                cx=(x1+x2)/2+random.uniform(-self.jitter,self.jitter)*w
                cy=(y1+y2)/2+random.uniform(-self.jitter,self.jitter)*h
                nw=w*(1+random.uniform(-self.jitter,self.jitter))
                nh=h*(1+random.uniform(-self.jitter,self.jitter))
                candidate=[cx-nw/2,cy-nh/2,cx+nw/2,cy+nh/2]
                a,b,c,d=crop_box(candidate,image.size)
                retained=max(0,min(c,x2)-max(a,x1))*max(0,min(d,y2)-max(b,y1))/(w*h)
                if retained>=.95: box=candidate; break
        crop=crop_box(box,image.size)
        image,mask,_=letterbox(image.crop(crop),mask.crop(crop),self.size)
        if self.augment:
            for transpose in [Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM]:
                if random.random()<.5: image=image.transpose(transpose); mask=mask.transpose(transpose)
            angle=random.uniform(-10,10)
            image=image.rotate(angle,resample=Image.Resampling.BILINEAR,fillcolor=0)
            mask=mask.rotate(angle,resample=Image.Resampling.NEAREST,fillcolor=0)
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.85,1.15))
            image=ImageEnhance.Contrast(image).enhance(random.uniform(.85,1.15))
            if random.random()<.15: image=image.filter(ImageFilter.GaussianBlur(random.uniform(.1,.7)))
        x=np.array(image,dtype=np.float32)/255
        x=(x-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(np.array(mask,dtype=np.float32)[None].copy())

def prepare_prediction_roi(image,box,size=352):
    """Input is RGB PIL image and detector xyxy box; never uses ground truth."""
    crop=crop_box(box,image.size)
    if crop[2]<=crop[0] or crop[3]<=crop[1]: raise ValueError('Invalid detector box')
    roi,_,meta=letterbox(image.crop(crop),size=size)
    x=np.array(roi,dtype=np.float32)/255
    x=(x-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
    return torch.from_numpy(x.transpose(2,0,1).copy()),dict(crop=crop,**meta)

def restore_probabilities(probabilities,metadata,original_size,threshold=.5):
    """Merge probabilities by maximum, then threshold. Empty detection => empty mask."""
    if len(probabilities)!=len(metadata): raise ValueError('Mismatched predictions/metadata')
    w,h=original_size; full=np.zeros((h,w),dtype=np.float32)
    for prob,meta in zip(probabilities,metadata):
        if isinstance(prob,torch.Tensor): prob=prob.detach().cpu().numpy()
        prob=np.asarray(prob,dtype=np.float32).squeeze()
        left,top,_,_=meta['padding']; nw,nh=meta['resized_size']
        unpadded=prob[top:top+nh,left:left+nw]
        x1,y1,x2,y2=meta['crop']
        resized=np.asarray(Image.fromarray(unpadded).resize((x2-x1,y2-y1),Image.Resampling.BILINEAR))
        full[y1:y2,x1:x2]=np.maximum(full[y1:y2,x1:x2],resized)
    return full,(full>=threshold).astype('uint8')
