"""Portable Colab/local YOLO CV configuration."""
import json
from pathlib import Path

def yolo_fold_yaml(fold, data_root=None):
    root = Path(data_root) if data_root else Path(__file__).resolve().parents[1]/'Data/processed'
    root = root.resolve()
    source = root/'kfold'/f'fold_{fold}'
    if not (source/'splits.json').exists(): raise ValueError(f'Unknown fold: {fold}')
    runtime = root/'kfold'/'runtime'/f'fold_{fold}'
    runtime.mkdir(parents=True, exist_ok=True)
    for split in ('train','val'):
        paths = (source/f'yolo_{split}.txt').read_text().splitlines()
        (runtime/f'{split}.txt').write_text('\n'.join((root/p).as_posix() for p in paths)+'\n')
    config = runtime/'data.yaml'
    # Absolute paths are generated on the host, never committed to the repository.
    config.write_text(f'train: {json.dumps((runtime/"train.txt").as_posix())}\nval: {json.dumps((runtime/"val.txt").as_posix())}\nnames:\n  0: polyp\n')
    return config
