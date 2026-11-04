"""Build grouped CV manifests without copying images or touching held-out test."""
import json
import random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'Data' / 'processed'

def main(k=5, seed=42):
    splits = json.loads((DATA/'splits.json').read_text())
    report = json.loads((DATA/'validation_report.json').read_text())
    rows = json.loads((DATA/'roi_manifest.json').read_text())
    pool = sorted(splits['train'] + splits['val'])
    test = set(splits['test'])
    parent = {key:key for key in pool}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    all_groups = report['similarity_groups'] + report['exact_duplicate_groups']
    for group in all_groups:
        assert not (set(group)&test and set(group)&set(pool)), 'Existing test leakage'
        members = [key for key in group if key in parent]
        for key in members[1:]: parent[find(key)] = find(members[0])
    groups = {}
    for key in pool: groups.setdefault(find(key), []).append(key)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    folds = [[] for _ in range(k)]
    for group in groups:
        dest = min(range(k), key=lambda i:len(folds[i]))
        folds[dest].extend(group)
    destination = DATA/'kfold'
    destination.mkdir(exist_ok=True)
    summary = []
    seen = Counter()
    for i, val in enumerate(folds, 1):
        val = set(val); train = set(pool)-val
        assert train.isdisjoint(val) and (train|val).isdisjoint(test)
        for group in all_groups:
            assert not (set(group)&train and set(group)&val)
        fold = destination/f'fold_{i}'
        fold.mkdir(exist_ok=True)
        manifest = dict(fold=i, seed=seed, train=sorted(train), val=sorted(val), test=sorted(test))
        (fold/'splits.json').write_text(json.dumps(manifest, indent=2))
        counts = {}
        for name, keys in [('train',train), ('val',val)]:
            selected = [dict(r, original_split=r['split'], split=name) for r in rows if r['source_id'] in keys]
            # Original paths are retained; no physical ROI duplication is needed.
            for r in selected:
                r['image'] = f"unet/images/{r['original_split']}/{r['file']}"
                r['mask'] = f"unet/masks/{r['original_split']}/{r['file']}"
            (fold/f'unet_{name}.json').write_text(json.dumps(selected, indent=2))
            paths = []
            for key in sorted(keys):
                original = 'train' if key in splits['train'] else 'val'
                matches = list((DATA/'yolo/images'/original).glob(key+'.*'))
                assert len(matches)==1
                path = matches[0]
                assert (DATA/'yolo/labels'/original/(key+'.txt')).exists()
                paths.append(path.relative_to(DATA).as_posix())
            (fold/f'yolo_{name}.txt').write_text('\n'.join(paths)+'\n')
            for r in selected:
                assert (DATA/r['image']).exists() and (DATA/r['mask']).exists()
            counts[name] = dict(images=len(keys), rois=len(selected))
        seen.update(val)
        summary.append(dict(fold=i, **counts))
    assert set(seen)==set(pool) and set(seen.values())=={1}
    result = dict(k=k, seed=seed, development_images=len(pool), held_out_test_images=len(test),
                  folds=summary, checks='PASS: disjoint train/val/test; grouped similarity; each development image validates exactly once; all paths and labels exist',
                  limitation='No patient IDs; similarity grouping is heuristic. Test must not be used for fold/model selection.')
    (destination/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
