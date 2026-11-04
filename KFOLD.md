# Training/Validation 5-Fold 교차검증

기존 train 800장 + val 100장 = 900장만 교차검증에 사용합니다. 기존 test 100장은 모든 fold에서 제외하며 최종 평가에만 사용합니다. seed=42, 이미지 단위 5-fold이며 기존 유사·중복 이미지 연결 그룹은 분리하지 않습니다. 환자 ID가 없어 환자 단위 분리는 보장되지 않습니다. 클래스가 모두 polyp인 group-balanced 분할이며 크기별 층화 분할은 아닙니다.

`python make_kfold.py`로 재생성합니다. 이미지와 마스크는 복사하지 않고 기존 파일을 참조합니다. `Data/processed/kfold/fold_1`부터 `fold_5`에 이미지 ID 분할, YOLO 이미지 목록, U-Net 이미지·마스크 경로와 ROI 메타데이터가 있습니다. 파일 목록 경로는 `Data/processed` 기준입니다. `summary.json`은 개수와 검증 결과입니다.

## Colab

최신 저장소를 사용한 상태에서:

```python
import sys
from pathlib import Path
sys.path.insert(0, '/content/Uni-D')
from Model.kfold import yolo_fold_yaml
from Model.data_pipeline import PolypROIDataset

FOLD = 1  # 1~5
YAML = yolo_fold_yaml(FOLD)
train_dataset = PolypROIDataset('train', fold=FOLD)
val_dataset = PolypROIDataset('val', fold=FOLD)
print(YAML, len(train_dataset), len(val_dataset))
```

YOLO의 `model.train(data=str(YAML), ...)`에 생성한 설정을 전달합니다. `yolo_fold_yaml`은 실행 환경에 맞는 절대 경로 목록과 YAML을 `kfold/runtime`에 생성합니다. U-Net의 DataLoader에는 위 dataset을 사용합니다. 기존 로더처럼 원본에서 온라인 jitter/증강을 하므로 기존 `Data/images`, `Data/masks`가 필요합니다. 생성된 `unet_train.json`, `unet_val.json`의 `image`/`mask` 경로로 저장된 ROI만 읽는 별도 로더도 구현할 수 있지만 이 경우 온라인 Box jitter는 적용되지 않습니다. 저장 마스크는 0/1입니다.

각 fold는 YOLO와 U-Net을 반드시 새 모델/초기 가중치로 시작해야 합니다. 이전 fold의 학습 가중치·optimizer·early-stopping 상태를 이어 쓰지 마세요. 모든 파생 ROI는 원본 이미지와 같은 fold에 속합니다. 각 fold의 validation 결과 평균과 표준편차로 설정을 선택하고, test는 선택에 사용하지 않습니다. 최종 설정 선택 후 900장으로 재학습하여 test 100장으로 최종 평가할 수 있습니다. 이때 학습 epoch 등은 CV에서 미리 정합니다. 이 작업은 데이터 분할 구성과 검증이며 실제 모델 학습은 포함하지 않습니다.
