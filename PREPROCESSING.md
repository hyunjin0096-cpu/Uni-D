# Kvasir-SEG 전처리

실행: `python preprocess.py` (Python, Pillow, NumPy). 기존 출력이 있으면 덮어쓰지 않고 중단합니다. 원본 `Data/images`, `Data/masks`, JSON은 보존됩니다.

## 완료된 데이터

`Data/processed/`에 다음을 생성했습니다.

- `splits.json`: seed 42, 원본 이미지 기준 train 800 / val 100 / test 100.
- `validation_report.json`: 파일 대응·크기·Box 범위 검사 결과, 중복 및 유사도 후보 목록.
- `yolo/images`, `yolo/labels`: 원본 이미지 복사본 및 모든 Box의 class 0 정규화 XYWH 라벨. `data.yaml`이 데이터 경로입니다.
- `unet/images`, `unet/masks`: Box별 15% 사방 여백, 경계 제한, 동일 좌표 Crop, 비율 유지 resize 및 검정 padding을 적용한 352×352 PNG. RGB bilinear, 마스크 grayscale >127 이진화 후 nearest. 마스크 파일 값은 **0/1**이므로 일반 뷰어에서는 거의 검게 보입니다.
- `roi_manifest.json`: 원본 ID, Box, Crop, 원본 크기, resize 크기, padding 기록. train/val/test ROI는 858/111/102개입니다.

## 학습 시 적용할 처리

```python
from Model.data_pipeline import PolypROIDataset
train = PolypROIDataset('train')
val = PolypROIDataset('val')
image, mask = train[0]  # float32 (3,352,352), (1,352,352)
```

로더는 원본을 읽어 train에서만 중심·크기 ±10% jitter를 적용합니다. 15% margin 이후 원래 Box 면적 95% 이상을 보존하는 후보를 사용하며 실패하면 정답 Box로 돌아갑니다. 이미지/마스크에 동일한 양방향 반전·±10도 회전, 이미지에만 밝기·대비 ±15%와 약한 blur를 적용합니다. RGB를 0~1로 바꾼 뒤 ImageNet mean/std 정규화를 적용합니다. val/test는 고정 처리입니다. 저장된 PNG는 증강하지 않은 기본 ROI이며 정규화된 float tensor는 로더에서 생성합니다.

YOLO는 `yolo/train_args.yaml`을 학습 설정에 결합해 입력 크기 640, mosaic 제외, 반전·회전·밝기 변경을 지정합니다. 원본 크기의 이미지를 저장했으므로 YOLO 구현체에서 letterbox와 0~1 scaling을 수행해야 합니다. 해당 YAML은 Ultralytics 형식이며 다른 구현체는 설정을 변환해야 합니다. YOLO용 contrast/blur는 `Model.data_pipeline.yolo_train_photometric`에 구현했습니다. 선택한 YOLO 구현체의 train 전용 PIL RGB transform hook에 연결하면 됩니다. 모델 학습은 실행하지 않았습니다.

## 추론 및 평가

`prepare_prediction_roi`는 예측 xyxy Box로 동일한 전처리와 정규화를 수행합니다. `restore_probabilities`에 sigmoid 후 352×352 확률맵과 반환 메타데이터를 넘기면 padding 제거, 원본 Crop 크기로 bilinear 복원, 최대 확률 병합, threshold 0.5 이진화를 수행합니다. 탐지 0개이면 빈 마스크를 반환합니다.

정답 Box 기반 ROI 평가는 실제 detector를 거친 시스템 평가와 구분해야 합니다. 모델·가중치가 없어 YOLO 예측 ROI, Dice/IoU/작은 용종 recall 및 ablation 실험은 아직 실행하지 않았습니다. 배경 ROI 추가는 선택 실험이므로 기본 데이터에는 포함하지 않았습니다.

## 분할의 한계

RGB 픽셀 SHA-256 완전 중복 검사와 64-bit dHash Hamming 거리 ≤5 후보 검사를 수행했습니다. 후보들의 연결 그룹을 같은 분할에 배정한 뒤 정확히 800/100/100을 맞췄습니다. 유사도는 휴리스틱이므로 모든 유사 영상을 식별했다고 보장할 수 없으며 환자 ID가 없어 환자 단위 분리를 보장할 수 없습니다. 학습 재현성은 별도로 Python/NumPy/PyTorch 및 DataLoader worker seed를 고정해야 합니다.

