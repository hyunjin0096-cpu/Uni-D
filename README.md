# Uni-D

YOLO → ResNet34 Encoder + U-Net Decoder를 위한 Kvasir-SEG 전처리 데이터와 코드입니다.

- 원본: `Data/images`, `Data/masks`, `Data/kavsir_bboxes.json`
- 처리 결과: `Data/processed` (train/val/test 800/100/100장)
- 실행: `pip install -r requirements.txt` 후 `python preprocess.py`
- 기존 처리 결과가 있으면 덮어쓰지 않고 중단합니다. 재생성하려면 기존 결과를 다른 위치에 보관하세요.
- 학습 로더 및 추론 ROI 복원: `Model/data_pipeline.py`
- 상세 설정·사용법·검증 범위: [PREPROCESSING.md](PREPROCESSING.md)

YOLO `data.yaml`은 파일 위치를 기준으로 이미지 경로를 해석하는 Ultralytics 형식입니다. 학습 프레임워크는 별도로 설치해야 합니다. 모델 학습 및 성능 평가는 아직 수행하지 않았습니다.

데이터 출처: [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/). 원본 데이터의 이용 조건과 인용 지침을 따라 사용하세요.
