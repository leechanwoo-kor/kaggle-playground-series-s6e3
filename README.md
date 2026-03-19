# Kaggle Playground Series - Season 6, Episode 3
**Customer Churn Prediction (이탈 예측)**

## 📌 프로젝트 소개
본 프로젝트는 특정 서비스 고객들의 이탈 여부(Churn: Yes/No)를 예측하는 캐글 경진대회(이진 분류 문제)의 코드 및 실험 관리 저장소입니다.

## 🎯 평가 지표 (Metric)
- **ROC AUC** (Area Under the Receiver Operating Characteristic Curve)

## 📁 주요 디렉토리 구조
- `data/`: `train.csv`, `test.csv`, `sample_submission.csv` 등 원본 데이터 보관
- `docs/`: 대회 규정, EDA 기록, 그리고 `experiment_log.md` (중요: 실험 이력 관리) 저장
- `submissions/`: 캐글 리더보드에 제출할 결과물(CSV)이 생성되는 곳
- `baseline_model.py`: 분류(Classifier) 모델 기준 기본 동작 파이프라인
- `experiment_xgbregressor.py`: 회귀(Regressor) 기반의 앙상블 적용 테스트 스크립트

## 🛠 실행 안내 (Getting Started)
1. 환경 세팅: `requirements.txt` 또는 `cc` 가상환경 이용
2. `python baseline_model.py` 등을 실행하여 `submissions` 스코어 생성
3. 결과물 등록 후, 반드시 `docs/experiment_log.md` 갱신!
