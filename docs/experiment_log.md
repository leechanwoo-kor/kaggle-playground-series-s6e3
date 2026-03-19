# Kaggle Playground Series S6E3: Experiment Log

이 문서는 베이스라인 모델을 기준으로 진행되는 다양한 실험(모델, Feature Engineering, 하이퍼파라미터 변경 등)의 결과와 성과를 꼼꼼하게 추적하기 위해 작성되었습니다.

## 🛠 실험 환경 (Setup)
*   **평가 지표:** ROC AUC (Area Under the Receiver Operating Characteristic Curve)
*   **검증 전략:** 진행상황에 따라 5-fold StratifiedKFold 등 기본 전략 명시

---

## 📊 실험 기록 (Experiment Tracking)

| Exp ID | 날짜 (Date) | 모델 (Model) | 주요 변경 사항 (Description/Features) | CV 전략 | OOF AUC | LB (Public) | 비고 (Notes) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **#001** | 2026-03-19 | XGBoost (Baseline) | 기본 `baseline_model.py` 실행 (수치형 유지 + 범주형 Label Encoding), 하이퍼파라미터 초기값 | 5-Fold SKF | 0.91608 | 0.91339 | 향후 이 점수를 베이스라인 기준으로 삼을 예정입니다. |
| **#002** | 2026-03-19 | XGBRegressor | `experiment_xgbregressor.py` 생성. Classifier 대신 Regressor를 사용하여 연속값 산출 및 AUC 랭킹 활용 | 5-Fold SKF | 0.91508 | 0.91255 | Baseline(0.91608/0.91339) 대비 하락. 이후 앙상블 블렌딩 재료로 활용 예정 |
| **#003** | 2026-03-19 | Ensemble (Blend) | XGBClassifier(0.6) + XGBRegressor(0.4) 가중 평균(Weighted Average) 블렌딩 | - | (합성 산출물) | 0.91329 | 단일 XGBRegressor(0.91255) 보단 크게 올랐으나 Baseline(0.91339) 에는 미치지 못함 |
| **#004** | - | - | - | - | - | - | - |
| **#005** | - | - | - | - | - | - | - |

---

## 📝 상세 실행 기록 (Detailed Notes)

### Exp #001: 베이스라인 모델 구성 (Baseline XGBoost)
*   **목적:** 기본 파이프라인(데이터셋 로드, 5-Fold 검증, Target 변환 등) 검증 및 초기 점수 확보.
*   **Feature Engineering:** 타겟(Churn) 단순 매핑(`Yes/No` $\rightarrow$ `1/0`) 외에 파생변수 생성은 하지 않음. 범주형 변수(`object`, `string`)는 `sklearn.preprocessing.LabelEncoder` 적용. (`id` 컬럼 제거)
*   **파라미터:** 
    *   `objective`: binary:logistic
    *   `learning_rate`: 0.05
    *   `max_depth`: 5
    *   `n_estimators`: 500
*   **결과 요약:** OOF 예측 확률 값을 모아 ROC AUC 산출 후, 결과를 바탕으로 `submissions/xgb_baseline_cv[점수].csv` 파일 생성.
