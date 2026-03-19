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
| **#004** | 2026-03-19 | XGBClassifier (FE) | `experiment_xgb_fe.py` 생성. 도메인 지식 기반 파생 변수 6개 추가 (Binning, 비율 분할, 자동결제 여부 등) | 5-Fold SKF | 0.91619 | 0.91343 | Baseline(0.91339) 무사히 돌파! 조금 더 파격적이거나 고도화된 FE 기법 필요확인 |
| **#005** | 2026-03-19 | XGBClassifier (Adv FE) | `experiment_xgb_fe_advanced.py` 생성. 상위권 커널의 타겟 인코딩 제외 FE(약정 진척도, 결제액 오차, 서비스 조합 등) 적용 | 5-Fold SKF | ~0.9161 | 0.91358 | 단일 모델. 단순 FE(#004: 0.91343) 대비 Public LB **0.00015 상승!** 고급 FE의 확실한 효과 입증 |
| **#006** | 2026-03-19 | XGBClassifier (FE + TE) | `experiment_xgb_fe_te.py` 실행. 외부 데이터 없이 K-Fold OOF 기반 Target Encoding 적용 | 5-Fold SKF | 0.91601 | 0.91347 | 단일 모델. 기존 005(0.91358) 대비 LB 점수 감소. 순수 내부 데이터 TE 한계치 도달 |

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

### Exp #002: XGBRegressor로의 실험적 전환
*   **목적:** 고객 이탈은 이진 분류 문제지만, 예측되는 확률값(Probability)의 랭킹(Ranking)으로 평가하는 ROC AUC 지표 특성상 수치형 회귀(Regressor) 트리가 이점을 가질 수 있는지 테스트.
*   **결과 요약:** OOF 0.91508, LB 0.91255로 베이스라인(0.91608) 대비 소폭 하락. 단일 모델로는 약점이 있으나 트리 분기 방식이 완전히 다르므로 추후 앙상블(블렌딩) 시 높은 시너지가 날 수 있음을 확인.

### Exp #003: 첫 번째 앙상블 (가중 평균 블렌딩)
*   **목적:** `XGBClassifier` 뼈대에 `XGBRegressor` 결과값을 소량 조미료처럼 혼합(Blending).
*   **설정:** Classifier(0.6) + Regressor(0.4) 가중평균
*   **결과 요약:** LB 0.91329 달성. 단일 Regressor보단 높지만 단일 베이스라인(0.91339) 벽을 넘지 못함. 

### Exp #004: 도메인 지식 파생 변수 기초 (Feature Engineering)
*   **목적:** 통신사 도메인 지식을 바탕으로 새로운 정보를 창출해 모델에 먹여줌.
*   **적용:** 자동결제 여부, 부양가족 유무 통합, 부가서비스 총 결제 개수, 연차별 구간화 등 총 6종 피처 추가.
*   **결과 요약:** OOF 0.91619, LB 0.91343 도달. 드디어 베이스라인 돌파 성공! 잘 만들어진 파생 변수가 앙상블보다 훨씬 점수 향상에 기여함을 입증.

### Exp #005: 0.918+ 랭커들의 고급 FE 벤치마킹 (TE는 제외)
*   **목적:** 상위권 유저들이 창조한 변태적인(?) 피처들(약정 기간 자체 진척도, 예상 요금 vs 실제 결제액 오차, VIP 특수 등)을 차용. 오버피팅을 낳기 쉬운 TE(타겟 인코딩)는 일부러 떼고 순수 피처 퀄리티만 검증함.
*   **결과 요약:** 로컬 OOF는 ~0.9161로 제자리였으나, 실제 평가인 LB에서 **0.91358 (+0.00015)** 로 눈에 띄는 상승! 모델의 일반화 성능이 한층 강화됨을 증명.

### Exp #006: 기초 Target Encoding 세팅 (내부 데이터 OOF 모델링)
*   **목적:** 0.918 이상 달성을 위한 필수 스킬인 '타겟 인코딩(Target Encoding)'을 최초 설계 및 도입. 단, 외부 추가 데이터 도움 없이 현재 캐글 데이터 안에서만 5-Fold를 철저히 나눠 Leakage(컨닝) 없이 통계를 뽑음.
*   **적용:** 단순 컬럼 뿐 아니라 `Contract`와 `InternetService` 등 핵심 변수 쌍(Pair)을 곱해 만든 파생그룹에 이탈확률(Mean)과 이탈편차(Std) 반영.
*   **결과 요약:** OOF 0.91601. 수치가 뒤로 미세하게 밀림. 제한된 데이터 내부 통계만으론 TE가 큰 위력을 발휘하지 못하거나 노이즈(Noise)로 작용할 수도 있음. (현재 LB 평가 대기 중, 필요시 다음 턴에 외부 IBM 원본 데이터 7천 장을 긴급 수혈할 명분 발생)
