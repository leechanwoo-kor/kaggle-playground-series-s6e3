# 알고리즘 및 작동 원리 문서 (Algorithm & Strategies)

이 문서는 본 프로젝트에서 시도해 보는 모든 데이터 전처리 기법, 모델 종류, 그리고 학습 파라미터 세팅 등 **실험적/알고리즘적 결정사항**들을 상세히 남겨두는 곳입니다.

---

## 🚀 모델 아키텍처 (Models Used)

### 1. XGBClassifier (Baseline)
- **파일:** `baseline_model.py`
- **목적 함수:** `binary:logistic`
- **구동 원리:** 전형적인 이진 분류 모델. 로지스틱 함수를 이용하여 값을 0부터 1 사이 확률로 맵핑시킵니다.
- **초기 파라미터:** `learning_rate=0.05`, `max_depth=5`, `n_estimators=500`

### 2. XGBRegressor (Ranking Strategy)
- **파일:** `experiment_xgbregressor.py`
- **목적 함수:** `reg:squarederror`
- **구동 원리:** 타겟 값을 0과 1의 실수값으로 보고 예측 오차를 구합니다. 하지만 트리(Tree)가 각 피처 공간을 쪼개면서 내뱉는 연속적인 수치를 기반으로 AUC 랭킹(순위)을 파악하는 데 효과적으로 쓰이며, 향후 앙상블 모델의 다양성(Diversity)을 크게 높여 줍니다.

### 3. Weighted Blending (앙상블 기법)
- **파일:** `ensemble_blend.py`
- **전략:** Classifier의 높은 정확성(비중 60%)과 Regressor의 다양한 피처 쪼개기 순위(비중 40%)를 가중 평균(Weighted Average)하여 시너지를 창출합니다.
- **캐글 규정:** 완전히 합법적인 표준 테크닉입니다.

---

## 🔧 전처리 및 피처 엔지니어링 (Preprocessing & FE)

### Baseline Encoding
- **결측치:** 별도 대체(Impute)를 우선 적용하지 않음 (XGBoost 내장 처리능력 의존)
- **범주형(Categorical):** Scikit-Learn의 `LabelEncoder`를 사용하여 `object`, `string` 문자를 정수형 ID로 단순 치환. 
- **Target 변수:** `Yes` $\rightarrow$ 1, `No` $\rightarrow$ 0 직관적 매핑

---

## 📈 검증 전략 (Validation Strategy)

### Stratified K-Fold (5 Splits)
- `Churn`(이탈) 타겟 클래스의 구성 비율 불균형을 고려하여, **일반 K-Fold가 아닌 Stratified K-Fold(SKF)**를 채택.
- 폴드(Fold) 수를 5개로 나눠 얻어진 5개의 검증 셋 확률값(OOF: Out-Of-Fold)을 기반으로 성능의 일반화 능력을 파악합니다.
- 최종 제출물은 이 5개의 모델이 예측한 값을 앙상블(평균 분할)하여 생성됩니다.
