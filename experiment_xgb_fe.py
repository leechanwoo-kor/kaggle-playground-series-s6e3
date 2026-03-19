import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import os

def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    sub = pd.read_csv('data/sample_submission.csv')
    return train, test, sub

def feature_engineering(df):
    """
    Kaggle Telco Customer Churn 데이터에 특화된 Feature Engineering.
    기존 데이터(df)를 받아 새로운 피처들이 추가된 데이터프레임을 반환합니다.
    """
    df = df.copy()
    
    # 1. 수치형 변수 변환 및 에러 처리 (빈 공간 ' ' 등을 NaN으로)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # 결측치를 tenure가 0인 경우 0으로 치환하거나 중앙값으로 채움
        df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = 0.0
    
    # 2. 파생 변수: 가족 유무 (SeniorCitizen, Partner, Dependents 조합)
    # Partner나 Dependents 값이 'Yes'이 하나라도 있으면 가족이 있다고 판단
    if 'Partner' in df.columns and 'Dependents' in df.columns:
        df['HasFamily'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
        
    # 3. 파생 변수: 전체 가입된 부가 서비스 수 (Total Services)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    available_cols = [col for col in service_cols if col in df.columns]
    
    if available_cols:
        df['TotalServices'] = (df[available_cols] == 'Yes').sum(axis=1)
        
    # 4. 파생 변수: 자동 결제 여부
    if 'PaymentMethod' in df.columns:
        df['IsAutoPayment'] = df['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
        
    # 5. 비율/수학적 결합 변수: 
    # - 실제 지불한 월 평균 요금 (TotalCharges / tenure)
    # tenure가 0인 경우 0으로 처리 방지
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['ActualMonthlyCharges_Ratio'] = df['TotalCharges'] / (df['tenure'] + 0.001)
        
    # 6. 범주 분할(Binning): 가입 기간(tenure)을 특정 연차 그룹으로 묶음
    if 'tenure' in df.columns:
        df['Tenure_Group'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 60, 100], labels=['0-1Y', '1-2Y', '2-4Y', '4-5Y', '5Y+'])
        
    return df

def preprocess(train, test):
    # Combine for easy preprocessing
    train_len = len(train)
    df = pd.concat([train.drop(['Churn', 'id'], axis=1, errors='ignore'), test.drop(['id'], axis=1, errors='ignore')], axis=0, ignore_index=True)
    
    # --- Feature Engineering 적용 ---
    df = feature_engineering(df)
    # ---------------------------------
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    
    # Encode categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        df[col] = le.transform(df[col])
        
    X_train = df.iloc[:train_len].copy()
    X_test = df.iloc[train_len:].copy()
    y_train = train['Churn'].map({'Yes': 1, 'No': 0}).copy()
    
    if y_train.isnull().any():
        y_train = train['Churn'].copy()
        print("Warning: Target mapping resulted in nulls, using original target column.")
    
    return X_train, y_train, X_test

def train_and_evaluate(X, y, X_test):
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # 기존 Best 모델인 XGBClassifier 사용
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 500,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"Starting {n_splits}-fold Cross Validation with XGBClassifier (Feature Engineering Applied)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=100
        )
        
        val_preds = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_preds
        
        fold_auc = roc_auc_score(y_va, val_preds)
        print(f"Fold {fold+1} AUC: {fold_auc:.5f}\n")
        
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"Out-Of-Fold AUC (Feature Engineering): {oof_auc:.5f}")
    
    return test_preds, oof_auc

def main():
    print("Loading data...")
    train, test, sub = load_data()
    
    print("Preprocessing & Feature Engineering...")
    X_train, y_train, X_test = preprocess(train, test)
    
    preds, oof_score = train_and_evaluate(X_train, y_train, X_test)
    
    # Save submission
    os.makedirs('submissions', exist_ok=True)
    sub_filename = f'submissions/xgb_fe_cv{oof_score:.4f}.csv'.replace('.', '_', 1)
    
    sub['Churn'] = preds
    sub.to_csv(sub_filename, index=False)
    print(f"Submission saved to {sub_filename}")

if __name__ == '__main__':
    main()
