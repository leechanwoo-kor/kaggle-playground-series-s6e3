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

def preprocess(train, test):
    # Combine for easy preprocessing
    df = pd.concat([train.drop(['Churn', 'id'], axis=1, errors='ignore'), test.drop(['id'], axis=1, errors='ignore')], axis=0)
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Encode categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        df[col] = le.transform(df[col])
        
    X_train = df.iloc[:len(train)].copy()
    X_test = df.iloc[len(train):].copy()
    y_train = train['Churn'].map({'Yes': 1, 'No': 0}).copy()
    
    # Validation check in case the mapping fails (e.g. already numeric)
    if y_train.isnull().any():
        y_train = train['Churn'].copy()
        print("Warning: Target mapping resulted in nulls, using original target column.")
    
    return X_train, y_train, X_test

def train_and_evaluate(X, y, X_test):
    # Stratified K-Fold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # XGBoost Parameters for REGRESSOR
    xgb_params = {
        'objective': 'reg:squarederror', # 회귀 모델용 목적 함수
        'eval_metric': 'auc',            # 하지만 평가 지표는 여전히 AUC로 설정 가능
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 500,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"Starting {n_splits}-fold Cross Validation with XGBRegressor...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # XGBRegressor 사용
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=100
        )
        
        # Regressor이므로 predict_proba 대신 predict 사용
        val_preds = model.predict(X_va)
        oof_preds[val_idx] = val_preds
        
        fold_auc = roc_auc_score(y_va, val_preds)
        print(f"Fold {fold+1} AUC: {fold_auc:.5f}\n")
        
        # Predict on test set
        test_preds += model.predict(X_test) / n_splits
        
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"Out-Of-Fold AUC (Regressor): {oof_auc:.5f}")
    
    return test_preds, oof_auc

def main():
    print("Loading data...")
    train, test, sub = load_data()
    
    print("Preprocessing data...")
    X_train, y_train, X_test = preprocess(train, test)
    
    preds, oof_score = train_and_evaluate(X_train, y_train, X_test)
    
    # Save submission
    os.makedirs('submissions', exist_ok=True)
    sub_filename = f'submissions/xgb_regressor_cv{oof_score:.4f}.csv'.replace('.', '_', 1)
    
    # 예측값이 반드시 0~1 사이로 예쁘게 떨어지지 않을 수 있지만, AUC 평가에서는 순위(Ranking)만 보기 때문에 그대로 제출합니다.
    sub['Churn'] = preds
    sub.to_csv(sub_filename, index=False)
    print(f"Submission saved to {sub_filename}")

if __name__ == '__main__':
    main()
