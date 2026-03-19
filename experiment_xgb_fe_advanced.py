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

def feature_engineering_advanced(df):
    """
    Kaggle Telco Customer Churn: Advanced Feature Engineering (No Target Encoding)
    """
    df = df.copy()
    
    # 1. Base Numerics Cleanup
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0.0)
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0)
        
    # 2. Financial Deviations & Ratios
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 0.001)
    df['total_vs_expected'] = df['TotalCharges'] / (df['MonthlyCharges'] * df['tenure'] + 0.001)
    df['total_minus_expected'] = df['TotalCharges'] - (df['MonthlyCharges'] * df['tenure'])
    
    # 3. Contract Math
    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24, 
                    'month-to-month': 1, 'one year': 12, 'two year': 24}
    if 'Contract' in df.columns:
        df['contract_length'] = df['Contract'].map(contract_map).fillna(1.0)
        df['contract_progress'] = df['tenure'] / (df['contract_length'] + 0.001)
        df['charges_per_contract'] = df['TotalCharges'] / (df['contract_length'] + 0.001)
        df['is_monthly_contract'] = (df['Contract'].str.lower() == 'month-to-month').astype(int)
        
    # 4. Service Usage
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    available_cols = [col for col in service_cols if col in df.columns]
    if available_cols:
        df['n_services'] = (df[available_cols] == 'Yes').sum(axis=1)
        df['charges_per_service'] = df['MonthlyCharges'] / (df['n_services'] + 1)
        df['services_tenure'] = df['n_services'] * df['tenure']
        
    # 5. Customer Segmentation Flags
    df['is_high_value'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].median()) & (df['tenure'] > 24)).astype(int)
    df['is_new_customer'] = (df['tenure'] <= 3).astype(int)
    df['is_loyal_customer'] = (df['tenure'] > 36).astype(int)
    
    # Existing useful features from previous iteration
    if 'Partner' in df.columns and 'Dependents' in df.columns:
        df['HasFamily'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
    if 'PaymentMethod' in df.columns:
        df['is_autopay'] = df['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
        
    # 6. Statistical Distributions (Log1p & Sqrt)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in num_cols:
        if col in df.columns:
            # clip lower=0 to avoid log/sqrt domain errors
            clipped_series = df[col].clip(lower=0)
            df[f'LOG1P_{col}'] = np.log1p(clipped_series)
            df[f'SQRT_{col}'] = np.sqrt(clipped_series)
            
    # 7. Feature Crosses (Interactions)
    cat_pairs = [
        ('Contract', 'InternetService'),
        ('PaymentMethod', 'Contract'),
        ('InternetService', 'OnlineSecurity')
    ]
    for c1, c2 in cat_pairs:
        if c1 in df.columns and c2 in df.columns:
            df[f'{c1}__{c2}'] = df[c1].astype(str) + '_' + df[c2].astype(str)
            
    return df

def preprocess(train, test):
    train_len = len(train)
    df = pd.concat([train.drop(['Churn', 'id'], axis=1, errors='ignore'), test.drop(['id'], axis=1, errors='ignore')], axis=0, ignore_index=True)
    
    # --- Advanced Feature Engineering ---
    df = feature_engineering_advanced(df)
    
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
        
    return X_train, y_train, X_test

def train_and_evaluate(X, y, X_test):
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 500,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"Starting {n_splits}-fold CV with XGBClassifier (Advanced FE)...")
    print(f"Number of Features used: {X.shape[1]}")
    
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
    print(f"Out-Of-Fold AUC (Advanced FE): {oof_auc:.5f}")
    
    return test_preds, oof_auc

def main():
    print("Loading data...")
    train, test, sub = load_data()
    
    print("Preprocessing & Advanced Feature Engineering...")
    X_train, y_train, X_test = preprocess(train, test)
    
    preds, oof_score = train_and_evaluate(X_train, y_train, X_test)
    
    # Save submission
    os.makedirs('submissions', exist_ok=True)
    sub_filename = f'submissions/xgb_fe_adv_cv{oof_score:.4f}.csv'.replace('.', '_', 1)
    
    sub['Churn'] = preds
    sub.to_csv(sub_filename, index=False)
    print(f"Submission saved to {sub_filename}")

if __name__ == '__main__':
    main()
