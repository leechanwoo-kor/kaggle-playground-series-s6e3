import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    sub = pd.read_csv('data/sample_submission.csv')
    return train, test, sub

def feature_engineering_advanced(df):
    df = df.copy()
    
    # 1. Base Numerics
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0.0)
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0)
        
    # 2. Financial Deviations
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 0.001)
    df['total_vs_expected'] = df['TotalCharges'] / (df['MonthlyCharges'] * df['tenure'] + 0.001)
    df['total_minus_expected'] = df['TotalCharges'] - (df['MonthlyCharges'] * df['tenure'])
    
    # 3. Contract Math
    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24, 
                    'month-to-month': 1, 'one year': 12, 'two year': 24}
    if 'Contract' in df.columns:
        df['contract_length'] = df['Contract'].map(contract_map).fillna(1.0)
        df['contract_progress'] = df['tenure'] / (df['contract_length'] + 0.001)
        
    # 4. Service Usage
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    available_cols = [col for col in service_cols if col in df.columns]
    if available_cols:
        df['n_services'] = (df[available_cols] == 'Yes').sum(axis=1)
        df['charges_per_service'] = df['MonthlyCharges'] / (df['n_services'] + 1)
        
    # 5. Customer Segmentation Flags
    df['is_high_value'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].median()) & (df['tenure'] > 24)).astype(int)
    
    # Existing features
    if 'Partner' in df.columns and 'Dependents' in df.columns:
        df['HasFamily'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
    if 'PaymentMethod' in df.columns:
        df['is_autopay'] = df['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
        
    # 7. Categorical Crosses (For TE)
    cat_pairs = [
        ('Contract', 'InternetService'),
        ('PaymentMethod', 'Contract'),
        ('InternetService', 'OnlineSecurity')
    ]
    for c1, c2 in cat_pairs:
        if c1 in df.columns and c2 in df.columns:
            df[f'{c1}__{c2}'] = df[c1].astype(str) + '_' + df[c2].astype(str)
            
    # Convert numerics to categorical bins for TE
    df['tenure_bin'] = pd.cut(df['tenure'], bins=10, labels=False).fillna(0).astype(str)
    df['monthly_charge_bin'] = pd.cut(df['MonthlyCharges'], bins=10, labels=False).fillna(0).astype(str)
            
    return df

def apply_target_encoding(X_tr, y_tr, X_va, X_te_fold, te_cols):
    stats = ['mean', 'std']
    
    X_tr_out = X_tr.copy()
    X_va_out = X_va.copy()
    X_te_out = X_te_fold.copy()
    
    for col in te_cols:
        for s in stats:
            X_tr_out[f'TE_{col}_{s}'] = 0.0
            
    kf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Inner K-Fold for X_tr
    for in_tr_idx, in_va_idx in kf_inner.split(X_tr, y_tr):
        X_in_tr = X_tr.iloc[in_tr_idx]
        y_in_tr = y_tr.iloc[in_tr_idx]
        X_in_va = X_tr.iloc[in_va_idx]
        
        for col in te_cols:
            tmp = pd.DataFrame({'val': X_in_tr[col], 'target': y_in_tr})
            agg = tmp.groupby('val')['target'].agg(stats)
            agg.columns = [f'TE_{col}_{s}' for s in stats]
            
            merged = X_in_va[[col]].merge(agg, left_on=col, right_index=True, how='left')
            for s in stats:
                X_tr_out.iloc[in_va_idx, X_tr_out.columns.get_loc(f'TE_{col}_{s}')] = merged[f'TE_{col}_{s}'].values
                
    # 2. Apply to X_va and X_te_fold using entire X_tr
    for col in te_cols:
        tmp = pd.DataFrame({'val': X_tr[col], 'target': y_tr})
        agg = tmp.groupby('val')['target'].agg(stats)
        agg.columns = [f'TE_{col}_{s}' for s in stats]
        
        merged_va = X_va[[col]].merge(agg, left_on=col, right_index=True, how='left')
        merged_te = X_te_fold[[col]].merge(agg, left_on=col, right_index=True, how='left')
        
        for s in stats:
            X_va_out[f'TE_{col}_{s}'] = merged_va[f'TE_{col}_{s}'].values
            X_te_out[f'TE_{col}_{s}'] = merged_te[f'TE_{col}_{s}'].values
            
            global_val = y_tr.mean() if s == 'mean' else 0
            X_tr_out[f'TE_{col}_{s}'] = X_tr_out[f'TE_{col}_{s}'].fillna(global_val)
            X_va_out[f'TE_{col}_{s}'] = X_va_out[f'TE_{col}_{s}'].fillna(global_val)
            X_te_out[f'TE_{col}_{s}'] = X_te_out[f'TE_{col}_{s}'].fillna(global_val)

    return X_tr_out, X_va_out, X_te_out

def main():
    print("Loading data...")
    train, test, sub = load_data()
    
    print("Preprocessing & Advanced Feature Engineering...")
    train_len = len(train)
    df = pd.concat([train.drop(['Churn', 'id'], axis=1, errors='ignore'), test.drop(['id'], axis=1, errors='ignore')], axis=0, ignore_index=True)
    df = feature_engineering_advanced(df)
    
    # 2. Identify TE columns before Label Encoding
    te_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    
    # Temporarily save original string categories for TE calculation
    df_te_base = df.copy()
    
    # Apply standard Label Encoding to native categorical features (for XGBoost)
    for col in te_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    X = df.iloc[:train_len].copy()
    X_test = df.iloc[train_len:].copy()
    
    # However, TE needs the original text categories to match easily (or we can use encoded, works structurally too).
    # Since we mapped them, the TE logic (using numeric encoded values or text) is mathematically same! So it's fine.
    
    y = train['Churn'].map({'Yes': 1, 'No': 0}).copy()
    if y.isnull().any():
        y = train['Churn'].copy()

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
    
    print(f"Starting {n_splits}-fold CV with XGBClassifier (Adv FE + TE without external)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        X_va, y_va = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
        X_te_fold = X_test.copy()
        
        # Apply TE securely
        X_tr, X_va, X_te_fold = apply_target_encoding(X_tr, y_tr, X_va, X_te_fold, te_cols)
        
        if fold == 0:
            print(f"Number of Features used: {X_tr.shape[1]}")
            
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
        
        test_preds += model.predict_proba(X_te_fold)[:, 1] / n_splits
        
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"Out-Of-Fold AUC (Advanced FE + TE): {oof_auc:.5f}")
    
    os.makedirs('submissions', exist_ok=True)
    sub_filename = f'submissions/xgb_fe_te_cv{oof_auc:.4f}.csv'.replace('.', '_', 1)
    
    sub['Churn'] = test_preds
    sub.to_csv(sub_filename, index=False)
    print(f"Submission saved to {sub_filename}")

if __name__ == '__main__':
    main()
