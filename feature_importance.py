import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    train = pd.read_csv('data/train.csv')
    return train

def preprocess(train):
    df = train.drop(['Churn', 'id'], axis=1, errors='ignore').copy()
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Encode categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        df[col] = le.transform(df[col])
        
    y_train = train['Churn'].map({'Yes': 1, 'No': 0}).copy()
    if y_train.isnull().any():
        y_train = train['Churn'].copy()
        print("Warning: Target mapping resulted in nulls, using original.")
        
    return df, y_train, cat_cols

def main():
    print("Loading data...")
    train = load_data()
    
    print("Preprocessing data...")
    X, y, cat_cols = preprocess(train)
    
    # XGBoost Parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 200, # Less estimators for quick importance check
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("Training XGBoost on full train set to extract feature importances...")
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X, y)
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance (Gain)': model.feature_importances_
    }).sort_values(by='Importance (Gain)', ascending=False)
    
    print("\\n=== Top 20 Features by Importance ===")
    top20_str = importance_df.head(20).to_string(index=False)
    print(top20_str)
    
    os.makedirs('docs', exist_ok=True)
    with open('docs/feature_importances.txt', 'w') as f:
        f.write(top20_str)
    
    # Save plot
    os.makedirs('docs', exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance (Gain)', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 XGBoost Feature Importances')
    plt.tight_layout()
    plt.savefig('docs/feature_importances.png')
    print("\\nFeature importance plot saved to docs/feature_importances.png")

if __name__ == '__main__':
    main()
