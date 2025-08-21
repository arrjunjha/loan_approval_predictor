import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import json
import os

def preprocess_data():
    """Load and preprocess the data"""
    # Load data
    df = pd.read_csv('data/raw/data.csv')
    print(f"Original data shape: {df.shape}")
    
    # Drop loan_id if exists
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])
    
    print("Columns in dataset:", df.columns.tolist())
    
    # Create new features (following your exact code)
    df['Movable_assets'] = df[' bank_asset_value'] + df[' luxury_assets_value']
    df['Immovable_assets'] = df[' residential_assets_value'] + df[' commercial_assets_value']
    
    # Drop original asset columns
    df.drop(columns=[' bank_asset_value', ' luxury_assets_value', 
                     ' residential_assets_value', ' commercial_assets_value'], inplace=True)
    
    # Label Encoding (following your exact mappings)
    df[' education'] = df[' education'].map({' Not Graduate': 0, ' Graduate': 1})
    df[' self_employed'] = df[' self_employed'].map({' No': 0, ' Yes': 1})
    df[' loan_status'] = df[' loan_status'].map({' Rejected': 0, ' Approved': 1})
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    print("Data after preprocessing:")
    print(df.info())
    print("\nTarget distribution:")
    print(df['loan_status'].value_counts())
    
    return df

def train_models():
    """Train multiple models with hyperparameter tuning (following your exact approach)"""
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    
    # Process data
    df = preprocess_data()
    
    # Split data (following your exact split)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('loan_status', axis=1), 
        df['loan_status'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Save processed data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Save feature names
    feature_names = list(X_train.columns)
    with open('data/processed/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    print(f"Feature columns: {feature_names}")
    
    # Train basic models first
    print("\n" + "="*50)
    print("TRAINING BASIC MODELS")
    print("="*50)
    
    basic_models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    basic_results = {}
    
    for name, model in basic_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        
        basic_results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'model': model
        }
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Hyperparameter tuning for Random Forest (following your exact params)
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING - RANDOM FOREST")
    print("="*50)
    
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rfc = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(estimator=rfc, param_grid=param_grid_rf, cv=3, 
                          verbose=0, n_jobs=-1, return_train_score=False)
    rf_grid.fit(X_train, y_train)
    
    print(f"Best RF Parameters: {rf_grid.best_params_}")
    
    # Train best RF
    best_rf = RandomForestClassifier(**rf_grid.best_params_, random_state=42)
    best_rf.fit(X_train, y_train)
    rf_pred = best_rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    
    print(f"Tuned RF - Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}")
    
    # Hyperparameter tuning for Decision Tree (following your exact params)
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING - DECISION TREE")
    print("="*50)
    
    param_grid_dt = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    dtree = DecisionTreeClassifier(random_state=42)
    dt_grid = GridSearchCV(dtree, param_grid_dt, cv=5, scoring='accuracy')
    dt_grid.fit(X_train, y_train)
    
    print(f"Best DT Parameters: {dt_grid.best_params_}")
    
    # Train best DT
    best_dt = dt_grid.best_estimator_
    dt_pred = best_dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_f1 = f1_score(y_test, dt_pred)
    
    print(f"Tuned DT - Accuracy: {dt_accuracy:.4f}, F1: {dt_f1:.4f}")
    
    # Compile all results
    all_results = {
        'Basic Models': basic_results,
        'Tuned Models': {
            'Random Forest': {
                'accuracy': rf_accuracy,
                'f1_score': rf_f1,
                'best_params': rf_grid.best_params_,
                'model': best_rf
            },
            'Decision Tree': {
                'accuracy': dt_accuracy,
                'f1_score': dt_f1,
                'best_params': dt_grid.best_params_,
                'model': best_dt
            }
        }
    }
    
    # Find best model overall
    all_accuracies = {}
    all_models = {}
    
    for category, models in all_results.items():
        for model_name, metrics in models.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                all_accuracies[f"{category}_{model_name}"] = metrics['accuracy']
                all_models[f"{category}_{model_name}"] = metrics['model']
    
    best_model_name = max(all_accuracies, key=all_accuracies.get)
    best_model = all_models[best_model_name]
    best_accuracy = all_accuracies[best_model_name]
    
    print(f"\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Best Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    # Save best model
    joblib.dump(best_model, 'loan_model.pkl')
    
    # Save results for Streamlit
    results_for_streamlit = {}
    for category, models in all_results.items():
        for model_name, metrics in models.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                key = f"{category}_{model_name}"
                results_for_streamlit[key] = {
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1_score'],
                    'best_params': metrics.get('best_params', 'N/A')
                }
    
    results_for_streamlit['best_model'] = best_model_name
    results_for_streamlit['best_accuracy'] = best_accuracy
    
    with open('model_results.json', 'w') as f:
        json.dump(results_for_streamlit, f, indent=2)
    
    print("\nAll models trained and results saved!")
    return all_results

if __name__ == "__main__":
    train_models()
