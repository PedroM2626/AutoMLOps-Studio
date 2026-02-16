"""03_modeling.py

Refatorado para focar exclusivamente em Logistic Regression com otimização via Optuna.
Treina o modelo a partir dos dados pré-processados, otimiza hiperparâmetros,
gera métricas e salva o melhor pipeline.
"""

from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import optuna

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

sns.set(style='whitegrid')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
METRICS_DIR = PROJECT_ROOT / 'reports' / 'metrics'
VIS_DIR = PROJECT_ROOT / 'reports' / 'visualizacoes'
MODEL_DIR = PROJECT_ROOT / 'src' / 'models'

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_processed():
    p = PROCESSED_DIR / 'processed_data.pkl'
    if not p.exists():
        raise FileNotFoundError(f'Processed data not found: {p}. Execute 02_preprocessing.py first')
    obj = joblib.load(p)
    return obj['train'], obj.get('validation', pd.DataFrame())

def objective(trial, X_train, y_train, X_val, y_val):
    # Hiperparâmetros para Tfidf
    max_features = trial.suggest_int('tfidf__max_features', 5000, 30000)
    ngram_range = trial.suggest_categorical('tfidf__ngram_range', [(1, 1), (1, 2), (1, 3)])
    
    # Hiperparâmetros para Logistic Regression
    C = trial.suggest_float('lr__C', 1e-3, 100, log=True)
    solver = trial.suggest_categorical('lr__solver', ['lbfgs', 'liblinear', 'saga'])
    
    # Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ('lr', LogisticRegression(C=C, solver=solver, max_iter=2000, random_state=42, n_jobs=-1 if solver != 'liblinear' else 1))
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    return f1_score(y_val, preds, average='macro')

def train_and_evaluate():
    df_train, df_val = load_processed()
    
    # Preparação de dados
    X_train = df_train['text_lemmatized'].astype(str)
    y_train = df_train['sentiment']
    X_val = df_val['text_lemmatized'].astype(str)
    y_val = df_val['sentiment']

    # Limpeza de vazios
    mask_train = X_train.str.strip().replace('', np.nan).notna()
    mask_val = X_val.str.strip().replace('', np.nan).notna()
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_val, y_val = X_val[mask_val], y_val[mask_val]

    print("Iniciando otimização com Optuna para Logistic Regression...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)

    print(f"Melhores hiperparâmetros: {study.best_params}")
    
    # Treinar modelo final com os melhores parâmetros
    best_params = study.best_params
    final_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=best_params['tfidf__max_features'], 
            ngram_range=best_params['tfidf__ngram_range']
        )),
        ('lr', LogisticRegression(
            C=best_params['lr__C'], 
            solver=best_params['lr__solver'], 
            max_iter=3000, 
            random_state=42,
            n_jobs=-1 if best_params['lr__solver'] != 'liblinear' else 1
        ))
    ])

    t0 = time.time()
    final_pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    
    t0 = time.time()
    preds = final_pipeline.predict(X_val)
    predict_time = time.time() - t0
    
    y_score = final_pipeline.predict_proba(X_val)

    # Métricas
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    report = classification_report(y_val, preds, output_dict=True)
    cm = confusion_matrix(y_val, preds)

    # ROC/PR para multiclasse
    classes = np.unique(y_val)
    y_val_b = label_binarize(y_val, classes=classes)
    roc_auc_macro = roc_auc_score(y_val_b, y_score, average='macro', multi_class='ovr')
    avg_precision_macro = average_precision_score(y_val_b, y_score, average='macro')

    # Salvar modelo
    joblib.dump(final_pipeline, MODEL_DIR / 'sentiment_model.pkl')
    print(f'[OK] Melhor modelo salvo em {MODEL_DIR / "sentiment_model.pkl"}')

    # Salvar métricas
    metrics_out = {
        'best_model': 'LogisticRegression_Optimized',
        'best_params': best_params,
        'results': {
            'LogisticRegression': {
                'accuracy': acc,
                'f1_macro': f1,
                'roc_auc_macro': roc_auc_macro,
                'average_precision_macro': avg_precision_macro,
                'train_time_seconds': train_time,
                'predict_time_seconds': predict_time,
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }
        }
    }
    (METRICS_DIR / 'model_metrics.json').write_text(json.dumps(metrics_out, indent=2))
    
    # Plots
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_val_b.ravel(), y_score.ravel())
    plt.plot(fpr, tpr, lw=2, label=f'LogisticRegression (AUC={roc_auc_macro:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Optimized Logistic Regression')
    plt.legend(loc='lower right'); plt.tight_layout(); plt.savefig(VIS_DIR / 'comparison_roc.png'); plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Optimized Logistic Regression')
    plt.xlabel('Predito'); plt.ylabel('Real')
    plt.tight_layout(); plt.savefig(VIS_DIR / 'evaluation_confusion_matrix.png'); plt.close()

    print(f'[RESULT] Final Logistic Regression — acc={acc:.4f} f1_macro={f1:.4f}')

if __name__ == '__main__':
    train_and_evaluate()
