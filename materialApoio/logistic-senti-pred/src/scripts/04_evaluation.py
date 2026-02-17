"""04_evaluation.py

Avaliação modular: carrega o modelo salvo em `src/models/sentiment_model.pkl` e
os dados pré-processados (pickle em `data/processed/processed_data.pkl`) para
gerar métricas e imagens de avaliação (matriz de confusão, ROC/PR quando
disponível).
"""

from pathlib import Path
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve

sns.set(style='whitegrid')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / 'src' / 'models' / 'sentiment_model.pkl'
PROCESSED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'processed_data.pkl'
METRICS_DIR = PROJECT_ROOT / 'reports' / 'metrics'
VIS_DIR = PROJECT_ROOT / 'reports' / 'visualizacoes'
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

def run_evaluation():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}. Run 03_modeling.py first')
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f'Processed data not found: {PROCESSED_PATH}. Run 02_preprocessing.py first')

    model = joblib.load(MODEL_PATH)
    data = joblib.load(PROCESSED_PATH)
    df = pd.concat([data.get('train', pd.DataFrame()), data.get('validation', pd.DataFrame())], ignore_index=True)

    if 'text_lemmatized' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError('Processed data missing required columns')

    # Split same as modeling (train/validation already available)
    X = df['text_lemmatized'].astype(str)
    y = df['sentiment']

    # For evaluation, use validation set if present
    val = data.get('validation')
    if val is None or val.empty:
        print('[WARN] Validation set empty — using whole dataset for evaluation')
        X_eval = X; y_eval = y
    else:
        X_eval = val['text_lemmatized'].astype(str); y_eval = val['sentiment']

    preds = model.predict(X_eval)
    try:
        probs = model.predict_proba(X_eval)
        has_probs = True
    except Exception:
        probs = None; has_probs = False

    report = classification_report(y_eval, preds, output_dict=True)
    acc = accuracy_score(y_eval, preds)
    cm = confusion_matrix(y_eval, preds)

    # save confusion matrix image
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'evaluation_confusion_matrix.png')
    plt.close()

    # Estrutura padronizada focada em Logistic Regression
    metrics = {
        'best_model': 'LogisticRegression',
        'results': {
            'LogisticRegression': {
                'accuracy': acc,
                'f1_macro': report['macro avg']['f1-score'],
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }
        },
        'model_classes': model.classes_.tolist()
    }

    if has_probs and probs is not None and probs.shape[1] == len(model.classes_):
        if len(model.classes_) == 2:
            # Avaliação Binária (ROC/PR)
            fpr, tpr, _ = roc_curve(y_eval, probs[:, 1], pos_label=model.classes_[1])
            roc_auc = auc(fpr, tpr)
            metrics['roc'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}

            precision, recall, _ = precision_recall_curve(y_eval, probs[:, 1], pos_label=model.classes_[1])
            metrics['pr'] = {'precision': precision.tolist(), 'recall': recall.tolist()}

            # save ROC/PR combined
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc:.3f}')
            plt.plot([0,1],[0,1],'k--', linewidth=0.5)
            plt.title('ROC Curve (Logistic Regression)')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, lw=2)
            plt.title('Precision-Recall (Logistic Regression)')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.tight_layout()
            plt.savefig(VIS_DIR / 'evaluation_roc_pr.png')
            plt.close()
        else:
            # Avaliação Multiclasse (Micro-average ROC se necessário)
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_auc_score

            y_eval_bin = label_binarize(y_eval, classes=model.classes_)
            n_classes = len(model.classes_)

            # Calcula ROC para cada classe
            plt.figure(figsize=(10, 8))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_eval_bin[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {model.classes_[i]} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multiclass ROC - Logistic Regression')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(VIS_DIR / 'evaluation_multiclass_roc.png')
            plt.close()
            
            # Adiciona AUC macro/weighted no json
            metrics['roc_auc_macro'] = roc_auc_score(y_eval_bin, probs, average='macro', multi_class='ovr')
            metrics['roc_auc_weighted'] = roc_auc_score(y_eval_bin, probs, average='weighted', multi_class='ovr')

    # write metrics json
    with (METRICS_DIR / 'model_metrics.json').open('w') as f:
        json.dump(metrics, f, indent=2)
    print(f'[OK] Evaluation metrics and images saved (see {METRICS_DIR} and {VIS_DIR})')

if __name__ == '__main__':
    run_evaluation()