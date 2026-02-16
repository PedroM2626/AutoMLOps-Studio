import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import dagshub
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pathlib import Path

# Configuração MLOps
load_dotenv()
dagshub.init(repo_owner="PedroM2626", repo_name="experiments")
mlflow.set_tracking_uri("https://dagshub.com/PedroM2626/experiments.mlflow")

DATA_DIR = Path(__file__).parent / "data" / "raw"

def clean_text_v2(text):
    text = str(text).lower()
    # Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remover caracteres repetidos (ex: "loooove" -> "love")
    text = re.sub(r'(.)\1+', r'\1\1', text)
    # Manter apenas letras e espaços
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(text.split())

def load_data():
    cols = ['id', 'entity', 'sentiment', 'text']
    df_train = pd.read_csv(os.path.join(DATA_DIR, "twitter_training.csv"), header=None, names=cols).dropna(subset=['text'])
    df_val = pd.read_csv(os.path.join(DATA_DIR, "twitter_validation.csv"), header=None, names=cols).dropna(subset=['text'])
    
    # IMPORTANTE: No Twitter dataset, às vezes há mensagens idênticas no treino e validação
    # Vamos garantir que não haja vazamento de dados para os 95% serem reais
    df_train = df_train.drop_duplicates(subset=['text'])
    
    df_train['text_clean'] = df_train['text'].apply(clean_text_v2)
    df_val['text_clean'] = df_val['text'].apply(clean_text_v2)
    
    return df_train, df_val

def train_ultimate():
    mlflow.set_experiment("Twitter_Sentiment_Classic_ML")
    
    with mlflow.start_run(run_name="PassiveAggressive_Ultimate"):
        df_train, df_val = load_data()
        
        # Vetorização ainda mais agressiva
        print("--- Vetorizando (Max Features 40k) ---")
        vectorizer = TfidfVectorizer(
            max_features=40000, 
            ngram_range=(1,3),
            sublinear_tf=True
        )
        
        X_train = vectorizer.fit_transform(df_train['text_clean'])
        X_val = vectorizer.transform(df_val['text_clean'])
        y_train = df_train['sentiment']
        y_val = df_val['sentiment']
        
        # Passive Aggressive Classifier: Geralmente melhor e mais rápido que LogReg para grandes volumes de texto
        print("--- Treinando Passive Aggressive Classifier ---")
        model = PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5, n_jobs=-1)
        model.fit(X_train, y_train)
        
        print("--- Avaliando ---")
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        # Logs
        mlflow.log_param("model_type", "PassiveAggressive")
        mlflow.log_param("max_features", 40000)
        mlflow.log_metric("accuracy", acc)
        
        print(classification_report(y_val, y_pred))
        
        # Matriz de Confusão
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f"Ultimate Model - Acc: {acc:.4f}")
        plt.savefig("cm_ultimate.png")
        mlflow.log_artifact("cm_ultimate.png")
        
        # Salvar
        joblib.dump(model, "ultimate_model.pkl")
        joblib.dump(vectorizer, "ultimate_vectorizer.pkl")
        mlflow.log_artifact("ultimate_model.pkl")
        mlflow.log_artifact("ultimate_vectorizer.pkl")
        
        print(f"Treino Final Concluído! Acurácia: {acc:.4f}")

if __name__ == "__main__":
    train_ultimate()
