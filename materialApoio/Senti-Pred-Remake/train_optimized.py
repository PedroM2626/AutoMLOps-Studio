import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
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

def clean_text(text):
    # Converter para minúsculo
    text = str(text).lower()
    # Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remover menções (@user) e hashtags (#)
    text = re.sub(r'\@\w+|\#','', text)
    # Remover caracteres especiais e números (mantendo apenas letras e espaços)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remover espaços extras
    text = " ".join(text.split())
    return text

def load_data():
    cols = ['id', 'entity', 'sentiment', 'text']
    df_train = pd.read_csv(os.path.join(DATA_DIR, "twitter_training.csv"), header=None, names=cols).dropna(subset=['text'])
    df_val = pd.read_csv(os.path.join(DATA_DIR, "twitter_validation.csv"), header=None, names=cols).dropna(subset=['text'])
    
    print("--- Limpando textos (Pré-processamento) ---")
    df_train['text_clean'] = df_train['text'].apply(clean_text)
    df_val['text_clean'] = df_val['text'].apply(clean_text)
    
    return df_train, df_val

def train_optimized():
    mlflow.set_experiment("Twitter_Sentiment_Classic_ML")
    
    with mlflow.start_run(run_name="Logistic_Regression_Optimized"):
        df_train, df_val = load_data()
        
        # Aumentamos max_features e usamos n-gramas de até 3 palavras
        print("--- Vetorizando com TF-IDF Otimizado ---")
        vectorizer = TfidfVectorizer(
            max_features=20000, 
            ngram_range=(1,3), 
            stop_words='english',
            sublinear_tf=True # Aplica escala logarítmica (ajuda em textos com termos muito repetidos)
        )
        
        X_train = vectorizer.fit_transform(df_train['text_clean'])
        X_val = vectorizer.transform(df_val['text_clean'])
        y_train = df_train['sentiment']
        y_val = df_val['sentiment']
        
        # Ajuste do parâmetro C (Regularização) - Menor C = maior regularização
        # C=10 costuma dar um boost em datasets maiores de texto
        print("--- Treinando Logistic Regression Otimizada ---")
        model = LogisticRegression(max_iter=2000, C=10, n_jobs=-1, solver='saga')
        model.fit(X_train, y_train)
        
        print("--- Avaliando ---")
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        # Logs de hiperparâmetros
        mlflow.log_param("model_type", "LogisticRegression_Optimized")
        mlflow.log_param("C", 10)
        mlflow.log_param("ngram_range", "(1,3)")
        mlflow.log_param("max_features", 20000)
        mlflow.log_param("clean_text", True)
        mlflow.log_metric("accuracy", acc)
        
        # Matriz de Confusão
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f"Confusion Matrix - Acc: {acc:.4f}")
        plt.savefig("cm_optimized.png")
        mlflow.log_artifact("cm_optimized.png")
        
        # Salvar Artefatos
        joblib.dump(model, "optimized_model.pkl")
        joblib.dump(vectorizer, "optimized_vectorizer.pkl")
        mlflow.log_artifact("optimized_model.pkl")
        mlflow.log_artifact("optimized_vectorizer.pkl")
        
        print(f"Treino Otimizado Concluído! Nova Acurácia: {acc:.4f}")

if __name__ == "__main__":
    train_optimized()
