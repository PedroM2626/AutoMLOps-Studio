import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
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

def clean_text_god_mode(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    # Mantendo alguns caracteres de pontuação que podem indicar sentimento (ex: !, ?)
    text = re.sub(r'[^a-z\s\!\?]', '', text)
    return " ".join(text.split())

def load_data():
    cols = ['id', 'entity', 'sentiment', 'text']
    df_train = pd.read_csv(os.path.join(DATA_DIR, "twitter_training.csv"), header=None, names=cols).dropna(subset=['text'])
    df_val = pd.read_csv(os.path.join(DATA_DIR, "twitter_validation.csv"), header=None, names=cols).dropna(subset=['text'])
    
    # Limpeza profunda
    df_train = df_train.drop_duplicates(subset=['text'])
    df_train['text_clean'] = df_train['text'].apply(clean_text_god_mode)
    df_val['text_clean'] = df_val['text'].apply(clean_text_god_mode)
    
    return df_train, df_val

def train_god_mode():
    mlflow.set_experiment("Twitter_Sentiment_Classic_ML")
    
    with mlflow.start_run(run_name="God_Mode_Ensemble"):
        df_train, df_val = load_data()
        
        print("--- Vetorizando (Max Features 50k + Bigrams/Trigrams) ---")
        vectorizer = TfidfVectorizer(
            max_features=50000, 
            ngram_range=(1,3),
            sublinear_tf=True,
            strip_accents='unicode'
        )
        
        X_train = vectorizer.fit_transform(df_train['text_clean'])
        X_val = vectorizer.transform(df_val['text_clean'])
        y_train = df_train['sentiment']
        y_val = df_val['sentiment']
        
        # Criando o Ensemble (A união faz a força)
        clf1 = PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5)
        clf2 = LogisticRegression(max_iter=2000, C=10, solver='saga', n_jobs=-1)
        clf3 = SGDClassifier(loss='modified_huber', max_iter=2000, n_jobs=-1) # Huber aceita probabilidades e é robusto a outliers
        
        print("--- Treinando Ensemble (Voting Classifier) ---")
        ensemble = VotingClassifier(
            estimators=[('pa', clf1), ('lr', clf2), ('sgd', clf3)],
            voting='hard', # 'hard' é mais rápido e funciona bem com o PA
            n_jobs=-1
        )
        
        ensemble.fit(X_train, y_train)
        
        print("--- Avaliando ---")
        y_pred = ensemble.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        # Logs
        mlflow.log_param("model_type", "VotingEnsemble")
        mlflow.log_param("estimators", "PA, LR, SGD")
        mlflow.log_param("max_features", 50000)
        mlflow.log_metric("accuracy", acc)
        
        # Matriz de Confusão
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=ensemble.classes_, yticklabels=ensemble.classes_)
        plt.title(f"God Mode Ensemble - Acc: {acc:.4f}")
        plt.savefig("cm_god_mode.png")
        mlflow.log_artifact("cm_god_mode.png")
        
        # Salvar
        joblib.dump(ensemble, "god_mode_model.pkl")
        joblib.dump(vectorizer, "god_mode_vectorizer.pkl")
        mlflow.log_artifact("god_mode_model.pkl")
        mlflow.log_artifact("god_mode_vectorizer.pkl")
        
        print(f"Treino God Mode Concluído! Nova Acurácia: {acc:.4f}")

if __name__ == "__main__":
    train_god_mode()
