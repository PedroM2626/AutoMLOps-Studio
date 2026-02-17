import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import mlflow
import dagshub
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Configuração MLOps
load_dotenv()
dagshub.init(repo_owner="PedroM2626", repo_name="experiments")
mlflow.set_tracking_uri("https://dagshub.com/PedroM2626/experiments.mlflow")

DATA_DIR = r"c:\Users\pedro\Downloads\experiments\experiments\Senti-Pred-Remake\data\raw"

def load_data():
    cols = ['id', 'entity', 'sentiment', 'text']
    df_train = pd.read_csv(os.path.join(DATA_DIR, "twitter_training.csv"), header=None, names=cols).dropna(subset=['text'])
    df_val = pd.read_csv(os.path.join(DATA_DIR, "twitter_validation.csv"), header=None, names=cols).dropna(subset=['text'])
    return df_train, df_val

def train_classic():
    mlflow.set_experiment("Twitter_Sentiment_Classic_ML")
    
    with mlflow.start_run(run_name="Logistic_Regression_TFIDF"):
        df_train, df_val = load_data()
        
        print("--- Vetorizando com TF-IDF ---")
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
        X_train = vectorizer.fit_transform(df_train['text'])
        X_val = vectorizer.transform(df_val['text'])
        y_train = df_train['sentiment']
        y_val = df_val['sentiment']
        
        print("--- Treinando Logistic Regression ---")
        model = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
        model.fit(X_train, y_train)
        
        print("--- Avaliando ---")
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        # Logs
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_metric("accuracy", acc)
        
        # Matriz de Confusão
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.savefig("cm_classic.png")
        mlflow.log_artifact("cm_classic.png")
        
        # Salvar Artefatos
        joblib.dump(model, "classic_model.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        mlflow.log_artifact("classic_model.pkl")
        mlflow.log_artifact("tfidf_vectorizer.pkl")
        
        print(f"Treino Clássico Concluído! Acurácia: {acc:.4f}")

if __name__ == "__main__":
    train_classic()