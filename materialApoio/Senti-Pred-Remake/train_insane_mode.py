import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
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

def clean_text_insane(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    # Manter letras, espaços e pontuação emocional importante
    text = re.sub(r'[^a-z\s\!\?]', '', text)
    return " ".join(text.split())

def load_data():
    cols = ['id', 'entity', 'sentiment', 'text']
    df_train = pd.read_csv(os.path.join(DATA_DIR, "twitter_training.csv"), header=None, names=cols).dropna(subset=['text'])
    df_val = pd.read_csv(os.path.join(DATA_DIR, "twitter_validation.csv"), header=None, names=cols).dropna(subset=['text'])
    
    # Limpeza profunda e remoção de duplicatas exatas
    df_train = df_train.drop_duplicates(subset=['text'])
    df_train['text_clean'] = df_train['text'].apply(clean_text_insane)
    df_val['text_clean'] = df_val['text'].apply(clean_text_insane)
    
    return df_train, df_val

def train_insane_mode():
    mlflow.set_experiment("Twitter_Sentiment_Classic_ML")
    
    with mlflow.start_run(run_name="Insane_Mode_Stacking"):
        df_train, df_val = load_data()
        
        print("--- Vetorizando (60k Features + Chi2 Selection) ---")
        vectorizer = TfidfVectorizer(
            max_features=60000, 
            ngram_range=(1,3),
            sublinear_tf=True
        )
        
        X_train_raw = vectorizer.fit_transform(df_train['text_clean'])
        X_val_raw = vectorizer.transform(df_val['text_clean'])
        y_train = df_train['sentiment']
        y_val = df_val['sentiment']
        
        # Seleção de Atributos: Reduzir de 60k para as 40k mais importantes
        # Isso remove o "ruído" estatístico que pode confundir o modelo
        print("--- Selecionando os melhores atributos (Chi2) ---")
        selector = SelectKBest(chi2, k=min(40000, X_train_raw.shape[1]))
        X_train = selector.fit_transform(X_train_raw, y_train)
        X_val = selector.transform(X_val_raw)
        
        # Normalização (ajuda o Meta-Learner do Stacking)
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Definindo os Especialistas (Base Models)
        base_models = [
            ('pa', PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5)),
            ('nb', ComplementNB(alpha=0.1)), # ComplementNB é excelente para texto desbalanceado
            ('lr', LogisticRegression(max_iter=2000, C=10, solver='saga', n_jobs=-1))
        ]
        
        # O Meta-Learner: Um modelo que aprende a combinar os especialistas
        meta_learner = LogisticRegression()
        
        print("--- Treinando Stacking Classifier (Insane Mode) ---")
        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=3, # Cross-validation interna para treinar o meta-learner
            n_jobs=-1,
            passthrough=False # Se True, o meta-learner vê também os dados originais (mais lento)
        )
        
        stacking.fit(X_train, y_train)
        
        print("--- Avaliando ---")
        y_pred = stacking.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        # Logs
        mlflow.log_param("model_type", "Stacking_Insane")
        mlflow.log_param("base_models", "PA, CNB, LR")
        mlflow.log_param("meta_learner", "LR")
        mlflow.log_param("k_best", 40000)
        mlflow.log_metric("accuracy", acc)
        
        # Matriz de Confusão
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=stacking.classes_, yticklabels=stacking.classes_)
        plt.title(f"Insane Mode Stacking - Acc: {acc:.4f}")
        plt.savefig("cm_insane.png")
        mlflow.log_artifact("cm_insane.png")
        
        # Salvar Pipeline Completo (Vetorizador + Seletor + Scaler + Modelo)
        # Para facilitar a inferência depois
        pipeline = {
            'vectorizer': vectorizer,
            'selector': selector,
            'scaler': scaler,
            'model': stacking
        }
        joblib.dump(pipeline, "insane_model_bundle.pkl")
        mlflow.log_artifact("insane_model_bundle.pkl")
        
        print(f"Treino Insane Mode Concluído! Nova Acurácia: {acc:.4f}")

if __name__ == "__main__":
    train_insane_mode()
