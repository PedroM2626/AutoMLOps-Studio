import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import mlflow
import dagshub
from dotenv import load_dotenv

# Configuração MLOps
load_dotenv()
REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "PedroM2626")
REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "experiments")

try:
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/PedroM2626/experiments.mlflow"))
except Exception as e:
    print(f"Erro ao inicializar DagsHub/MLflow: {e}")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATA_DIR = r"c:\Users\pedro\Downloads\experiments\experiments\Senti-Pred-Remake\data\raw"

def load_and_preprocess_data():
    train_path = os.path.join(DATA_DIR, "twitter_training.csv")
    val_path = os.path.join(DATA_DIR, "twitter_validation.csv")
    
    cols = ['id', 'entity', 'sentiment', 'text']
    df_train = pd.read_csv(train_path, header=None, names=cols)
    df_val = pd.read_csv(val_path, header=None, names=cols)
    
    # Limpeza básica
    df_train = df_train.dropna(subset=['text']).drop_duplicates()
    df_val = df_val.dropna(subset=['text']).drop_duplicates()
    
    # Mapeamento de labels
    # O dataset tem: Positive, Negative, Neutral, Irrelevant
    # O modelo RoBERTa Twitter tem: 0 -> Negative, 1 -> Neutral, 2 -> Positive
    # Vamos focar nas 3 principais e tratar Irrelevant como Neutral ou remover
    label_map = {
        'Negative': 0,
        'Neutral': 1,
        'Positive': 2,
        'Irrelevant': 1 # Tratando como neutro para simplificar ou poderíamos remover
    }
    
    df_train['label'] = df_train['sentiment'].map(label_map)
    df_val['label'] = df_val['sentiment'].map(label_map)
    
    return df_train[['text', 'label']], df_val[['text', 'label']]

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train():
    mlflow.set_experiment("Twitter_Sentiment_RoBERTa_V2")
    
    with mlflow.start_run():
        print("--- Carregando dados ---")
        df_train, df_val = load_and_preprocess_data()
        
        # df_train = df_train.sample(1000, random_state=42)
        # df_val = df_val.sample(200, random_state=42)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
        train_dataset = Dataset.from_pandas(df_train).map(tokenize_function, batched=True)
        val_dataset = Dataset.from_pandas(df_val).map(tokenize_function, batched=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True)
        
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            report_to="mlflow"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )
        
        print("--- Iniciando Treinamento ---")
        trainer.train()
        
        print("--- Avaliando ---")
        eval_results = trainer.evaluate()
        mlflow.log_metrics(eval_results)
        
        # Matriz de Confusão
        preds = trainer.predict(val_dataset).predictions.argmax(-1)
        cm = confusion_matrix(df_val['label'], preds)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # Salvar modelo
        model.save_pretrained("./best_model")
        tokenizer.save_pretrained("./best_model")
        mlflow.log_artifacts("./best_model", artifact_path="model")
        
        print("Treino concluído com sucesso!")

if __name__ == "__main__":
    train()
