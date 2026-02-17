"""01_eda.py

Script de Análise Exploratória (EDA) para o projeto Senti-Pred.

Gera gráficos PNG em `reports/visualizacoes/` com visualizações úteis antes do
pré-processamento (distribuição de comprimentos, distribuição de sentimentos,
top-words brutas, etc.).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style='whitegrid')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
VIS_DIR = PROJECT_ROOT / 'reports' / 'visualizacoes'
os.makedirs(VIS_DIR, exist_ok=True)

def find_raw_files():
    files = list(RAW_DIR.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'Nenhum arquivo CSV encontrado em {RAW_DIR}')
    # prefer explicit train/validation if present
    train = RAW_DIR / 'twitter_training.csv'
    val = RAW_DIR / 'twitter_validation.csv'
    if train.exists() and val.exists():
        return train, val
    if len(files) >= 2:
        return files[0], files[1]
    return files[0], None

def load_combined():
    train_path, val_path = find_raw_files()
    cols = ['tweet_id', 'entity', 'sentiment', 'text']
    df_train = pd.read_csv(train_path, names=cols, header=None, engine='python', encoding='utf-8')
    if val_path is not None:
        df_val = pd.read_csv(val_path, names=cols, header=None, engine='python', encoding='utf-8')
    else:
        df_val = pd.DataFrame(columns=cols)
    df_train['split'] = 'train'
    df_val['split'] = 'validation'
    df = pd.concat([df_train, df_val], ignore_index=True)
    return df, df_train, df_val

def run_eda():
    df, df_train, df_val = load_combined()
    print(f'Loaded: train={len(df_train)} | validation={len(df_val)} | total={len(df)}')

    # Head and info
    print(df.head())
    print(df.info())

    # text length distribution
    text_col = 'text'
    df['text_length'] = df[text_col].astype(str).apply(lambda s: len(s.split()))
    plt.figure(figsize=(10, 5))
    sns.histplot(df['text_length'], bins=40, kde=True)
    plt.title('Distribuição de comprimento de texto')
    plt.xlabel('Número de palavras')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'text_length.png')
    plt.close()

    # top words raw
    all_words = ' '.join(df[text_col].astype(str)).lower().split()
    top_raw = pd.Series(all_words).value_counts().head(20)
    plt.figure(figsize=(12, 5))
    top_raw.plot(kind='bar')
    plt.title('Top words (raw)')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'top_words_raw.png')
    plt.close()

    # sentiment distribution
    if 'sentiment' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x='sentiment', data=df)
        plt.title('Distribuição de Sentimentos (combined)')
        plt.tight_layout()
        plt.savefig(VIS_DIR / 'sentiment_distribution.png')
        plt.close()

    print('[OK] EDA concluída — gráficos salvos em reports/visualizacoes')

if __name__ == '__main__':
    run_eda()