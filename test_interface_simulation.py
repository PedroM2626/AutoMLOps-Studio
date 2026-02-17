
import pandas as pd
import numpy as np
from automl_engine import AutoMLTrainer, AutoMLDataProcessor
import os
import logging

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def callback(trial, score, full_name, dur, metrics=None):
    print(f"Callback -> Trial {trial.number}: {full_name} | Score: {score:.4f} | Duração: {dur:.2f}s")

def main():
    print("🚀 Iniciando Simulação de Teste da Interface AutoMLOps Studio")
    
    # 1. Carregar Dados
    train_path = r"c:\Users\pedro\Downloads\automlops-studio\data_lake\processed_train\v_20260217_175959.csv"
    test_path = r"c:\Users\pedro\Downloads\automlops-studio\data_lake\processed_validation\v_20260217_175938.csv"
    
    if not os.path.exists(train_path):
        print(f"❌ Arquivo de treino não encontrado: {train_path}")
        return

    print(f"📂 Carregando dados de: {train_path}")
    df_train = pd.read_csv(train_path)
    
    # Reduzir tamanho para teste rápido e evitar estouro de memória
    if len(df_train) > 1000:
        print(f"⚠️ Reduzindo dataset de {len(df_train)} para 1000 linhas para teste rápido.")
        df_train = df_train.sample(1000, random_state=42)

    df_test = pd.read_csv(test_path) if os.path.exists(test_path) else None
    if df_test is not None and len(df_test) > 500:
        df_test = df_test.sample(500, random_state=42)
    
    # Assumindo que a última coluna é o target se não especificado, ou 'target' se existir
    target_col = 'target' if 'target' in df_train.columns else df_train.columns[-1]
    print(f"🎯 Target identificado: {target_col}")

    # 2. Processamento de Dados
    print("\n🛠️ Processando dados...")
    
    # Configurar NLP leve para evitar memória alta
    nlp_config = {
        'max_features': 500, # Reduzido drasticamente para teste
        'vectorizer': 'tfidf'
    }
    
    processor = AutoMLDataProcessor(target_column=target_col, task_type='classification', nlp_config=nlp_config)
    X_train, y_train = processor.fit_transform(df_train)
    
    if df_test is not None:
        X_test, y_test = processor.transform(df_test)
    else:
        X_test, y_test = None, None

    # 3. Teste dos Modos de Otimização
    modes = ['random', 'bayesian', 'hyperband', 'grid']
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"🧪 Testando Modo de Otimização: {mode.upper()}")
        print(f"{'='*60}")
        
        trainer = AutoMLTrainer(task_type='classification', preset='fast') # Usando preset fast para teste rápido
        
        try:
            best_model = trainer.train(
                X_train, 
                y_train, 
                n_trials=2, # Poucos trials para teste rápido
                timeout=60, 
                experiment_name=f"Test_{mode}",
                optimization_mode=mode,
                validation_strategy='auto', # Testando validação automática também
                callback=callback,
                random_state=42
            )
            
            print(f"✅ Sucesso no modo {mode}!")
            print(f"🏆 Melhor Modelo: {trainer.best_params.get('model_name')}")
            print(f"📊 Score: {trainer.best_value:.4f}")
            
        except Exception as e:
            print(f"❌ Falha no modo {mode}: {e}")
            import traceback
            traceback.print_exc()

    print("\n🏁 Simulação concluída com sucesso!")

if __name__ == "__main__":
    main()
