import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Adicionar o diretório pai ao path para importar automl_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automl_engine import AutoMLDataProcessor, AutoMLTrainer

def test_reproducibility():
    print("🧪 Iniciando Teste de Reprodutibilidade...")
    
    # 1. Gerar dados dummy
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df['target'] = y
    
    # 2. Processar dados
    processor = AutoMLDataProcessor(target_column='target', task_type='classification')
    X_proc, y_proc = processor.fit_transform(df)
    
    # 3. Treino 1 com Seed Global
    print("\n--- Treino 1 (Seed: 42) ---")
    trainer1 = AutoMLTrainer(task_type='classification')
    trainer1.train(X_proc, y_proc, n_trials=2, selected_models=['random_forest'], random_state=42)
    best_score1 = trainer1.model_summaries['random_forest']['score']
    params1 = trainer1.model_summaries['random_forest']['params']
    
    # 4. Treino 2 com Mesma Seed Global
    print("\n--- Treino 2 (Seed: 42) ---")
    trainer2 = AutoMLTrainer(task_type='classification')
    trainer2.train(X_proc, y_proc, n_trials=2, selected_models=['random_forest'], random_state=42)
    best_score2 = trainer2.model_summaries['random_forest']['score']
    params2 = trainer2.model_summaries['random_forest']['params']
    
    # 5. Treino 3 com Seed Diferente
    print("\n--- Treino 3 (Seed: 100) ---")
    trainer3 = AutoMLTrainer(task_type='classification')
    trainer3.train(X_proc, y_proc, n_trials=2, selected_models=['random_forest'], random_state=100)
    best_score3 = trainer3.model_summaries['random_forest']['score']
    
    # Verificações
    print("\n📊 Resultados:")
    print(f"Score 1: {best_score1}")
    print(f"Score 2: {best_score2}")
    print(f"Score 3: {best_score3}")
    
    assert best_score1 == best_score2, "ERRO: Scores diferentes para a mesma seed!"
    assert params1 == params2, "ERRO: Parâmetros diferentes para a mesma seed!"
    print("\n✅ SUCESSO: Mesma seed gerou os mesmos resultados.")
    
    if best_score1 != best_score3:
        print("✅ SUCESSO: Seeds diferentes geraram resultados diferentes (como esperado).")
    else:
        print("ℹ️ NOTA: Seeds diferentes geraram o mesmo score (pode ocorrer em datasets pequenos).")

def test_per_model_seeds():
    print("\n🧪 Iniciando Teste de Seeds por Modelo...")
    
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df['target'] = y
    
    processor = AutoMLDataProcessor(target_column='target', task_type='classification')
    X_proc, y_proc = processor.fit_transform(df)
    
    seeds = {'random_forest': 123}
    
    trainer = AutoMLTrainer(task_type='classification')
    trainer.train(X_proc, y_proc, n_trials=2, selected_models=['random_forest'], random_state=seeds)
    
    print("✅ Treino finalizado com seeds por modelo.")
    for model, summary in trainer.model_summaries.items():
        print(f"Modelo: {model} | Melhor Score: {summary['score']}")

if __name__ == "__main__":
    try:
        test_reproducibility()
        test_per_model_seeds()
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        sys.exit(1)
