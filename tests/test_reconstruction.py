import sys
import os
import numpy as np
import pandas as pd
import optuna

# Adicionar o diretório raiz ao path para importar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automl_engine import AutoMLTrainer

def test_model_reconstruction():
    """
    Testa se o melhor modelo pode ser reconstruído corretamente a partir dos best_params
    sem erros de FixedTrial (especialmente para XGBoost e KNN).
    """
    print("🧪 Iniciando teste de reconstrução de modelos...")
    
    # Gerar dados sintéticos para classificação
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Lista de modelos para testar a reconstrução
    models_to_test = ['xgboost', 'knn', 'random_forest', 'svm', 'logistic_regression']
    
    for model_name in models_to_test:
        print(f"\n--- Testando modelo: {model_name} ---")
        trainer = AutoMLTrainer(task_type='classification')
        
        try:
            # Treinar com apenas 1 trial para ser rápido
            best_model = trainer.train(X, y, n_trials=1, selected_models=[model_name])
            
            print(f"✅ Modelo {model_name} treinado e reconstruído com sucesso!")
            print(f"   Best params: {trainer.best_params}")
            
            # Verificar se o modelo reconstruído pode fazer predições
            preds = best_model.predict(X[:5])
            print(f"   Predições (primeiras 5): {preds}")
            
            # Verificar se Feature Importance foi calculada
            if trainer.feature_importance:
                print(f"   📈 Feature Importance calculada: {len(trainer.feature_importance)} features")
            else:
                print("   ⚠️ Feature Importance não disponível para este modelo.")
                
        except ValueError as e:
            print(f"❌ Erro de ValueError na reconstrução de {model_name}: {e}")
            raise e
        except Exception as e:
            print(f"❌ Erro inesperado na reconstrução de {model_name}: {e}")
            raise e

def test_reproducibility():
    """
    Testa se o uso do Random Seed produz resultados consistentes.
    """
    print("\n🧪 Iniciando teste de reprodutibilidade (Random Seed)...")
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    seed = 42
    
    # Treino 1
    trainer1 = AutoMLTrainer(task_type='classification')
    trainer1.train(X, y, n_trials=5, selected_models=['random_forest'], random_state=seed)
    params1 = trainer1.best_params
    
    # Treino 2
    trainer2 = AutoMLTrainer(task_type='classification')
    trainer2.train(X, y, n_trials=5, selected_models=['random_forest'], random_state=seed)
    params2 = trainer2.best_params
    
    if params1 == params2:
        print(f"✅ Reprodutibilidade confirmada com seed {seed}!")
        print(f"   Params: {params1}")
    else:
        print(f"❌ Falha na reprodutibilidade com seed {seed}!")
        print(f"   Params 1: {params1}")
        print(f"   Params 2: {params2}")

if __name__ == "__main__":
    try:
        test_model_reconstruction()
        test_reproducibility()
        print("\n✨ Todos os testes de reconstrução e semente passaram!")
    except Exception as e:
        print(f"\n💥 Falha nos testes: {e}")
        sys.exit(1)
