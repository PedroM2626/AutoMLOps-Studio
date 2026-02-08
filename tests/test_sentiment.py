import pandas as pd
import numpy as np
from automl_engine import AutoMLTrainer, AutoMLDataProcessor
from sklearn.metrics import accuracy_score, confusion_matrix
import time

def run_sentiment_test():
    print("🚀 Iniciando teste automatizado com dataset 'sentiment'...")
    
    # 1. Simular carregamento do dataset sentiment
    # Como não temos o arquivo físico agora, vamos criar um mock que simule o comportamento
    try:
        df = pd.read_csv("data_lake/sentiment/v_20260204_153233.csv")
    except:
        # Fallback para dados sintéticos se o arquivo não existir
        print("⚠️ Arquivo não encontrado, gerando dados sintéticos...")
        data = {
            'text': ['bom', 'ruim', 'excelente', 'pessimo', 'legal', 'chato'] * 20,
            'sentiment': [1, 0, 1, 0, 1, 0] * 20
        }
        df = pd.DataFrame(data)
    
    target = 'sentiment'
    
    print(f"📊 Dataset criado com {len(df)} linhas. Target: {target}")
    
    # 2. Processamento
    print("⚙️ Processando dados...")
    processor = AutoMLDataProcessor(target_column=target, task_type='classification')
    X, y = processor.fit_transform(df)
    
    # 3. Treinamento Automático
    print("🤖 Iniciando AutoML (Modo Automático)...")
    trainer = AutoMLTrainer(task_type='classification')
    
    def test_callback(trial, score, m_name, dur, metrics=None):
        metrics_str = f" | Metrics: {metrics}" if metrics else ""
        print(f"  ✨ Trial {trial.number+1}: {m_name} | Score: {score:.4f} | Tempo: {dur:.2f}s{metrics_str}")

    start_time = time.time()
    # Testando com 2 modelos e 3 trials cada (total 6)
    best_model = trainer.train(
        X, y, 
        n_trials=3, 
        callback=test_callback, 
        selected_models=['logistic_regression', 'random_forest'],
        early_stopping_rounds=5
    )
    
    duration = time.time() - start_time
    print(f"✅ Treinamento concluído em {duration:.2f}s")
    
    # 4. Avaliação
    print("📊 Avaliando melhor modelo...")
    metrics, y_pred = trainer.evaluate(X, y)
    
    print("\n--- Resultados Finais ---")
    for m_name, m_val in metrics.items():
        if m_name != 'confusion_matrix':
            print(f"📈 {m_name.upper()}: {m_val:.4f}")
    
    if 'confusion_matrix' in metrics:
        print("🧩 Matriz de Confusão:")
        print(np.array(metrics['confusion_matrix']))
    
    print("\n--- Verificação de Erros ---")
    if metrics.get('accuracy', 0) == 0:
        print("❌ ERRO: A acurácia está zerada!")
    elif metrics.get('accuracy', 0) > 0.5:
        print("🎉 SUCESSO: Modelo aprendeu corretamente.")
    else:
        print("⚠️ AVISO: Acurácia baixa, mas diferente de zero.")

if __name__ == "__main__":
    run_sentiment_test()
