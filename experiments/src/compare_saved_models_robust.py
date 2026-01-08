import pandas as pd
import numpy as np
import pickle
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import setup_mlflow

def load_validation_data(file_path):
    """Carrega e prepara os dados de validação."""
    print(f"Carregando dados de validação de: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Dados carregados com sucesso. Shape: {df.shape}")
        print(f"Colunas disponíveis: {df.columns.tolist()}")
        
        # Identificar colunas de texto e sentimento
        text_columns = [col for col in df.columns if 'text' in col.lower()]
        sentiment_columns = [col for col in df.columns if 'sentiment' in col.lower()]
        
        if not text_columns:
            raise ValueError("Nenhuma coluna de texto encontrada nos dados de validação")
        if not sentiment_columns:
            raise ValueError("Nenhuma coluna de sentimento encontrada nos dados de validação")
        
        text_col = text_columns[0]  # Usar primeira coluna de texto encontrada
        sentiment_col = sentiment_columns[0]  # Usar primeira coluna de sentimento encontrada
        
        print(f"Usando coluna de texto: '{text_col}'")
        print(f"Usando coluna de sentimento: '{sentiment_col}'")
        
        # Preparar dados
        X = df[text_col].fillna('')
        y = df[sentiment_col]
        
        print(f"Distribuição de classes: {y.value_counts().to_dict()}")
        
        return X, y, text_col, sentiment_col
    except Exception as e:
        print(f"Erro ao carregar dados de validação: {e}")
        raise

def load_saved_model_safe(model_path):
    """Tenta carregar um modelo salvo usando diferentes métodos."""
    print(f"Tentando carregar modelo de: {model_path}")
    
    # Primeiro, tentar com joblib (mais robusto para sklearn)
    try:
        model = joblib.load(model_path)
        print(f"Modelo carregado com sucesso usando joblib: {type(model)}")
        return model
    except Exception as e1:
        print(f"Falha com joblib: {e1}")
    
    # Depois, tentar com pickle em diferentes modos
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Modelo carregado com sucesso usando pickle: {type(model)}")
        return model
    except Exception as e2:
        print(f"Falha com pickle padrão: {e2}")
    
    # Tentar com encoding diferente
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        print(f"Modelo carregado com sucesso usando pickle (latin1): {type(model)}")
        return model
    except Exception as e3:
        print(f"Falha com pickle latin1: {e3}")
    
    # Tentar com protocol fixo
    try:
        with open(model_path, 'rb') as f:
            # Forçar protocolo mais antigo
            import pickle5
            model = pickle5.load(f)
        print(f"Modelo carregado com sucesso usando pickle5: {type(model)}")
        return model
    except Exception as e4:
        print(f"Falha com pickle5: {e4}")
        print("Todas as tentativas de carregamento falharam.")
        raise

def inspect_model(model, model_name):
    """Inspeciona o modelo para entender sua estrutura."""
    print(f"\nInspecionando modelo: {model_name}")
    print(f"Tipo: {type(model)}")
    
    # Verificar se é um pipeline sklearn
    if hasattr(model, 'steps'):
        print("É um pipeline sklearn com os seguintes passos:")
        for i, (name, step) in enumerate(model.steps):
            print(f"  {i}: {name} -> {type(step)}")
    
    # Verificar se tem transformadores
    if hasattr(model, 'transform'):
        print("Tem método transform")
    
    # Verificar se tem predict
    if hasattr(model, 'predict'):
        print("Tem método predict")
    
    # Verificar se tem predict_proba
    if hasattr(model, 'predict_proba'):
        print("Tem método predict_proba")
    
    # Verificar classes
    if hasattr(model, 'classes_'):
        print(f"Classes: {model.classes_}")

def evaluate_model(model, X_val, y_val, model_name, experiment_name="model_comparison"):
    """Avalia um modelo e retorna métricas."""
    print(f"\nAvaliando modelo: {model_name}")
    
    try:
        # Fazer predições
        if hasattr(model, 'predict'):
            print("Fazendo predições...")
            y_pred = model.predict(X_val)
            print(f"Predições concluídas: {len(y_pred)} amostras")
        else:
            raise ValueError(f"Modelo {model_name} não possui método predict")
        
        # Calcular métricas
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Logar no MLflow
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"evaluation_{model_name}") as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("n_samples", len(y_val))
            
            # Logar classification report como artefato
            try:
                report = classification_report(y_val, y_pred, output_dict=True)
                mlflow.log_dict(report, "classification_report.json")
            except Exception as e:
                print(f"Não foi possível logar classification report: {e}")
            
            # Logar matriz de confusão como artefato
            try:
                cm = confusion_matrix(y_val, y_pred)
                mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")
            except Exception as e:
                print(f"Não foi possível logar confusion matrix: {e}")
            
            # Logar modelo se possível
            try:
                mlflow.sklearn.log_model(model, "model")
                print("Modelo logado no MLflow com sucesso")
            except Exception as e:
                print(f"Não foi possível logar o modelo {model_name} no MLflow: {e}")
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'run_id': run.info.run_id
        }
        
    except Exception as e:
        print(f"Erro ao avaliar modelo {model_name}: {e}")
        raise

def create_ensemble_predictions(models, model_names, X_val):
    """Cria predições ensemble usando votação majoritária."""
    print("\nCriando predições ensemble com votação majoritária...")
    
    try:
        # Obter predições de cada modelo
        predictions = []
        for model, name in zip(models, model_names):
            print(f"Obtendo predições do modelo {name}...")
            y_pred = model.predict(X_val)
            predictions.append(y_pred)
            print(f"Predições do modelo {name}: {len(y_pred)} amostras")
        
        # Votação majoritária
        predictions_array = np.array(predictions)
        ensemble_predictions = []
        
        for i in range(predictions_array.shape[1]):
            votes = predictions_array[:, i]
            # Contar votos para cada classe
            unique, counts = np.unique(votes, return_counts=True)
            # Escolher classe com mais votos
            majority_vote = unique[np.argmax(counts)]
            ensemble_predictions.append(majority_vote)
        
        print(f"Predições ensemble criadas: {len(ensemble_predictions)} amostras")
        return np.array(ensemble_predictions)
        
    except Exception as e:
        print(f"Erro ao criar predições ensemble: {e}")
        raise

def evaluate_ensemble(models, model_names, X_val, y_val, experiment_name="model_comparison"):
    """Avalia o modelo ensemble."""
    print(f"\nAvaliando modelo ensemble...")
    
    try:
        # Criar predições ensemble
        y_pred_ensemble = create_ensemble_predictions(models, model_names, X_val)
        
        # Calcular métricas do ensemble
        accuracy = accuracy_score(y_val, y_pred_ensemble)
        precision = precision_score(y_val, y_pred_ensemble, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred_ensemble, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred_ensemble, average='weighted', zero_division=0)
        
        print(f"Acurácia (Ensemble): {accuracy:.4f}")
        print(f"Precisão (Ensemble): {precision:.4f}")
        print(f"Recall (Ensemble): {recall:.4f}")
        print(f"F1-Score (Ensemble): {f1:.4f}")
        
        # Logar no MLflow
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="evaluation_ensemble_voting") as run:
            mlflow.log_param("model_name", "ensemble_voting")
            mlflow.log_param("models_used", ", ".join(model_names))
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("n_samples", len(y_val))
            
            # Logar classification report
            try:
                report = classification_report(y_val, y_pred_ensemble, output_dict=True)
                mlflow.log_dict(report, "classification_report.json")
            except Exception as e:
                print(f"Não foi possível logar classification report do ensemble: {e}")
            
            # Logar matriz de confusão
            try:
                cm = confusion_matrix(y_val, y_pred_ensemble)
                mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")
            except Exception as e:
                print(f"Não foi possível logar confusion matrix do ensemble: {e}")
        
        return {
            'model_name': 'ensemble_voting',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'run_id': run.info.run_id
        }
        
    except Exception as e:
        print(f"Erro ao avaliar ensemble: {e}")
        raise

def compare_models(model_paths, validation_data_path):
    """Compara múltiplos modelos salvos no mesmo dataset de validação."""
    print("="*60)
    print("INICIANDO COMPARAÇÃO DE MODELOS")
    print("="*60)
    
    # Setup MLflow e DagsHub
    setup_mlflow("model_comparison")
    
    # Carregar dados de validação
    X_val, y_val, text_col, sentiment_col = load_validation_data(validation_data_path)
    
    # Carregar modelos
    models = []
    model_names = []
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.pkl', '')
        try:
            model = load_saved_model_safe(model_path)
            inspect_model(model, model_name)
            models.append(model)
            model_names.append(model_name)
        except Exception as e:
            print(f"Erro ao carregar modelo {model_name}: {e}")
            print("Continuando com os modelos restantes...")
            continue
    
    if not models:
        raise ValueError("Nenhum modelo foi carregado com sucesso!")
    
    # Avaliar cada modelo
    results = []
    for model, model_name in zip(models, model_names):
        try:
            result = evaluate_model(model, X_val, y_val, model_name)
            results.append(result)
        except Exception as e:
            print(f"Erro ao avaliar modelo {model_name}: {e}")
            continue
    
    # Avaliar ensemble (se tivermos mais de um modelo)
    if len(models) > 1:
        try:
            ensemble_result = evaluate_ensemble(models, model_names, X_val, y_val)
            results.append(ensemble_result)
        except Exception as e:
            print(f"Não foi possível avaliar ensemble: {e}")
    
    # Criar resumo da comparação
    print("\n" + "="*60)
    print("RESUMO DA COMPARAÇÃO")
    print("="*60)
    
    if results:
        comparison_df = pd.DataFrame(results)
        print(comparison_df[['model_name', 'accuracy', 'precision', 'recall', 'f1_score']].round(4))
        
        # Identificar melhor modelo
        best_model_idx = comparison_df['f1_score'].idxmax()
        best_model = comparison_df.iloc[best_model_idx]
        
        print(f"\nMelhor modelo: {best_model['model_name']}")
        print(f"F1-Score: {best_model['f1_score']:.4f}")
        print(f"Acurácia: {best_model['accuracy']:.4f}")
        
        # Logar resumo no MLflow
        mlflow.set_experiment("model_comparison")
        with mlflow.start_run(run_name="comparison_summary"):
            mlflow.log_dict(comparison_df.to_dict('records'), "comparison_results.json")
            mlflow.log_param("best_model", best_model['model_name'])
            mlflow.log_metric("best_f1_score", best_model['f1_score'])
            mlflow.log_metric("best_accuracy", best_model['accuracy'])
            
            # Logar resumo textual
            summary_text = f"""
            Comparação de Modelos - Resumo
            ======================================
            Modelos avaliados: {', '.join(model_names)}
            Melhor modelo: {best_model['model_name']}
            Melhor F1-Score: {best_model['f1_score']:.4f}
            Melhor Acurácia: {best_model['accuracy']:.4f}
            
            Resultados detalhados:
            {comparison_df.to_string()}
            """
            mlflow.log_text(summary_text, "comparison_summary.txt")
        
        return comparison_df, best_model
    else:
        print("Nenhum resultado de avaliação foi obtido.")
        return None, None

def main():
    """Função principal para execução da comparação."""
    # Configurar paths
    model_dir = "c:\\Users\\pedro\\Downloads\\experiments\\models"
    validation_path = "c:\\Users\\pedro\\Downloads\\experiments\\processed_validation.csv"
    
    # Modelos a comparar
    model_paths = [
        os.path.join(model_dir, "senti-pred-free-mlops-knn.pkl"),
        os.path.join(model_dir, "senti-pred-linearsvc.pkl")
    ]
    
    # Verificar se arquivos existem
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Erro: Modelo não encontrado: {model_path}")
            return
    
    if not os.path.exists(validation_path):
        print(f"Erro: Arquivo de validação não encontrado: {validation_path}")
        return
    
    # Executar comparação
    try:
        comparison_df, best_model = compare_models(model_paths, validation_path)
        
        if comparison_df is not None and best_model is not None:
            print("\n" + "="*60)
            print("COMPARAÇÃO CONCLUÍDA COM SUCESSO!")
            print("="*60)
            print(f"Resultados salvos no MLflow/DagsHub")
            print(f"Melhor modelo: {best_model['model_name']}")
            print(f"Confira os detalhes no MLflow UI ou DagsHub")
        else:
            print("A comparação foi concluída, mas não houve resultados válidos.")
        
    except Exception as e:
        print(f"Erro durante a comparação: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()