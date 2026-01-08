"""
Comparação completa de modelos usando TODAS as features do dataset
Incluindo tweet_id, entity, e outras colunas (como zeros quando necessário)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import mlflow
from src.utils import setup_mlflow
import warnings
warnings.filterwarnings('ignore')

def load_validation_data_full_features(csv_path):
    """Carrega dados de validação com TODAS as features"""
    print(f"Carregando dados de validação de: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset carregado: {len(df)} amostras")
    print(f"Colunas disponíveis: {list(df.columns)}")
    
    # Verificar se temos a coluna 'sentiment' (target)
    if 'sentiment' not in df.columns:
        print("❌ ERRO: Coluna 'sentiment' não encontrada!")
        return None, None
    
    # Preparar features completas
    X = df.copy()
    y = X.pop('sentiment')
    
    # Remover colunas que não são features (metadata)
    cols_to_remove = ['tweet_id', 'split']  # Manter entity, text, etc.
    for col in cols_to_remove:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    print(f"Features preparadas: {X.shape[1]} colunas")
    print(f"Classes alvo: {y.value_counts().to_dict()}")
    
    return X, y

def load_saved_model_full(model_path):
    """Carrega modelo salvo"""
    print(f"\nCarregando modelo de: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None

def prepare_features_for_model(X, model_name):
    """Prepara features específicas para cada modelo"""
    print(f"\nPreparando features para {model_name}...")
    
    # Para KNN que foi treinado com features completas
    if 'knn' in model_name.lower():
        # Usar todas as colunas disponíveis
        feature_cols = [col for col in X.columns if col not in ['text', 'text_clean', 'text_no_stop', 'text_lemmatized']]
        if not feature_cols:
            # Se não houver outras colunas, criar features dummy
            print("Criando features dummy para KNN...")
            X_prepared = pd.DataFrame({
                'entity_encoded': LabelEncoder().fit_transform(X.get('entity', ['unknown'] * len(X))),
                'text_length': X.get('text', '').str.len(),
                'word_count': X.get('text_clean', '').str.split().str.len()
            })
        else:
            X_prepared = X[feature_cols].copy()
            # Codificar variáveis categóricas
            for col in X_prepared.columns:
                if X_prepared[col].dtype == 'object':
                    le = LabelEncoder()
                    X_prepared[col] = le.fit_transform(X_prepared[col].astype(str))
    
    # Para LinearSVC que foi treinado só com texto
    elif 'linearsvc' in model_name.lower():
        # Usar texto processado
        if 'text_lemmatized' in X.columns:
            X_prepared = X['text_lemmatized'].fillna('')
        elif 'text_clean' in X.columns:
            X_prepared = X['text_clean'].fillna('')
        else:
            X_prepared = X['text'].fillna('') if 'text' in X.columns else [''] * len(X)
    
    else:
        # Padrão: usar texto
        text_col = 'text_lemmatized' if 'text_lemmatized' in X.columns else 'text_clean' if 'text_clean' in X.columns else 'text'
        X_prepared = X[text_col].fillna('') if text_col in X.columns else [''] * len(X)
    
    print(f"Features preparadas: {type(X_prepared)} com shape {len(X_prepared)}")
    return X_prepared

def evaluate_model_complete(model, X_val, y_val, model_name, experiment_name="model_comparison_complete"):
    """Avalia modelo completo com logging MLflow"""
    print(f"\n📊 Avaliando {model_name}...")
    
    try:
        # Preparar features
        X_prepared = prepare_features_for_model(X_val, model_name)
        
        # Fazer predições
        y_pred = model.predict(X_prepared)
        
        # Calcular métricas
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        print(f"✅ {model_name} avaliado com sucesso!")
        print(f"   Acurácia: {accuracy:.4f}")
        print(f"   Precisão: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        # Logar no MLflow
        setup_mlflow(experiment_name)
        with mlflow.start_run(run_name=f"evaluation_{model_name}_complete") as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("features_used", "complete_dataset")
            mlflow.log_param("validation_samples", len(y_val))
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Classification report
            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            mlflow.log_dict(report, "classification_report.json")
            
            # Salvar modelo como artefato
            import pickle
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
                pickle.dump(model, f)
                mlflow.log_artifact(f.name, "model")
                os.unlink(f.name)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'run_id': run.info.run_id
        }
        
    except Exception as e:
        print(f"❌ Erro ao avaliar {model_name}: {e}")
        return None

def main():
    """Função principal de comparação completa"""
    print("🔍 COMPARAÇÃO COMPLETA DE MODELOS COM TODAS AS FEATURES")
    print("="*60)
    
    # Configurações
    model_paths = {
        'knn': 'models/senti-pred-free-mlops-knn.pkl',
        'linearsvc': 'models/senti-pred-linearsvc.pkl'
    }
    validation_path = 'processed_validation.csv'
    
    # Carregar dados de validação
    X_val, y_val = load_validation_data_full_features(validation_path)
    if X_val is None or y_val is None:
        print("❌ Erro ao carregar dados de validação")
        return
    
    # Resultados
    results = []
    
    # Avaliar cada modelo
    for model_type, model_path in model_paths.items():
        print(f"\n{'='*50}")
        print(f"AVALIANDO: {model_type.upper()}")
        print(f"{'='*50}")
        
        # Carregar modelo
        model = load_saved_model_full(model_path)
        if model is None:
            continue
        
        # Avaliar
        result = evaluate_model_complete(
            model, X_val, y_val, 
            model_name=f"senti-pred-{model_type}",
            experiment_name="model_comparison_complete"
        )
        
        if result:
            results.append(result)
    
    # Resumo final
    print(f"\n{'='*60}")
    print("📊 RESUMO DA COMPARAÇÃO COMPLETA")
    print(f"{'='*60}")
    
    if not results:
        print("❌ Nenhum modelo foi avaliado com sucesso")
        return
    
    # Ordenar por F1-score
    results_sorted = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    
    print("\n🏆 RANKING POR F1-SCORE:")
    for i, result in enumerate(results_sorted, 1):
        print(f"{i}. {result['model_name']}")
        print(f"   📈 Acurácia: {result['accuracy']:.4f}")
        print(f"   🎯 Precisão: {result['precision']:.4f}")
        print(f"   🔄 Recall: {result['recall']:.4f}")
        print(f"   ⚖️  F1-Score: {result['f1_score']:.4f}")
        print()
    
    # Melhor modelo
    best_model = results_sorted[0]
    print(f"🥇 MELHOR MODELO: {best_model['model_name']}")
    print(f"   Com F1-Score de {best_model['f1_score']:.4f}")
    print(f"   Run ID: {best_model['run_id']}")

if __name__ == "__main__":
    main()