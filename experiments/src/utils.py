import os
import mlflow
import dagshub
from dotenv import load_dotenv

def setup_mlflow(experiment_name="mlops_experiment"):
    """
    Configura o MLflow para usar o DagsHub como backend.
    Carrega variáveis de ambiente do arquivo .env.
    """
    load_dotenv()
    
    # Configurações do DagsHub
    dagshub_user = os.getenv('DAGSHUB_USER', 'PedroM2626')
    dagshub_repo = os.getenv('DAGSHUB_REPO', 'experiments')
    dagshub_token = os.getenv('DAGSHUB_TOKEN', '')
    
    try:
        # Inicializar DagsHub
        dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
        
        # Configurar MLflow para usar DagsHub
        tracking_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Configurar autenticação se houver token
        if dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            dagshub.auth.add_app_token(dagshub_token)
            
        print(f"✅ MLflow configurado com DagsHub: {tracking_uri}")
        
    except Exception as e:
        print(f"⚠️  Erro ao configurar DagsHub, usando local: {e}")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(experiment_name)
    print(f"📊 Experimento MLflow: {experiment_name}")
    
    try:
        print(f"🔗 Tracking URI: {mlflow.get_tracking_uri()}")
    except:
        pass
    
    return mlflow

def setup_mlflow_dagshub():
    """
    Alias para setup_mlflow com configuração específica do DagsHub
    """
    return setup_mlflow()