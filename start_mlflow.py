#!/usr/bin/env python3
"""
MLflow initialization script for Free MLOps.
Run this script to start MLflow UI for experiment tracking.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Start MLflow UI for experiment tracking."""
    
    print("🚀 Iniciando MLflow UI para Free MLOps...")
    print("📊 Experimentos serão salvos em: ./mlruns")
    print("🌐 Acesse: http://localhost:5000")
    print("⏹️  Pressione Ctrl+C para parar")
    print("-" * 50)
    
    # Create mlruns directory if it doesn't exist
    mlruns_dir = Path("./mlruns")
    mlruns_dir.mkdir(exist_ok=True)
    
    try:
        # Start MLflow UI
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--backend-store-uri", "sqlite:///mlflow.db"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  MLflow UI parado.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao iniciar MLflow: {e}")
        print("💡 Certifique-se de que o MLflow está instalado: pip install mlflow")
    except FileNotFoundError:
        print("❌ MLflow não encontrado. Instale com: pip install mlflow")

if __name__ == "__main__":
    main()
