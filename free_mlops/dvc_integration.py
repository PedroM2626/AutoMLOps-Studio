from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from free_mlops.config import get_settings


class DVCIntegration:
    """Integração com DVC para versionamento de datasets e pipelines."""
    
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.project_root = self.settings.artifacts_dir.parent
        self.dvc_dir = self.project_root / ".dvc"
        self.data_dir = self.project_root / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Metadados de datasets
        self.metadata_file = self.settings.artifacts_dir / "dvc_metadata.json"
        
    def initialize_dvc(self) -> Dict[str, Any]:
        """Inicializa repositório DVC no projeto."""
        try:
            # Verificar se DVC está instalado
            subprocess.run(["dvc", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("DVC não está instalado. Instale com: pip install dvc")
        
        result = {"initialized": False, "messages": []}
        
        try:
            # Inicializar DVC se não existir
            if not self.dvc_dir.exists():
                subprocess.run(["dvc", "init"], cwd=self.project_root, check=True, capture_output=True)
                result["messages"].append("DVC inicializado com sucesso")
                result["initialized"] = True
            else:
                result["messages"].append("DVC já está inicializado")
            
            # Configurar remote local para armazenamento
            remote_path = self.project_root / "dvc_storage"
            remote_path.mkdir(exist_ok=True)
            
            try:
                subprocess.run([
                    "dvc", "remote", "add", "-d", "local_storage", str(remote_path)
                ], cwd=self.project_root, check=True, capture_output=True)
                result["messages"].append("Remote storage configurado")
            except subprocess.CalledProcessError:
                # Remote pode já existir
                result["messages"].append("Remote storage já configurado")
            
            # Criar estrutura de diretórios
            (self.data_dir / "raw").mkdir(exist_ok=True)
            (self.data_dir / "processed").mkdir(exist_ok=True)
            (self.data_dir / "models").mkdir(exist_ok=True)
            
            result["messages"].append("Estrutura de diretórios criada")
            
        except subprocess.CalledProcessError as e:
            result["messages"].append(f"Erro: {e}")
            result["error"] = str(e)
        
        return result
    
    def add_dataset(
        self,
        dataset_path: Union[str, Path],
        dataset_name: str,
        dataset_type: str = "raw",
        description: str = "",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Adiciona dataset ao controle DVC."""
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")
        
        # Copiar para diretório DVC
        target_dir = self.data_dir / dataset_type
        target_dir.mkdir(exist_ok=True)
        
        target_path = target_dir / dataset_name
        shutil.copy2(dataset_path, target_path)
        
        try:
            # Adicionar ao DVC
            subprocess.run([
                "dvc", "add", str(target_path)
            ], cwd=self.project_root, check=True, capture_output=True)
            
            # Adicionar ao Git
            subprocess.run([
                "git", "add", f"{target_path}.dvc", ".gitignore"
            ], cwd=self.project_root, check=True, capture_output=True)
            
            # Salvar metadados
            dataset_info = {
                "name": dataset_name,
                "type": dataset_type,
                "description": description,
                "tags": tags or [],
                "metadata": metadata or {},
                "file_path": str(target_path),
                "file_size": target_path.stat().st_size,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dvc_tracked": True,
            }
            
            self._save_dataset_metadata(dataset_info)
            
            return {
                "success": True,
                "dataset_info": dataset_info,
                "message": f"Dataset {dataset_name} adicionado ao DVC"
            }
            
        except subprocess.CalledProcessError as e:
            # Limpar arquivo copiado se falhou
            if target_path.exists():
                target_path.unlink()
            raise RuntimeError(f"Erro ao adicionar dataset ao DVC: {e}")
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Retorna informações de um dataset."""
        metadata = self._load_metadata()
        return metadata.get("datasets", {}).get(dataset_name)
    
    def list_datasets(
        self,
        dataset_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Lista datasets com filtros opcionais."""
        metadata = self._load_metadata()
        datasets = list(metadata.get("datasets", {}).values())
        
        # Filtrar por tipo
        if dataset_type:
            datasets = [d for d in datasets if d.get("type") == dataset_type]
        
        # Filtrar por tags
        if tags:
            datasets = [d for d in datasets if any(tag in d.get("tags", []) for tag in tags)]
        
        # Ordenar por data de criação (mais recentes primeiro)
        datasets.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return datasets
    
    def get_dataset_versions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Retorna histórico de versões de um dataset."""
        try:
            # Usar DVC para obter histórico
            result = subprocess.run([
                "dvc", "log", str(self.data_dir / "raw" / dataset_name)
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                return []
            
            # Processar output (simplificado)
            versions = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    # Parse da linha do log do DVC
                    parts = line.split()
                    if len(parts) >= 2:
                        versions.append({
                            "hash": parts[0],
                            "date": parts[1] if len(parts) > 1 else "",
                            "message": " ".join(parts[2:]) if len(parts) > 2 else "",
                        })
            
            return versions
            
        except subprocess.CalledProcessError:
            return []
    
    def checkout_dataset_version(
        self,
        dataset_name: str,
        version_hash: str,
        target_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Faz checkout de uma versão específica do dataset."""
        
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset {dataset_name} não encontrado")
        
        source_path = Path(dataset_info["file_path"])
        
        if target_path is None:
            target_path = source_path.parent / f"{dataset_name}_{version_hash[:8]}"
        
        try:
            # Fazer checkout da versão
            subprocess.run([
                "dvc", "checkout", version_hash, str(source_path)
            ], cwd=self.project_root, check=True, capture_output=True)
            
            # Copiar para target
            if target_path != source_path:
                shutil.copy2(source_path, target_path)
            
            return {
                "success": True,
                "version_hash": version_hash,
                "target_path": str(target_path),
                "message": f"Versão {version_hash[:8]} do dataset {dataset_name} restaurada"
            }
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Erro ao fazer checkout: {e}")
    
    def create_pipeline(
        self,
        pipeline_name: str,
        stages: List[Dict[str, Any]],
        description: str = "",
    ) -> Dict[str, Any]:
        """Cria um pipeline DVC."""
        
        pipeline_file = self.project_root / "dvc.yaml"
        
        # Ler pipeline existente se houver
        pipeline_data = {}
        if pipeline_file.exists():
            try:
                with open(pipeline_file, 'r') as f:
                    import yaml
                    pipeline_data = yaml.safe_load(f) or {}
            except Exception:
                pipeline_data = {}
        
        # Adicionar novo estágio
        pipeline_data["stages"] = pipeline_data.get("stages", {})
        
        for stage in stages:
            stage_name = stage["name"]
            pipeline_data["stages"][stage_name] = {
                "cmd": stage["cmd"],
                "deps": stage.get("deps", []),
                "outs": stage.get("outs", []),
                "params": stage.get("params", []),
                "metrics": stage.get("metrics", []),
            }
        
        # Salvar pipeline
        try:
            import yaml
            with open(pipeline_file, 'w') as f:
                yaml.dump(pipeline_data, f, default_flow_style=False)
            
            # Salvar metadados do pipeline
            pipeline_info = {
                "name": pipeline_name,
                "description": description,
                "stages": stages,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "file_path": str(pipeline_file),
            }
            
            metadata = self._load_metadata()
            if "pipelines" not in metadata:
                metadata["pipelines"] = {}
            metadata["pipelines"][pipeline_name] = pipeline_info
            self._save_metadata(metadata)
            
            return {
                "success": True,
                "pipeline_info": pipeline_info,
                "message": f"Pipeline {pipeline_name} criado"
            }
            
        except Exception as e:
            raise RuntimeError(f"Erro ao criar pipeline: {e}")
    
    def run_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        """Executa um pipeline DVC."""
        try:
            result = subprocess.run([
                "dvc", "repro"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "pipeline_name": pipeline_name,
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "pipeline_name": pipeline_name,
            }
    
    def get_data_lineage(self, dataset_name: str) -> Dict[str, Any]:
        """Retorna linhagem de dados para um dataset."""
        try:
            # Usar DVC para obter gráfico de dependências
            result = subprocess.run([
                "dvc", "dag", "--dot"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"error": "Não foi possível gerar grafo de dependências"}
            
            # Processar output DOT (simplificado)
            lineage = {
                "dataset": dataset_name,
                "dependencies": [],
                "downstream": [],
                "graph": result.stdout,
            }
            
            # Análise simples do grafo
            lines = result.stdout.split('\n')
            for line in lines:
                if '->' in line and dataset_name in line:
                    parts = line.split('->')
                    if len(parts) == 2:
                        source = parts[0].strip().replace('"', '')
                        target = parts[1].strip().replace('"', '')
                        
                        if source == dataset_name:
                            lineage["downstream"].append(target)
                        elif target == dataset_name:
                            lineage["dependencies"].append(source)
            
            return lineage
            
        except subprocess.CalledProcessError:
            return {"error": "Erro ao obter linhagem de dados"}
    
    def _save_dataset_metadata(self, dataset_info: Dict[str, Any]) -> None:
        """Salva metadados de um dataset."""
        metadata = self._load_metadata()
        if "datasets" not in metadata:
            metadata["datasets"] = {}
        
        metadata["datasets"][dataset_info["name"]] = dataset_info
        self._save_metadata(metadata)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Carrega metadados DVC."""
        if self.metadata_file.exists():
            try:
                return json.loads(self.metadata_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"datasets": {}, "pipelines": {}}
    
    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Salva metadados DVC."""
        self.metadata_file.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def get_dvc_status(self) -> Dict[str, Any]:
        """Retorna status atual do DVC."""
        try:
            # Verificar se DVC está inicializado
            if not self.dvc_dir.exists():
                return {"initialized": False, "message": "DVC não inicializado"}
            
            # Obter status
            result = subprocess.run([
                "dvc", "status"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            return {
                "initialized": True,
                "status_output": result.stdout,
                "status_error": result.stderr,
                "return_code": result.returncode,
            }
            
        except subprocess.CalledProcessError as e:
            return {"initialized": True, "error": str(e)}
        except FileNotFoundError:
            return {"initialized": False, "error": "DVC não encontrado"}


def get_dvc_integration() -> DVCIntegration:
    """Factory function para DVCIntegration."""
    return DVCIntegration()
