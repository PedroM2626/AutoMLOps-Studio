from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from free_mlops.config import get_settings


class DataValidator:
    """Validador de dados usando Pandera para schema validation."""
    
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.schemas_dir = self.settings.artifacts_dir / "schemas"
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadados de validação
        self.validation_metadata_file = self.settings.artifacts_dir / "validation_metadata.json"
        
    def create_schema_from_dataframe(
        self,
        df: pd.DataFrame,
        schema_name: str,
        description: str = "",
        strict: bool = True,
        coerce: bool = True,
    ) -> Dict[str, Any]:
        """Cria schema Pandera a partir de um DataFrame."""
        
        try:
            import pandera.pandas as pa
        except ImportError:
            raise ImportError("Pandera não está instalado. Instale com: pip install pandera")
        
        # Inferir schema do DataFrame
        schema_info = {
            "name": schema_name,
            "description": description,
            "strict": strict,
            "coerce": coerce,
            "columns": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Analisar cada coluna
        for col_name, col_data in df.items():
            col_info = self._analyze_column(col_name, col_data)
            schema_info["columns"][col_name] = col_info
        
        # Criar schema Pandera
        schema_columns = {}
        for col_name, col_info in schema_info["columns"].items():
            pandas_dtype = self._get_pandera_dtype(col_info["pandas_dtype"])
            
            # Criar coluna com validações
            column_args = {
                "dtype": pandas_dtype,
                "nullable": col_info.get("nullable", True),
                "coerce": coerce,
                "required": not col_info.get("nullable", True),
            }
            
            # Adicionar validações específicas
            checks = []
            
            if col_info.get("unique_values", False):
                # Para verificar unicidade, usamos uma função customizada
                def check_unique(series):
                    return len(series) == len(series.unique())
                checks.append(pa.Check(check_unique, error="Values must be unique"))
            
            if "min_value" in col_info:
                checks.append(pa.Check.ge(col_info["min_value"]))
            
            if "max_value" in col_info:
                checks.append(pa.Check.le(col_info["max_value"]))
            
            if "min_length" in col_info:
                checks.append(pa.Check.str_length(min_value=col_info["min_length"]))
            
            if "max_length" in col_info:
                checks.append(pa.Check.str_length(max_value=col_info["max_length"]))
            
            if col_info.get("categories"):
                checks.append(pa.Check.isin(col_info["categories"]))
            
            if checks:
                column_args["checks"] = checks
            
            schema_columns[col_name] = pa.Column(**column_args)
        
        # Criar schema
        schema = pa.DataFrameSchema(
            columns=schema_columns,
            strict=strict,
            coerce=coerce,
            name=schema_name,
            description=description,
        )
        
        # Salvar schema
        schema_file = self.schemas_dir / f"{schema_name}.json"
        
        # Preparar dados para salvar (sem o objeto schema)
        save_data = {
            "metadata": schema_info,
            "columns": {}
        }
        
        for col_name, col_schema in schema.columns.items():
            col_data = {
                "dtype": str(col_schema.dtype),
                "nullable": bool(col_schema.nullable),
                "coerce": bool(col_schema.coerce),
                "required": bool(col_schema.required),
                "checks": []
            }
            
            # Adicionar informações dos checks
            for check in col_schema.checks:
                check_info = {
                    "name": check.name if hasattr(check, 'name') else str(check),
                    "error": check.error if hasattr(check, 'error') else str(check)
                }
                col_data["checks"].append(check_info)
            
            save_data["columns"][col_name] = col_data
        
        schema_file.write_text(
            json.dumps(save_data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )
        
        return {
            "schema_name": schema_name,
            "schema_file": str(schema_file),
            "schema": schema,
            "schema_info": schema_info,
            "validation_results": self._validate_dataframe(df, schema),
        }
    
    def _analyze_column(self, col_name: str, col_data: pd.Series) -> Dict[str, Any]:
        """Analisa uma coluna para inferir propriedades."""
        
        col_info = {
            "pandas_dtype": str(col_data.dtype),
            "nullable": col_data.isna().any(),
            "unique_count": col_data.nunique(),
            "total_count": len(col_data),
        }
        
        # Análise por tipo
        if pd.api.types.is_numeric_dtype(col_data):
            col_info.update(self._analyze_numeric_column(col_data))
        elif pd.api.types.is_string_dtype(col_data) or col_data.dtype == 'object':
            col_info.update(self._analyze_string_column(col_data))
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            col_info.update(self._analyze_datetime_column(col_data))
        elif pd.api.types.is_bool_dtype(col_data):
            col_info.update(self._analyze_boolean_column(col_data))
        
        return col_info
    
    def _analyze_numeric_column(self, col_data: pd.Series) -> Dict[str, Any]:
        """Analisa coluna numérica."""
        
        clean_data = col_data.dropna()
        
        info = {
            "min_value": float(clean_data.min()) if len(clean_data) > 0 else None,
            "max_value": float(clean_data.max()) if len(clean_data) > 0 else None,
            "mean": float(clean_data.mean()) if len(clean_data) > 0 else None,
            "std": float(clean_data.std()) if len(clean_data) > 0 else None,
        }
        
        # Verificar se é inteiro
        if clean_data.dtype in ['int64', 'int32'] or (
            len(clean_data) > 0 and (clean_data == clean_data.astype(int)).all()
        ):
            info["is_integer"] = True
            info["unique_values"] = len(clean_data.unique()) < len(clean_data) * 0.5
        
        return info
    
    def _analyze_string_column(self, col_data: pd.Series) -> Dict[str, Any]:
        """Analisa coluna de texto."""
        
        clean_data = col_data.dropna().astype(str)
        
        info = {
            "min_length": int(clean_data.str.len().min()) if len(clean_data) > 0 else None,
            "max_length": int(clean_data.str.len().max()) if len(clean_data) > 0 else None,
            "mean_length": float(clean_data.str.len().mean()) if len(clean_data) > 0 else None,
        }
        
        # Verificar categorias (se poucos valores únicos)
        unique_ratio = clean_data.nunique() / len(clean_data)
        if unique_ratio < 0.1:  # Menos de 10% de valores únicos
            info["categories"] = list(clean_data.unique())
            info["unique_values"] = True
        
        return info
    
    def _analyze_datetime_column(self, col_data: pd.Series) -> Dict[str, Any]:
        """Analisa coluna de datetime."""
        
        clean_data = col_data.dropna()
        
        info = {
            "min_date": str(clean_data.min()) if len(clean_data) > 0 else None,
            "max_date": str(clean_data.max()) if len(clean_data) > 0 else None,
        }
        
        return info
    
    def _analyze_boolean_column(self, col_data: pd.Series) -> Dict[str, Any]:
        """Analisa coluna booleana."""
        
        clean_data = col_data.dropna()
        
        info = {
            "true_count": int(clean_data.sum()) if len(clean_data) > 0 else 0,
            "false_count": int((~clean_data).sum()) if len(clean_data) > 0 else 0,
        }
        
        return info
    
    def _get_pandera_dtype(self, pandas_dtype: str) -> Any:
        """Converte dtype do pandas para dtype do Pandera."""
        
        try:
            import pandera as pa
        except ImportError:
            raise ImportError("Pandera não está instalado")
        
        dtype_mapping = {
            'int64': pa.Int64,
            'int32': pa.Int32,
            'float64': pa.Float64,
            'float32': pa.Float32,
            'bool': pa.Bool,
            'object': pa.String,
            'string': pa.String,
            'datetime64[ns]': pa.DateTime,
        }
        
        return dtype_mapping.get(pandas_dtype, pa.String)
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        schema_name: str,
        lazy: bool = False,
    ) -> Dict[str, Any]:
        """Valida DataFrame contra schema existente."""
        
        try:
            import pandera as pa
        except ImportError:
            raise ImportError("Pandera não está instalado")
        
        # Carregar schema
        schema_file = self.schemas_dir / f"{schema_name}.json"
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema {schema_name} não encontrado")
        
        schema_dict = json.loads(schema_file.read_text(encoding="utf-8"))
        schema = pa.DataFrameSchema(**schema_dict)
        
        # Validar
        validation_result = self._validate_dataframe(df, schema, lazy)
        
        # Salvar resultado
        self._save_validation_result(schema_name, validation_result)
        
        return validation_result
    
    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        schema,
        lazy: bool = False,
    ) -> Dict[str, Any]:
        """Executa validação do DataFrame."""
        
        try:
            import pandera as pa
        except ImportError:
            raise ImportError("Pandera não está instalado")
        
        try:
            validated_df = schema.validate(df, lazy=lazy)
            
            return {
                "success": True,
                "validated_shape": validated_df.shape,
                "original_shape": df.shape,
                "validation_time": datetime.now(timezone.utc).isoformat(),
                "errors": [],
                "warnings": [],
            }
            
        except pa.errors.SchemaErrors as e:
            errors = []
            for error in e.failure_cases:
                errors.append({
                    "column": error.get("column", "unknown"),
                    "check": error.get("check", "unknown"),
                    "failure_case": str(error.get("failure_case", ""))[:100],
                    "index": str(error.get("index", "")),
                })
            
            return {
                "success": False,
                "validated_shape": None,
                "original_shape": df.shape,
                "validation_time": datetime.now(timezone.utc).isoformat(),
                "errors": errors,
                "warnings": [],
            }
        
        except Exception as e:
            return {
                "success": False,
                "validated_shape": None,
                "original_shape": df.shape,
                "validation_time": datetime.now(timezone.utc).isoformat(),
                "errors": [{"error": str(e)}],
                "warnings": [],
            }
    
    def list_schemas(self) -> List[Dict[str, Any]]:
        """Lista todos os schemas disponíveis."""
        
        schemas = []
        
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                schema_dict = json.loads(schema_file.read_text(encoding="utf-8"))
                metadata = schema_dict.get("metadata", {})
                
                schemas.append({
                    "name": metadata.get("name", schema_file.stem),
                    "description": metadata.get("description", ""),
                    "created_at": metadata.get("created_at", ""),
                    "columns_count": len(metadata.get("columns", {})),
                    "strict": metadata.get("strict", False),
                    "file_path": str(schema_file),
                })
            except Exception:
                continue
        
        return sorted(schemas, key=lambda x: x["created_at"], reverse=True)
    
    def get_schema_details(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Retorna detalhes de um schema."""
        
        schema_file = self.schemas_dir / f"{schema_name}.json"
        if not schema_file.exists():
            return None
        
        try:
            schema_dict = json.loads(schema_file.read_text(encoding="utf-8"))
            return schema_dict
        except Exception:
            return None
    
    def compare_schemas(self, schema1_name: str, schema2_name: str) -> Dict[str, Any]:
        """Compara dois schemas."""
        
        schema1 = self.get_schema_details(schema1_name)
        schema2 = self.get_schema_details(schema2_name)
        
        if not schema1 or not schema2:
            return {"error": "Schema(s) não encontrado(s)"}
        
        comparison = {
            "schema1": schema1_name,
            "schema2": schema2_name,
            "columns_added": [],
            "columns_removed": [],
            "columns_modified": [],
            "identical": True,
        }
        
        cols1 = schema1.get("columns", {})
        cols2 = schema2.get("columns", {})
        
        # Colunas adicionadas
        for col in cols2:
            if col not in cols1:
                comparison["columns_added"].append(col)
                comparison["identical"] = False
        
        # Colunas removidas
        for col in cols1:
            if col not in cols2:
                comparison["columns_removed"].append(col)
                comparison["identical"] = False
        
        # Colunas modificadas
        for col in cols1:
            if col in cols2:
                if cols1[col] != cols2[col]:
                    comparison["columns_modified"].append({
                        "column": col,
                        "old": cols1[col],
                        "new": cols2[col],
                    })
                    comparison["identical"] = False
        
        return comparison
    
    def get_validation_history(self, schema_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna histórico de validações para um schema."""
        
        metadata = self._load_validation_metadata()
        history = metadata.get("validation_history", [])
        
        # Filtrar por schema
        schema_history = [
            result for result in history
            if result.get("schema_name") == schema_name
        ]
        
        # Ordenar por data (mais recentes primeiro)
        schema_history.sort(key=lambda x: x.get("validation_time", ""), reverse=True)
        
        return schema_history[:limit]
    
    def _save_validation_result(self, schema_name: str, result: Dict[str, Any]) -> None:
        """Salva resultado de validação."""
        
        metadata = self._load_validation_metadata()
        
        if "validation_history" not in metadata:
            metadata["validation_history"] = []
        
        result["schema_name"] = schema_name
        metadata["validation_history"].append(result)
        
        # Manter apenas últimos 1000 resultados
        if len(metadata["validation_history"]) > 1000:
            metadata["validation_history"] = metadata["validation_history"][-1000:]
        
        self._save_validation_metadata(metadata)
    
    def _load_validation_metadata(self) -> Dict[str, Any]:
        """Carrega metadados de validação."""
        
        if self.validation_metadata_file.exists():
            try:
                return json.loads(self.validation_metadata_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        
        return {"validation_history": []}
    
    def _save_validation_metadata(self, metadata: Dict[str, Any]) -> None:
        """Salva metadados de validação."""
        
        self.validation_metadata_file.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )
    
    def export_schema(self, schema_name: str, format: str = "json") -> str:
        """Exporta schema em diferentes formatos."""
        
        schema_details = self.get_schema_details(schema_name)
        if not schema_details:
            raise ValueError(f"Schema {schema_name} não encontrado")
        
        if format == "json":
            return json.dumps(schema_details, ensure_ascii=False, indent=2, default=str)
        
        elif format == "yaml":
            try:
                import yaml
                return yaml.dump(schema_details, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML não está instalado")
        
        elif format == "python":
            return self._schema_to_python_code(schema_details)
        
        else:
            raise ValueError(f"Formato {format} não suportado")
    
    def _schema_to_python_code(self, schema_details: Dict[str, Any]) -> str:
        """Converte schema para código Python."""
        
        try:
            import pandera as pa
        except ImportError:
            raise ImportError("Pandera não está instalado")
        
        metadata = schema_details.get("metadata", {})
        columns = metadata.get("columns", {})
        
        code_lines = [
            "import pandera as pa",
            "",
            f"schema = pa.DataFrameSchema(",
            f"    name='{metadata.get('name', 'schema')}',",
            f"    description='{metadata.get('description', '')}',",
            f"    strict={metadata.get('strict', True)},",
            f"    columns={{",
        ]
        
        for col_name, col_info in columns.items():
            pandas_dtype = self._get_pandera_dtype(col_info["pandas_dtype"])
            
            col_code = f"        '{col_name}': pa.Column("
            col_code += f"dtype={pandas_dtype.__name__}, "
            col_code += f"nullable={col_info.get('nullable', True)}, "
            col_code += f"required={not col_info.get('nullable', True)}"
            
            # Adicionar checks se houver
            checks = []
            if col_info.get("unique_values", False):
                checks.append("pa.Check.unique()")
            if "min_value" in col_info:
                checks.append(f"pa.Check.ge({col_info['min_value']})")
            if "max_value" in col_info:
                checks.append(f"pa.Check.le({col_info['max_value']})")
            if col_info.get("categories"):
                checks.append(f"pa.Check.isin({col_info['categories']})")
            
            if checks:
                col_code += f", checks=[{', '.join(checks)}]"
            
            col_code += ")"
            code_lines.append(col_code)
        
        code_lines.extend([
            "    }",
            ")",
            "",
            "# Exemplo de uso:",
            "# validated_df = schema.validate(your_dataframe)"
        ])
        
        return "\n".join(code_lines)


def get_data_validator() -> DataValidator:
    """Factory function para DataValidator."""
    return DataValidator()
