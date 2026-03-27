from src.core.processor import AutoMLDataProcessor
from src.core.trainer import TransformersWrapper
from src.engines import classical as _classical_engine
from src.engines.classical import AutoMLTrainer as _BaseAutoMLTrainer, load_pipeline, save_pipeline

TRANSFORMERS_AVAILABLE = _classical_engine.TRANSFORMERS_AVAILABLE


class AutoMLTrainer(_BaseAutoMLTrainer):
    """Compatibility adapter for unified interface and test-time patching."""

    @staticmethod
    def _sync_transformer_runtime():
        # Keep runtime flags bound to this module so patch('automl_engine.*') works.
        _classical_engine.TRANSFORMERS_AVAILABLE = TRANSFORMERS_AVAILABLE
        _classical_engine.TransformersWrapper = TransformersWrapper

    def get_supported_models(self):
        return self.get_available_models()

    def _instantiate_model(self, name, params):
        self._sync_transformer_runtime()
        return super()._instantiate_model(name, params)

    def get_model_params_schema(self, model_name):
        self._sync_transformer_runtime()
        return super().get_model_params_schema(model_name)

__all__ = [
    "AutoMLDataProcessor",
    "AutoMLTrainer",
    "TransformersWrapper",
    "TRANSFORMERS_AVAILABLE",
    "load_pipeline",
    "save_pipeline",
]