import os
from huggingface_hub import HfApi, HfFileSystem, create_repo, login
from typing import Optional, Dict, Any
import joblib
import tempfile
import json

def deploy_to_huggingface(
    model_path: str,
    repo_id: str,
    token: str,
    private: bool = False,
    model_card_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Deploys a local model file (or directory) to Hugging Face Hub.
    Returns the URL of the repository.
    """
    try:
        api = HfApi(token=token)
        
        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        except Exception as e:
            print(f"Repo creation issue: {e}")

        # Upload files
        if os.path.isdir(model_path):
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                repo_type="model"
            )
        else:
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=os.path.basename(model_path),
                repo_id=repo_id,
                repo_type="model"
            )

        # Create/Update Model Card
        card_content = f"""---
language: en
license: mit
tags:
- automl
- scikit-learn
- generated-by-automlops-studio
---

# {repo_id.split('/')[-1]}

This model was trained using [AutoMLOps Studio](https://github.com/PedroM2626/automlops-studio).

## Metadata
```json
{json.dumps(model_card_data or {}, indent=2)}
```
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(card_content)
            temp_card = f.name
        
        api.upload_file(
            path_or_fileobj=temp_card,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        os.remove(temp_card)

        return f"https://huggingface.co/{repo_id}"
    except Exception as e:
        raise Exception(f"Hugging Face deployment failed: {str(e)}")

def test_inference_hf(repo_id: str, token: str, inputs: Dict[str, Any]) -> Any:
    """
    Tests inference using HF Inference API (if supported by the model type).
    Note: For scikit-learn models, this might require a custom handler or 
    generic 'predict' endpoint if the repo is set up for it.
    """
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=repo_id, token=token)
    # This is a generic placeholder. Actual call depends on task.
    # For tabular models, HF usually doesn't have a default 'inference' widget 
    # unless you use 'skops' or similar, but we can try generic requests.
    return "Inference triggered. Check repo status for API availability."
