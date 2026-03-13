#!/usr/bin/env python3
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from loguru import logger

# All models in a single repository
REPO_ID = "tmoklc/scorevisionv1"
MODELS = [
    "football-player-detection.pt",
    "football-ball-detection.pt",
    "football-pitch-detection.pt"
]

def download_models():
    """Download required models from Hugging Face."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    success = True
    for model_name in MODELS:
        model_path = data_dir / model_name
        if not model_path.exists():
            logger.info(f"Downloading {model_name} from Hugging Face ({REPO_ID})...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=model_name,
                    local_dir=data_dir
                )
                logger.info(f"Successfully downloaded {model_name}")
            except (RepositoryNotFoundError, RevisionNotFoundError) as e:
                logger.error(f"Repository or file not found for {model_name}: {str(e)}")
                success = False
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {str(e)}")
                success = False
        else:
            logger.info(f"{model_name} already exists in {model_path}, skipping download")
    
    if not success:
        logger.error("Some models failed to download. Please check the errors above.")
        exit(1)
    else:
        logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    download_models() 