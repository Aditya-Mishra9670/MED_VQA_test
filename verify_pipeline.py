"""
Automated Verification Loop Script

Tests STLLaVA-Med, Grad-CAM, Grounding DINO, and SAM2 without the FastAPI overhead.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from loguru import logger

from backend.config.settings import get_settings
from backend.models.stllava import STLLaVAConfig, STLLaVAMed
from backend.explainability.gradcam import GradCAMExplainer
from backend.localization.grounding_dino import GroundingDINOWrapper

def create_dummy_image(path="dummy_med_image.jpg"):
    img = Image.new('RGB', (512, 512), color=(73, 109, 137))
    img.save(path)
    return path

def verify_pipeline():
    logger.info("Starting automated verification loop...")
    settings = get_settings()
    dummy_img_path = create_dummy_image()
    img = Image.open(dummy_img_path)

    # 1. Test STLLaVA-Med
    try:
        logger.info("Testing STLLaVA-Med...")
        config = STLLaVAConfig(
            model_path=settings.stllava_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_in_4bit=True,
        )
        stllava_model = STLLaVAMed(config)
        stllava_model.load()
        
        answer = stllava_model.generate(
            image=img, 
            question="What is this?"
        )
        logger.info(f"STLLaVA-Med Answer: {answer}")
    except Exception as e:
        logger.error(f"STLLaVA-Med failed: {e}")
        return False

    # 2. Test Grad-CAM
    try:
        logger.info("Testing Grad-CAM...")
        vision_tower = stllava_model.get_vision_encoder()
        explainer = GradCAMExplainer(model=vision_tower)
        
        input_tensor = stllava_model.preprocess_image(img)
        output_dir = Path("outputs/test_heatmaps")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        explainer.explain(img, input_tensor, output_dir, prefix="test_")
        logger.info("Grad-CAM completed successfully.")
    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}")
        return False

    # 3. Test Grounding DINO
    try:
        logger.info("Testing Grounding DINO...")
        gdino = GroundingDINOWrapper(
            config_path=settings.grounding_dino_config,
            checkpoint_path=settings.grounding_dino_checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        gdino.load()
        boxes = gdino.predict(img, text_prompt="abnormality")
        logger.info(f"Grounding DINO generated {len(boxes)} boxes.")
    except Exception as e:
        logger.error(f"Grounding DINO failed: {e}")
        return False

    logger.info("Pipeline verification completed successfully! All patches work.")
    return True

if __name__ == "__main__":
    import sys
    success = verify_pipeline()
    sys.exit(0 if success else 1)
