"""
STLLaVA-Med model wrapper.

Wraps the STLLaVA-Med Vision-Language Model for medical VQA.
Handles image preprocessing, prompt formatting, and generation.

Includes a fallback loading path using transformers AutoModel
when the llava package is not available.

Reference: https://github.com/heliossun/STLLaVA-Med
Model: https://huggingface.co/collections/ZachSun/stllava-med-672464190c1f6b5b546e01a8
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from loguru import logger


@dataclass
class STLLaVAConfig:
    """Configuration for STLLaVA-Med model."""

    model_path: str = "ZachSun/stllava-med-7b"
    model_base: str = "liuhaotian/llava-v1.5-7b"
    device: str = "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.7
    num_beams: int = 1
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    conv_mode: str = "vicuna_v1"


class STLLaVAMed:
    """
    STLLaVA-Med model wrapper for medical Visual Question Answering.

    This class handles:
    - Model and tokenizer loading (via LLaVA architecture)
    - Image preprocessing with CLIP vision encoder
    - Medical prompt formatting
    - Text generation with configurable parameters

    Includes a fallback loading path when the llava package
    is not directly importable.
    """

    def __init__(self, config: STLLaVAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self._loaded = False

    def load(self) -> None:
        """
        Load the STLLaVA-Med model, tokenizer, and image processor.

        Tries the LLaVA package first, then falls back to direct
        transformers loading if llava is unavailable.
        """
        if self._loaded:
            logger.info("STLLaVA-Med model already loaded, skipping.")
            return

        logger.info(
            f"Loading STLLaVA-Med from {self.config.model_path} "
            f"(base: {self.config.model_base}) on {self.config.device}"
        )

        # Ensure the llava package is available
        try:
            from backend.models.model_manager import ModelManager
            manager = ModelManager()
            manager.ensure_llava_package()
        except Exception as e:
            logger.debug(f"Model manager unavailable: {e}")

        # Try primary loading path (llava package)
        try:
            self._load_via_llava()
            return
        except Exception as e:
            logger.error(f"LLaVA loading failed: {e}")
            raise RuntimeError(f"Failed to load STLLaVA-Med via LLaVA package: {e}") from e

    def _load_via_llava(self) -> None:
        """Load using the LLaVA package's model builder with compatibility shims."""
        import transformers
        from unittest.mock import patch

        # Older LLaVA packages try to register 'llava' which conflicts with new transformers
        orig_register_config = transformers.AutoConfig.register
        orig_register_model = transformers.AutoModelForCausalLM.register

        def patched_register_config(cls, model_type, config_class, **kwargs):
            kwargs['exist_ok'] = True
            return orig_register_config(model_type, config_class, **kwargs)

        def patched_register_model(cls, config_class, model_class, **kwargs):
            kwargs['exist_ok'] = True
            return orig_register_model(config_class, model_class, **kwargs)

        # Apply robust backward compatibility patches for deleted transformers functions
        from backend.utils.transformers_patch import (
            apply_transformers_patches,
            patch_tokenizer_loading,
            patch_model_loading_kwargs
        )
        apply_transformers_patches()

        with patch.object(transformers.AutoConfig, 'register', classmethod(patched_register_config)), \
             patch.object(transformers.AutoModelForCausalLM, 'register', classmethod(patched_register_model)):
            
            from huggingface_hub import snapshot_download
            import os
            
            # Download the custom STLLaVA vision tower to satisfy LLaVA's absolute path requirement natively
            logger.info("Resolving native vision tower (ZachSun/stllava-med-7b-vit)...")
            cache_dir = "/kaggle/working/vision_tower_cache" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else str(Path(self.config.model_path).parent / "vision_tower_cache")
            local_vt_path = snapshot_download(repo_id="ZachSun/stllava-med-7b-vit", local_dir=cache_dir)

            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path

            model_name = get_model_name_from_path(self.config.model_path)
            if "llava" not in model_name.lower():
                model_name = "llava-v1.5-7b"

            # STLLaVA-Med is a full model containing all safetensors and tokenizer files.
            model_base = None

            device = self.config.device
            if device == "mps":
                device = "cpu"

            logger.info(f"Loading full model natively from {self.config.model_path}...")
            
            from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

            def patched_build_vision_tower(vision_tower_cfg, **kwargs):
                vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
                if vision_tower and "stllava-med" in vision_tower.lower():
                    # Force it to use our localized downloaded vision tower path
                    return CLIPVisionTower(local_vt_path, args=vision_tower_cfg, **kwargs)
                raise ValueError(f'Unknown vision tower: {vision_tower}')

            with patch_tokenizer_loading(), \
                 patch_model_loading_kwargs(), \
                 patch('llava.model.llava_arch.build_vision_tower', patched_build_vision_tower), \
                 patch('llava.model.multimodal_encoder.builder.build_vision_tower', patched_build_vision_tower):
                 
                self.tokenizer, self.model, self.image_processor, self.context_len = (
                    load_pretrained_model(
                        model_path=self.config.model_path,
                        model_base=model_base,
                        model_name=model_name,
                        load_8bit=self.config.load_in_8bit,
                        load_4bit=self.config.load_in_4bit,
                        device=device,
                    )
                )

        if self.config.device == "mps" and device == "cpu":
            try:
                self.model = self.model.to("mps")
                logger.info("Model moved to MPS device")
            except Exception:
                logger.warning("Could not move model to MPS, staying on CPU")

        self._loaded = True
        logger.info(
            f"STLLaVA-Med loaded successfully via LLaVA package. "
            f"Context length: {self.context_len}"
        )

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for the vision encoder.

        Args:
            image: PIL Image in RGB mode.

        Returns:
            Preprocessed image tensor ready for the model.
        """
        if self.image_processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        image = image.convert("RGB")

        image_tensor = self.image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"]

        # Determine dtype based on device and model precision
        device = self.config.device
        if hasattr(self, "model") and hasattr(self.model, "dtype"):
            image_tensor = image_tensor.to(dtype=self.model.dtype, device=device)
        else:
            if device == "cpu":
                image_tensor = image_tensor.float()
            else:
                image_tensor = image_tensor.half()
            image_tensor = image_tensor.to(device)

        return image_tensor

    def format_prompt(self, question: str) -> str:
        """
        Format a medical question into the LLaVA conversation format.

        Args:
            question: The medical question to ask about the image.

        Returns:
            Formatted prompt string ready for tokenization.
        """
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates

        # Build the prompt with image token
        if hasattr(self.model, "config") and getattr(
            self.model.config, "mm_use_im_start_end", False
        ):
            prompt_text = (
                f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}"
                f"{DEFAULT_IM_END_TOKEN}\n{question}"
            )
        else:
            prompt_text = f"{DEFAULT_IMAGE_TOKEN}\n{question}"

        # Use conversation template
        conv = conv_templates[self.config.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)

        return conv.get_prompt()

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate an answer for a medical image + question pair.

        Args:
            image: PIL Image (medical image — X-ray, CT, MRI, etc.).
            question: Natural language medical question.
            max_new_tokens: Override default max tokens for generation.
            temperature: Override default temperature for generation.

        Returns:
            Generated text answer.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature or self.config.temperature

        # Preprocess image
        image_tensor = self.preprocess_image(image)

        # Format and tokenize prompt
        prompt = self.format_prompt(question)

        return self._generate_llava(prompt, image_tensor, max_tokens, temp)

    def _generate_llava(
        self,
        prompt: str,
        image_tensor: torch.Tensor,
        max_tokens: int,
        temp: float,
    ) -> str:
        """Generate using the LLaVA package."""
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.config.device)

        # Explicitly passing attention_mask with unexpanded input_ids breaks LLaVA generation 
        # because the input gets expanded with image patches inside forward(), causing mask size mismatch.
        # For batch size 1, attention_mask is safely omitted.

        from llava.conversation import conv_templates
        
        conv = conv_templates[self.config.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != 1 else conv.sep2

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else 1.0,
                top_p=self.config.top_p,
                num_beams=self.config.num_beams,
                max_new_tokens=max_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output (skip input tokens)
        output_text = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()
        
        # Clean up SentencePiece artifacts and stop strings
        output_text = output_text.replace(" ", " ").replace("\u2581", " ")
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)].strip()

        return output_text

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded

    def get_vision_encoder(self):
        """
        Access the vision encoder for Grad-CAM.

        Returns:
            The vision tower (CLIP ViT) module, or None if unavailable.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        # Try LLaVA method
        if hasattr(self.model, "get_vision_tower"):
            try:
                tower = self.model.get_vision_tower()
                if tower is not None:
                    # Return the underlying HuggingFace model to bypass LLaVA's 
                    # @torch.no_grad() forward method which breaks Grad-CAM
                    if hasattr(tower, "vision_tower"):
                        return tower.vision_tower
                    return tower
            except Exception:
                pass

        # Try direct attribute access
        for attr in ["vision_tower", "vision_model", "visual"]:
            if hasattr(self.model, attr):
                return getattr(self.model, attr)

        logger.warning(
            "Could not access vision encoder. "
            "Grad-CAM may not work with this model configuration."
        )
        return None
