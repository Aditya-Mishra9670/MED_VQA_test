import torch
from contextlib import contextmanager
from typing import Optional


@contextmanager
def patch_tokenizer_loading():
    """
    Context manager that forces AutoTokenizer.from_pretrained to use
    use_fast=False.
    """
    from transformers import AutoTokenizer
    from unittest.mock import patch

    _original_from_pretrained = AutoTokenizer.from_pretrained.__func__

    @classmethod
    def _patched_from_pretrained(cls, *args, **kwargs):
        kwargs["use_fast"] = False
        return _original_from_pretrained(cls, *args, **kwargs)

    with patch.object(AutoTokenizer, 'from_pretrained', _patched_from_pretrained):
        yield

@contextmanager
def patch_model_loading_kwargs():
    """
    Intercepts load_in_8bit/4bit to convert to BitsAndBytesConfig.
    """
    from transformers.modeling_utils import PreTrainedModel
    from unittest.mock import patch
    
    _orig_from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _patched_from_pretrained(cls, *args, **kwargs):
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        
        if (load_in_8bit or load_in_4bit) and "quantization_config" not in kwargs:
            try:
                import torch
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    llm_int8_skip_modules=['mm_projector', 'vision_tower', 'vision_model'],
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except ImportError:
                pass

        return _orig_from_pretrained(cls, *args, **kwargs)

    with patch.object(PreTrainedModel, 'from_pretrained', _patched_from_pretrained):
        yield


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0):
    """Fallback implementation for older LLaVA models."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Fallback implementation for older LLaVA models."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def apply_transformers_patches():
    """
    Applies runtime monkey-patches to `transformers` to support STLLaVA-Med.
    Older forks of LLaVA import internal methods from transformers which were 
    deleted in >=4.37.0. We inject them dynamically to prevent crashes.
    """
    import sys
    
    # List of all transformer modeling files that older LLaVA forks might import from
    target_modules = [
        "transformers.models.bloom.modeling_bloom",
        "transformers.models.llama.modeling_llama",
        "transformers.models.opt.modeling_opt",
        "transformers.models.gpt_neox.modeling_gpt_neox",
        "transformers.models.gptj.modeling_gptj",
    ]
    
    for module_name in target_modules:
        try:
            # Try to import the module
            import importlib
            module = importlib.import_module(module_name)
            
            # Inject missing functions if they don't exist
            if not hasattr(module, '_expand_mask'):
                setattr(module, '_expand_mask', _expand_mask)
            if not hasattr(module, '_make_causal_mask'):
                setattr(module, '_make_causal_mask', _make_causal_mask)
        except Exception:
            pass

    # Patch DynamicCache to be subscriptable for older LLaVA code
    # Older code expects past_key_values to be a tuple-of-tuples:
    #   past_key_values[-1][-1].shape[-2]  → gets value tensor's seq_len
    #
    # API evolution across transformers versions:
    #   - Old (<=4.40): DynamicCache with self.key_cache / self.value_cache lists
    #   - New (>=4.48): DynamicCache with self._cache list of DynamicLayer objects
    #     where each DynamicLayer has .keys and .values (plural) tensors
    try:
        from transformers.cache_utils import DynamicCache

        def _extract_kv_from_layer(layer):
            """Extract (key, value) tensors from a cache layer object."""
            # Newest API: DynamicLayer uses .keys / .values (plural)
            if hasattr(layer, 'keys') and hasattr(layer, 'values'):
                return (layer.keys, layer.values)
            # Some versions might use singular .key / .value
            if hasattr(layer, 'key') and hasattr(layer, 'value'):
                return (layer.key, layer.value)
            # If the layer is itself a tuple (key, value)
            if isinstance(layer, tuple) and len(layer) == 2:
                return layer
            return None

        def dynamic_cache_getitem(self, idx):
            # Newest API: _cache list of DynamicLayer objects
            if hasattr(self, '_cache') and isinstance(self._cache, list):
                layer = self._cache[idx]
                result = _extract_kv_from_layer(layer)
                if result is not None:
                    return result

            # Old API: key_cache / value_cache parallel lists
            if hasattr(self, 'key_cache') and hasattr(self, 'value_cache'):
                return (self.key_cache[idx], self.value_cache[idx])

            # Fallback: try iterating (DynamicCache may expose __iter__)
            try:
                # Remove our __getitem__ temporarily to avoid recursion
                cache_list = list(super(DynamicCache, self).__iter__())
                layer = cache_list[idx]
                result = _extract_kv_from_layer(layer)
                if result is not None:
                    return result
            except (TypeError, AttributeError, StopIteration):
                pass

            raise AttributeError(
                f"Cannot index DynamicCache: unrecognized internal structure. "
                f"Attrs: {[a for a in dir(self) if not a.startswith('__')]}"
            )

        def dynamic_cache_len(self):
            if hasattr(self, '_cache') and isinstance(self._cache, list):
                return len(self._cache)
            if hasattr(self, 'key_cache') and isinstance(self.key_cache, list):
                return len(self.key_cache)
            # Fallback
            try:
                return sum(1 for _ in super(DynamicCache, self).__iter__())
            except (TypeError, AttributeError):
                return 0

        # Always override to ensure our robust version is used
        DynamicCache.__getitem__ = dynamic_cache_getitem
        DynamicCache.__len__ = dynamic_cache_len
    except Exception:
        pass
