import torch
from typing import Optional

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
    # Older code expects past_key_values to be a tuple: past_key_values[-1][-1]
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, '__getitem__'):
            def dynamic_cache_getitem(self, idx):
                return (self.key_cache[idx], self.value_cache[idx])
            DynamicCache.__getitem__ = dynamic_cache_getitem
        if not hasattr(DynamicCache, '__len__'):
            def dynamic_cache_len(self):
                return len(self.key_cache)
            DynamicCache.__len__ = dynamic_cache_len
    except Exception:
        pass
