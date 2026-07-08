"""
predict_standalone.py

Standalone inference script for the merged STLLaVA-Med model.
Uses 4-bit NF4 quantization to comfortably fit the T4 GPU and executes
a clean, guaranteed-stable greedy decoding forward pass.
"""

import torch
import warnings
from PIL import Image
from transformers import BitsAndBytesConfig

# Suppress annoying HuggingFace warnings
warnings.filterwarnings("ignore")

def main():
    merged_model_path = "/kaggle/working/stllava-med-final"
    
    print("=" * 60)
    print(" STLLaVA-Med Standalone Inference (T4 GPU)")
    print("=" * 60)
    
    print("\n[1/4] Importing LLaVA components (with transformers patch)...")
    try:
        import transformers
        from unittest.mock import patch
        
        orig_register_config = transformers.AutoConfig.register
        orig_register_model = transformers.AutoModelForCausalLM.register

        def patched_register_config(cls, model_type, config_class, **kwargs):
            kwargs['exist_ok'] = True
            return orig_register_config(model_type, config_class, **kwargs)

        def patched_register_model(cls, config_class, model_class, **kwargs):
            kwargs['exist_ok'] = True
            return orig_register_model(config_class, model_class, **kwargs)
            
        # Patch missing functions for older LLaVA forks (bloom, llama, etc.)
        def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
            bsz, tgt_len = input_ids_shape
            mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(dtype)
            if past_key_values_length > 0:
                mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
            return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

        def _expand_mask(mask, dtype, tgt_len=None):
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len
            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            inverted_mask = 1.0 - expanded_mask
            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

        target_modules = [
            "transformers.models.bloom.modeling_bloom",
            "transformers.models.llama.modeling_llama",
            "transformers.models.opt.modeling_opt",
            "transformers.models.gpt_neox.modeling_gpt_neox",
            "transformers.models.gptj.modeling_gptj",
        ]
        
        import importlib
        for module_name in target_modules:
            try:
                module = importlib.import_module(module_name)
                if not hasattr(module, '_expand_mask'):
                    setattr(module, '_expand_mask', _expand_mask)
                if not hasattr(module, '_make_causal_mask'):
                    setattr(module, '_make_causal_mask', _make_causal_mask)
            except Exception:
                pass
                
        # Patch PreTrainedModel to gracefully handle load_in_4bit in newer transformers
        from transformers.modeling_utils import PreTrainedModel
        orig_from_pretrained_model = PreTrainedModel.from_pretrained.__func__
        
        @classmethod
        def patched_from_pretrained_model(cls, *args, **kwargs):
            load_in_8bit = kwargs.pop("load_in_8bit", False)
            load_in_4bit = kwargs.pop("load_in_4bit", False)
            
            if (load_in_8bit or load_in_4bit) and "quantization_config" not in kwargs:
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=load_in_8bit,
                        load_in_4bit=load_in_4bit,
                        llm_int8_skip_modules=['mm_projector', 'vision_tower', 'vision_model'],
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                except ImportError:
                    pass

            return orig_from_pretrained_model(cls, *args, **kwargs)

        with patch.object(transformers.AutoConfig, 'register', classmethod(patched_register_config)), \
             patch.object(transformers.AutoModelForCausalLM, 'register', classmethod(patched_register_model)), \
             patch.object(PreTrainedModel, 'from_pretrained', patched_from_pretrained_model):
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from llava.conversation import conv_templates, SeparatorStyle
    except ImportError:
        raise ImportError("The 'llava' package is not installed. Please install it from STLLaVA-Med.")

    print(f"\n[2/4] Loading model from {merged_model_path} in 4-bit nf4...")
    # LLaVA's load_pretrained_model automatically applies the correct BitsAndBytesConfig when load_4bit=True
    model_name = get_model_name_from_path(merged_model_path)
    
    # We must mock tokenizer loading slightly just in case of use_fast=True issues
    import transformers
    orig_from_pretrained = transformers.AutoTokenizer.from_pretrained
    def patched_from_pretrained(*args, **kwargs):
        kwargs['use_fast'] = False
        return orig_from_pretrained(*args, **kwargs)
    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=merged_model_path,
        model_base=None,
        model_name=model_name,
        load_4bit=True,
        device="cuda"
    )
    
    print("\n[3/4] Preparing input image and prompt...")
    # Example input image
    image_path = "test_image.jpg" # Update this to your actual test image path
    if not os.path.exists(image_path):
        # Create a dummy blank image for testing if none exists
        print(f"  Warning: {image_path} not found. Creating a blank dummy image for testing.")
        Image.new('RGB', (224, 224), color = 'gray').save(image_path)
        
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().cuda()
    
    # Construct the strictly required LLaVA conversation template
    question = "What abnormalities are visible in this medical scan?"
    
    if getattr(model.config, 'mm_use_im_start_end', False):
        prompt_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        prompt_qs = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], prompt_qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(f"  Constructed Prompt:\n{prompt}")

    # Tokenize input and explicitly create attention_mask
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    attention_mask = torch.ones_like(input_ids).cuda()

    print("\n[4/4] Executing forward pass (Greedy Decoding)...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            do_sample=False,        # Greedy decoding for absolute stability
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

    # Decode outputs
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"  [Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(conv.sep):
        outputs = outputs[:-len(conv.sep)]
        
    # Final cleanup for weird SentencePiece token artifacts
    outputs = outputs.replace(" ", " ")

    print("\n" + "=" * 60)
    print(" GENERATED MEDICAL ANSWER:")
    print("=" * 60)
    print(outputs)
    print("=" * 60)

if __name__ == "__main__":
    import os
    main()
