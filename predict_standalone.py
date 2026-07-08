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
    
    import transformers
    if transformers.__version__ > "4.36.2":
        raise EnvironmentError(
            f"Transformers version is {transformers.__version__}. STLLaVA-Med technically requires "
            f"transformers==4.36.2 to run natively without monkey patches. "
            f"Please run: !pip install transformers==4.36.2"
        )
        
    print("\n[1/4] Preparing Config and Vision Tower (Technical Fix)...")
    import json
    import os
    config_path = os.path.join(merged_model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
            
        vision_tower_id = config_data.get("mm_vision_tower", "")
        # If the vision tower is the remote ZachSun ID, download it to satisfy the absolute path requirement natively
        if vision_tower_id == "ZachSun/stllava-med-7b-vit":
            print(f"  -> Detected remote vision tower: {vision_tower_id}. Downloading to local cache to satisfy LLaVA's absolute path requirement natively...")
            from huggingface_hub import snapshot_download
            local_vt_path = snapshot_download(repo_id=vision_tower_id, local_dir="/kaggle/working/vision_tower_cache")
            config_data["mm_vision_tower"] = local_vt_path
            
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            print(f"  -> config.json updated with absolute vision tower path: {local_vt_path}")

    print("\n[2/4] Importing LLaVA components (Native)...")
    try:
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
