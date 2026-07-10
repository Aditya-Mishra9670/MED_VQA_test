"""
predict_standalone.py

Standalone inference script for STLLaVA-Med.
This is the mathematically sound, purely native pipeline.
"""

import torch
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

def main():
    # 1. THE BIGGEST FIX: STLLaVA-Med is a FULL MODEL, not a Delta model!
    # Our previous merge script was mathematically destroying the weights by adding two 14GB models together.
    # We point DIRECTLY to the provided full model folder in the read-only Kaggle input directory.
    model_path = "/kaggle/input/models/systemsuperadmin/stllava01/pytorch/default/1/stllava-med-7b-files"
    
    print("=" * 60)
    print(" STLLaVA-Med Standalone Inference (Native)")
    print("=" * 60)
    
    import transformers
    if transformers.__version__ != "4.31.0":
        raise EnvironmentError(
            f"Transformers version is {transformers.__version__}. STLLaVA-Med STRICTLY requires "
            f"transformers==4.31.0 (as well as accelerate==0.21.0 and bitsandbytes==0.41.0). "
            f"Please run:\n!pip install transformers==4.31.0 accelerate==0.21.0 bitsandbytes==0.41.0"
        )
        
    print("\n[1/4] Native Vision Tower Resolution...")
    from huggingface_hub import snapshot_download
    print(f"  -> Downloading STLLaVA vision tower to local cache to satisfy LLaVA's absolute path requirement...")
    local_vt_path = snapshot_download(repo_id="ZachSun/stllava-med-7b-vit", local_dir="/kaggle/working/vision_tower_cache")
    print(f"  -> Vision tower cached at: {local_vt_path}")

    print("\n[2/4] Loading Full Model Natively in 4-bit...")
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    
    model_name = get_model_name_from_path(model_path)
    
    # We pass the local vision tower path dynamically via kwargs to override the read-only config.json in memory!
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_4bit=True,
        device="cuda",
        mm_vision_tower=local_vt_path  # Native override for the vision tower
    )
    
    print("\n[3/4] Preparing Input Image and Medical Prompt...")
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        import os
        print(f"  Warning: {image_path} not found. Creating a blank dummy image for testing.")
        Image.new('RGB', (224, 224), color = 'gray').save(image_path)
        
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().cuda()
    
    question = "What abnormalities are visible in this medical scan?"
    
    if getattr(model.config, 'mm_use_im_start_end', False):
        prompt_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        prompt_qs = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], prompt_qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    print("\n[4/4] Executing Forward Pass...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=512,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(conv.sep):
        outputs = outputs[:-len(conv.sep)]
        
    print("\n" + "=" * 60)
    print(" GENERATED MEDICAL ANSWER:")
    print("=" * 60)
    print(outputs)
    print("=" * 60)

if __name__ == "__main__":
    import os
    main()
