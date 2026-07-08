"""
merge_delta_weights.py

Standalone script to merge STLLaVA-Med delta weights with the LLaVA 1.5 Base Model.
Designed specifically for Kaggle environments to stay strictly under the 29GB CPU RAM limit
by processing safetensor shards incrementally.

Requirements:
- transformers
- safetensors
- torch
"""

import os
import glob
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors.torch import load_file

def main():
    base_path = "/kaggle/input/models/systemsuperadmin/stllava02/pytorch/default/1/llava-v1.5-7b-files"
    delta_path = "/kaggle/input/models/systemsuperadmin/stllava01/pytorch/default/1/stllava-med-7b-files"
    output_path = "/kaggle/working/stllava-med-final"
    
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 60)
    print(" STLLaVA-Med CPU-Merging Pipeline")
    print("=" * 60)
    
    # 1. Copy config and tokenizer files from delta path
    print(f"\n[1/3] Copying configuration and tokenizer files from {delta_path}...")
    for filename in os.listdir(delta_path):
        if not filename.endswith(".safetensors") and not filename.endswith(".bin") and not filename.endswith(".pt"):
            src = os.path.join(delta_path, filename)
            dst = os.path.join(output_path, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
    print("Files copied successfully.")
    
    print(f"\n[2/3] Loading Base Model into CPU RAM (Takes ~14GB)...")
    # Load base model into CPU using FP16 to keep memory at ~14GB
    # This handles any format (safetensors/bin, sharded/unsharded) automatically
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, 
        device_map="cpu", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    base_sd = base_model.state_dict()
    
    print(f"\n[3/3] Applying Delta Weights and Saving...")
    delta_files = glob.glob(os.path.join(delta_path, "*.safetensors"))
    if not delta_files:
        # Fallback to .bin if safetensors don't exist
        delta_files = glob.glob(os.path.join(delta_path, "*.bin"))
        
    if not delta_files:
        raise ValueError(f"No delta weight files (.safetensors or .bin) found in {delta_path}")
        
    delta_files.sort()
    for delta_file in delta_files:
        print(f"  -> Merging {os.path.basename(delta_file)}...")
        if delta_file.endswith(".safetensors"):
            delta_dict = load_file(delta_file, device="cpu")
        else:
            delta_dict = torch.load(delta_file, map_location="cpu")
            
        for k, v in delta_dict.items():
            if k in base_sd:
                # Math: W_final = W_base + W_delta
                base_sd[k].data += v.to(torch.float16)
            else:
                # E.g. new projector weights
                base_sd[k] = v.to(torch.float16)
                
        # Free memory
        del delta_dict
        import gc
        gc.collect()
        
    print(f"\nSaving fully merged model to {output_path}...")
    base_model.save_pretrained(output_path, safe_serialization=True)
    
    print("\nMerge complete! Standalone model is fully prepared.")

if __name__ == "__main__":
    main()
