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
import json
import shutil
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def main():
    base_path = "/kaggle/input/models/systemsuperadmin/stllava02/pytorch/default/1/llava-v1.5-7b-files"
    delta_path = "/kaggle/input/models/systemsuperadmin/stllava01/pytorch/default/1/stllava-med-7b-files"
    output_path = "/kaggle/working/stllava-med-final"
    
    os.makedirs(output_path, exist_ok=True)
    
    print("=" * 60)
    print(" STLLaVA-Med CPU-Merging Pipeline")
    print("=" * 60)
    
    # 1. Copy config and tokenizer files from delta path
    print(f"\n[1/4] Copying configuration and tokenizer files from {delta_path}...")
    for filename in os.listdir(delta_path):
        if not filename.endswith(".safetensors") and not filename.endswith(".bin"):
            src = os.path.join(delta_path, filename)
            dst = os.path.join(output_path, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
    print("Files copied successfully.")
    
    # 2. Build map of Base safetensors
    base_index_file = os.path.join(base_path, "model.safetensors.index.json")
    if not os.path.exists(base_index_file):
        raise FileNotFoundError(f"Base index not found: {base_index_file}")
        
    with open(base_index_file, "r") as f:
        base_index = json.load(f)
        
    # Build a map of tensor_name -> base_shard_file
    base_tensor_map = base_index.get("weight_map", {})
    
    # 3. Process Delta safetensors one by one to save RAM
    delta_files = glob.glob(os.path.join(delta_path, "*.safetensors"))
    delta_files.sort()
    
    if not delta_files:
        raise ValueError(f"No .safetensors files found in delta path: {delta_path}")
        
    print(f"\n[2/4] Merging {len(delta_files)} delta shard(s)...")
    
    # We will build a new index for the merged weights
    new_weight_map = {}
    
    for delta_file in delta_files:
        shard_name = os.path.basename(delta_file)
        print(f"  -> Processing {shard_name}...")
        
        # Load the delta shard (loaded entirely into CPU RAM ~ 2-4GB)
        delta_dict = load_file(delta_file, device="cpu")
        merged_dict = {}
        
        # To minimize loading base shards multiple times, group required base tensors by shard
        required_base_shards = {}
        for tensor_name in delta_dict.keys():
            base_shard = base_tensor_map.get(tensor_name)
            if base_shard:
                required_base_shards.setdefault(base_shard, []).append(tensor_name)
                
        # Load required base shards incrementally
        base_shard_dicts = {}
        for base_shard, tensor_names in required_base_shards.items():
            base_shard_path = os.path.join(base_path, base_shard)
            if not os.path.exists(base_shard_path):
                print(f"     Warning: Base shard missing {base_shard}")
                continue
            base_shard_dicts[base_shard] = load_file(base_shard_path, device="cpu")
            
        # Perform the merge mathematically
        for tensor_name, delta_tensor in tqdm(delta_dict.items(), desc=f"Merging {shard_name}", leave=False):
            base_shard = base_tensor_map.get(tensor_name)
            
            if base_shard and base_shard in base_shard_dicts and tensor_name in base_shard_dicts[base_shard]:
                base_tensor = base_shard_dicts[base_shard][tensor_name]
                # Match dtypes
                if base_tensor.dtype != delta_tensor.dtype:
                    base_tensor = base_tensor.to(delta_tensor.dtype)
                
                # Math: W_final = W_base + W_delta
                merged_tensor = base_tensor + delta_tensor
                merged_dict[tensor_name] = merged_tensor
            else:
                # If tensor is not in base (e.g., brand new projector weights or special tokens)
                merged_dict[tensor_name] = delta_tensor
                
            new_weight_map[tensor_name] = shard_name
            
        # Free memory of base shards
        del base_shard_dicts
        
        # Save the merged shard
        merged_file_path = os.path.join(output_path, shard_name)
        save_file(merged_dict, merged_file_path, metadata={"format": "pt"})
        
        # Free memory of merged/delta
        del delta_dict
        del merged_dict
        import gc
        gc.collect()
        
    print(f"\n[3/4] Generating new model.safetensors.index.json...")
    new_index = {
        "metadata": base_index.get("metadata", {}),
        "weight_map": new_weight_map
    }
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)
        
    print("\n[4/4] Merge complete!")
    print(f"Standalone model is fully prepared at: {output_path}")

if __name__ == "__main__":
    main()
