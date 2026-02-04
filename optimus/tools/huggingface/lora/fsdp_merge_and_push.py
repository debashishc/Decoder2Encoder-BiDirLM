import argparse
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer
import os
import shutil
from huggingface_hub import HfApi

def fsdp_merge_and_push(
    base_model_path: str,
    checkpoint_dir: str,
    hub_repo_id: str,
    revision: str = "main",
    hf_token: str = None,
    private: bool = False
):
    print("=== Starting Workflow: FSDP -> Merge -> Push ===")
    
    print(f"[1/6] Loading Base Model from: {base_model_path}")
    model = AutoModel.from_pretrained(
        base_model_path,
        device_map="cpu", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print(f"[2/6] Initializing LoRA Config from: {checkpoint_dir}")
    peft_config = PeftConfig.from_pretrained(checkpoint_dir)
    model = PeftModel(model, peft_config)
    
    print("[3/6] Mapping and Loading FSDP Weights...")
    local_state_dict = model.state_dict()

    reader = FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata.keys())
    
    dcp_load_plan = {}
    
    for local_key, local_tensor in local_state_dict.items():
        core_key = local_key.replace("base_model.model.", "")
        
        found_in_checkpoint = False
        for ck in checkpoint_keys:
            if ck.endswith(core_key):
                dcp_load_plan[ck] = local_tensor
                found_in_checkpoint = True
                break
        
        if "embed_tokens" in local_key and found_in_checkpoint:
            print(f"      -> FOUND TRAINED EMBEDDING: {local_key}")
        if "lm_head" in local_key and found_in_checkpoint:
            print(f"      -> FOUND TRAINED HEAD: {local_key}")

    mapped_count = len(dcp_load_plan)
    print(f"      Mapped {mapped_count} tensors from checkpoint.")
    
    if mapped_count == 0:
        raise RuntimeError("No keys matched! The checkpoint might be empty or format is completely wrong.")
    
    dcp.load(state_dict=dcp_load_plan, checkpoint_id=checkpoint_dir)
    print("      FSDP weights loaded successfully.")

    print("[4/6] Merging LoRA into Base Model...")
    model = model.merge_and_unload()
    
    print("[5/6] Saving merged model to temporary folder...")
    temp_output_dir = "temp_merged_model_upload"
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir)

    model.save_pretrained(temp_output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(temp_output_dir)
    
    print(f"[6/6] Pushing to Hub: {hub_repo_id} (Branch: {revision})")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hub_repo_id, exist_ok=True, private=private)
    
    if revision != "main":
        api.create_branch(repo_id=hub_repo_id, branch=revision, exist_ok=True)

    api.upload_folder(
        folder_path=temp_output_dir,
        repo_id=hub_repo_id,
        revision=revision,
        repo_type="model"
    )
    
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_output_dir)
    
    print(f"\nSUCCESS! Model pushed to https://huggingface.co/{hub_repo_id}/tree/{revision}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--hub_repo_id", type=str, required=True)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--private", action="store_true", default=False)
    
    args = parser.parse_args()

    fsdp_merge_and_push(
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_dir,
        hub_repo_id=args.hub_repo_id,
        revision=args.revision,
        hf_token=args.hf_token,
        private=args.private
    )