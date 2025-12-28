import argparse
import shutil
import os
import torch
from modeling_biqwen_local import BiQwen3ForMaskedLM
from configuration_biqwen import BiQwen3Config
from huggingface_hub import create_branch, HfApi
from huggingface_hub.utils import RepositoryNotFoundError


def main():
    print("Uploading model folder to Hugging Face Hub")
    parser = argparse.ArgumentParser(description="Convert model to Hugging Face format and push to hub")
    parser.add_argument("--weight_path", type=str, required=True, help="Directory containing the DCP checkpoint")
    parser.add_argument("--model_size", type=str, default="600m", help="Model size (default: '600m')")
    parser.add_argument("--organization", type=str, required=True, help="Hugging Face organization to push the model to")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to push to the hub")
    parser.add_argument("--private", action="store_true", default=False, help="Whether to make the repository private (default: False)")
    parser.add_argument("--revision", action="store_true", default=False, help="Used model folder name as revision (default: False)")
    parser.add_argument("--token", default="", type=str, help="Hugging Face token for authentication")

    args = parser.parse_args()

    # Remove tmp folder if it exists
    try:
        shutil.rmtree("./tmp")
    except FileNotFoundError:
        pass
    os.makedirs("./tmp", exist_ok=True)

    # Load config and model
    print(f"Loading config for model size: {args.model_size}")
    config = BiQwen3Config.from_pretrained(f'{args.model_size}.json')
    model = BiQwen3ForMaskedLM(config)

    # Load checkpoint weights
    print(f"Loading state_dict from: {args.weight_path}/model.pt")
    state_dict_path = os.path.join(args.weight_path, "model.pt")
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.save_pretrained("./tmp")

    # Copy python source files to tmp for completeness
    print("Copying configuration and modeling files to tmp directory")
    shutil.copy2("./configuration_biqwen.py", "./tmp/configuration_biqwen.py")
    shutil.copy2("./modeling_biqwen.py", "./tmp/modeling_biqwen.py")
    shutil.copy2("./tokenizer_config.json", "./tmp/tokenizer_config.json")
    shutil.copy2("./tokenizer.json", "./tmp/tokenizer.json")
    shutil.copy2("./vocab.json", "./tmp/vocab.json")

    api = HfApi(token=args.token)
    try:
        api.repo_info(repo_id=f"{args.organization}/{args.model_name}", repo_type="model")
        print("Model already exists on the hub.")
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id=f"{args.organization}/{args.model_name}",
            repo_type="model",
            private=args.private,
            token=args.token
        )
        
    if args.revision:
        revision = args.weight_path.split("/")[-1]
        print("Creating a new branch for the revision:", revision)
        create_branch(
            repo_id=f"{args.organization}/{args.model_name}",
            branch=revision,
            repo_type="model",
            token=args.token
        )

    # Build the command to run
    command = (
        f'huggingface-cli upload-large-folder '
        f'"{args.organization}/{args.model_name}" '
        f'--repo-type=model '
        f'"./tmp" '
        f'--num-workers=16 '
        f'{"--token " + args.token if args.token else ""} '
        f'--revision {revision if args.revision else "main"} '
        f'{"--private" if args.private else ""}'
    )

    print(f"Running command:\n{command}")
    os.system(command)

if __name__ == "__main__":
    main()