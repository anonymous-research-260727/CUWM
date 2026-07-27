python3 - << 'EOF'
from huggingface_hub import snapshot_download
import os

MODEL_BASE = os.environ.get("MODEL_BASE", "models/Qwen")

models = {
    "Qwen/Qwen-Image-Edit": os.path.join(MODEL_BASE, "Qwen-Image-Edit"),
}

for repo_id, target_dir in models.items():
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=target_dir,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print(f"Done: {repo_id}\n")

print("All models downloaded successfully!")
EOF
