from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir=r"C:\Users\NSP\.cache\huggingface\hub",
    local_dir_use_symlinks=False
)
