from huggingface_hub import snapshot_download

# 下载模型
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_dir="./Qwen2.5-0.5B-Instruct",  # 自定义保存路径
    revision="main"
)


# 下载数据集
snapshot_download(
    repo_id="swulling/gsm8k_chinese",
    repo_type="dataset",
    local_dir="./gsm8k_chinese",  #  自定义完整路径
)