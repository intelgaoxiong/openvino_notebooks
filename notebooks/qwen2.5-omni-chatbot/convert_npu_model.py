from pathlib import Path
from qwen2_5_omni_helper import convert_qwen2_5_omni_model
import nncf

# pip install -q "git+https://github.com/huggingface/transformers" torchvision" "accelerate" "qwen-omni-utils[decord]" "gradio>=4.19" --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu
# pip install -q "openvino==2025.1.0" "nncf>=2.16.0"

model_id = "Qwen/Qwen2.5-Omni-7B"
model_dir = Path(model_id.split("/")[-1])

compression_configuration = {
    "mode": nncf.CompressWeightsMode.NF4,
    "group_size": -1,
    "ratio": 1.0,
    "backup_mode": nncf.BackupMode.NONE,
}

convert_qwen2_5_omni_model(model_id, model_dir, compression_configuration)
