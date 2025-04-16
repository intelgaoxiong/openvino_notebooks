from pathlib import Path
from qwen2_5_omni_helper import convert_qwen2_5_omni_model

model_id = "Qwen/Qwen2.5-Omni-7B"
model_dir = Path(model_id.split("/")[-1])

import nncf

compression_configuration = {
    "mode": nncf.CompressWeightsMode.NF4,
    "group_size": -1,
    "ratio": 1.0,
}

convert_qwen2_5_omni_model(model_id, model_dir, compression_configuration)