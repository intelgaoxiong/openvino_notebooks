from pathlib import Path
from minicpm_o_helper import convert_minicpmo26

model_id = "openbmb/MiniCPM-o-2_6"

model_dir = convert_minicpmo26(model_id)

from minicpm_o_helper import compression_widget

to_compress_weights = compression_widget()

to_compress_weights

import nncf
import gc
import openvino as ov

from minicpm_o_helper import llm_path, copy_llm_files


compression_configuration = {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 1.0, "all_layers": True}


core = ov.Core()
llm_int4_path = Path("language_model_int4") / llm_path.name
if to_compress_weights.value and not (model_dir / llm_int4_path).exists():
    ov_model = core.read_model(model_dir / llm_path)
    ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
    ov.save_model(ov_compressed_model, model_dir / llm_int4_path)
    del ov_compressed_model
    del ov_model
    gc.collect()
    copy_llm_files(model_dir, llm_int4_path.parent)