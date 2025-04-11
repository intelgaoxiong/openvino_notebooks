import os
from pathlib import Path
from notebook_utils import device_widget

device = device_widget(default="AUTO", exclude=["NPU"])

device

from minicpm_o_helper import llm_path, lm_variant_selector, init_model

model_dir = Path("MiniCPM-o-2_6")

llm_int4_path = Path("language_model_int4") / llm_path.name

use_int4_lang_model = lm_variant_selector(model_dir / llm_int4_path)

use_int4_lang_model

print("use_int4_lang_model:", use_int4_lang_model.value)

print("device:", device.value)

#Use case requires larger LLM token length
ov_model = init_model(model_dir, llm_int4_path.parent, "NPU", 2048, 128)
tokenizer = ov_model.processor.tokenizer

import math
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa


def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print("video_duration:", video.duration)

    temp_audio_file_path = os.path.join(os.path.expanduser("~"), "Desktop", "temp.wav")
    video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
    audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

    # 1 frame + 1s audio chunk
    contents = []
    for i in range(num_units):
        frame = video.get_frame(i + 1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr * i : sr * (i + 1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])

    return contents


video_path = "MiniCPM-o-2_6/ckpt/assets/Skiing.mp4"
# if use voice clone prompt, please set ref_audio
ref_audio_path = "MiniCPM-o-2_6/ckpt/assets/demo.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
sys_msg = ov_model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")
# or use default prompt
# sys_msg = model.get_sys_prompt(mode='omni', language='en')

contents = get_video_chunk_content(video_path)
msg = {"role": "user", "content": contents}
msgs = [sys_msg, msg]

# please set generate_audio=True and output_audio_path to save the tts result
generate_audio = False
output_audio_path = "output.wav"

res = ov_model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.5,
    max_new_tokens=4096,
    omni_input=True,  # please set omni_input=True when omni inference
    use_tts_template=True,
    generate_audio=generate_audio,
    output_audio_path=output_audio_path,
    max_slice_nums=1,
    use_image_id=False,
    return_dict=True,
)
print(res)

def calc_mean_and_std(durations):
    if len(durations) == 0:
        return -1, -1

    # Convert durations to milliseconds
    durations_ms = np.array(durations) * 1000.0

    # Calculate mean
    mean = np.mean(durations_ms)

    # Calculate standard deviation
    std = np.std(durations_ms)

    return mean, std

vpm_mean, vpm_std = calc_mean_and_std(ov_model.vpm_times)
resampler_mean, resampler_std = calc_mean_and_std(ov_model.resampler_times)

durations = ov_model.llm.llm_times[1:]
tpot_mean, tpot_std = calc_mean_and_std(durations)
throughput_mean, throughput_std = [1000.0 / tpot_mean, (tpot_std * 1000.0) / (tpot_mean * tpot_mean)]
ttft_mean, ttft_std = calc_mean_and_std([ov_model.llm.llm_times[0]])
print(f"LLM compilation time: {ov_model.llm.llm_compilation_time} s")
print(f"TTFT: {ttft_mean} ± {ttft_std} ms/token")
print(f"TPOT: {tpot_mean} ± {tpot_std} ms/token")
print(f"Throughput: {throughput_mean} ± {throughput_std} tokens/s")

print(f"vision encoder throughput: {1000.0 / vpm_mean} ± {(vpm_std * 1000.0) / (vpm_mean * vpm_mean)}fps")
print(f"Resampling throughput: {1000.0 / resampler_mean} ± {(resampler_std * 1000.0) / (resampler_mean * resampler_mean)}fps")
print(f"vLLM embedding time: {ov_model.vllm_emb_time}m for {len(ov_model.vpm_times)} images")

print(f"audio encode time: {ov_model.apm_time}m")
