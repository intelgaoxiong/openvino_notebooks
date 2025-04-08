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

ov_model = init_model(model_dir, llm_int4_path.parent, "NPU")
tokenizer = ov_model.processor.tokenizer

from decord import VideoReader, cpu
from PIL import Image

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number
def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames
video_path = "MiniCPM-o-2_6/ckpt/assets/Skiing.mp4"
frames = encode_video(video_path)
question = "Describe the video"
msgs = [
    {'role': 'user', 'content': frames + [question]}, 
]
# Set decode params for video
params={}
params["use_image_id"] = False
params["max_slice_nums"] = 1 # use 1 if cuda OOM and video resolution >  448*448

# Warm up
ov_model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    **params
)

# Chat
answer_generator = ov_model.chat(
    msgs=msgs,
    stream=True,
    tokenizer=tokenizer,
    **params
)

# Iterate over the generator to process each piece of data
for partial_answer in answer_generator:
    # Print each piece of data without a newline
    print(partial_answer, end='', flush=True)

# Optionally, print a newline at the end to ensure the prompt returns to the next line
print()

import numpy as np

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

durations = ov_model.llm.llm_times[1:]
tpot_mean, tpot_std = calc_mean_and_std(durations)
throughput_mean, throughput_std = [1000.0 / tpot_mean, (tpot_std * 1000.0) / (tpot_mean * tpot_mean)]
ttft_mean, ttft_std = calc_mean_and_std([ov_model.llm.llm_times[0]])
print(f"LLM compilation time: {ov_model.llm.llm_compilation_time} s")
print(f"TTFT: {ttft_mean} ± {ttft_std} ms/token")
print(f"TPOT: {tpot_mean} ± {tpot_std} ms/token")
print(f"Throughput: {throughput_mean} ± {throughput_std} tokens/s")