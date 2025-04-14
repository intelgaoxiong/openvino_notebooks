
import torch
import os
from PIL import Image
import shutil
# from transformers import AutoModel, AutoTokenizer
import intel_extension_for_pytorch as ipex
from neural_compressor.transformers import AutoModel, AutoModelForCausalLM, RtnConfig
from transformers import  AutoTokenizer, StoppingCriteria
import time
import queue
# from screen_shot import screen_shot
import multiprocessing
import cv2
import numpy as np
from decord import VideoReader, cpu, VideoLoader, ndarray
import soundfile as sf
from dataclasses import dataclass
from typing import Optional
import time
import librosa
import requests

# load omni model default, the default init_vision/init_audio/init_tts is True
# if load vision-only model, please set init_audio=False and init_tts=False
# if load audio-only model, please set init_vision=False
MODEL_PATH = "MiniCPM-o-2_6-int4"
screen_shots_path = "screenshots\\"
screen_shot_time = 10
screen_shot_cadence = 0.5

url = "http://127.0.0.1:5000/receive_text"
url_headers = {"Content-Type": "application/json"}
welcome_words = "Hi, I am Intel AI assistant, it is my honor to talk with you。"

data_queue = queue.Queue()
MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number

class custom_stop(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_count = 0
        self.term_count = 0
        self.tokenizer = tokenizer

        self.stop_sign = "."
        self.stop_times_minus_one = 1

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = input_ids[0]
        comma_token_id = self.tokenizer.encode(self.stop_sign)[0]
        for token_id in generated_text:
            if token_id == comma_token_id:
                self.stop_count = self.stop_count + 1
            if self.stop_count > self.stop_times_minus_one:
                return True
        return False

    def contain_stop_sign(self, text):
        if self.stop_sign in text:
            return True
        return False


@dataclass
class StreamChunk:
    text: Optional[str] = None
    audio_wav: Optional[np.ndarray] = None
    sampling_rate: Optional[int] = None
    is_end: bool = False

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

def create_video_from_images(image_list, output_video_path):
    for i in range(0, len(image_list)):
        new_filename = f"encode_{i+1:03d}.png"
        shutil.copy(image_list[i], new_filename)

    ffmpeg_command = "ffmpeg -i encode_%03d.png -y -vcodec libx264 "+output_video_path
    os.system(ffmpeg_command)

def encode_from_images(file_list):
    file_name = "screenshot.mp4"
    create_video_from_images(file_list, file_name)
    vr = VideoReader(file_name, ctx=cpu(0))
    frames = [frame.asnumpy() for frame in vr]
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

# put the image file under folder into one list
def get_file_name_list(folder_path, time, cadence):
    image_count = int(time / cadence)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # sorted with time
    sorted_files = sorted(files, key=os.path.getctime)

    return sorted_files[(len(sorted_files)-image_count):]

# read image file name list, encode as model input format
def read_from_images(file_list):
    frames = []

    for img_file in file_list:
        img = cv2.imread(img_file)  # 读取 PNG 图像
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式（符合 Decord 要求）
        frames.append(Image.fromarray(img_rgb.astype('uint8'))) # 将图像数据转为 Decord ndarray

    return frames

class MiniCPM:

    def __init__(self):
        self.device="xpu"
        self.generate_audio = False  # miniCPM audio output performance is not enough, use text output + TTS
        self.session_id = '123'

    def load_model(self):
        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            init_vision=True,
            init_audio=True,
            init_tts=True
        )

        self.model=self.model.to(self.device)
        print(self.model)
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, torch_dtype=torch.float16,trust_remote_code=True)

        # # In addition to vision-only mode, tts processor and vocos also needs to be initialized
        self.model.init_tts()
        self.model.tts.float()

        self.stop_criteria = custom_stop(self.tokenizer)

        #Warm up session
        video_path="screenshot.mp4"#"bbb_sunflower_2160p_60fps_normal.mp4"
        frames = encode_video(video_path)
        question = "请用一句话回答下面的问题。 问题：表述这个视频"
        msgs = [
            {'role': 'user', 'content': frames + [question]}, 
        ]

        # Set decode params for video
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 5 # use 1 if cuda OOM and video resolution > 448*448  Tuning for quality & perf balance
        
        torch.xpu.synchronize()
        st=time.time()
        answer = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            **params
        )
        torch.xpu.synchronize()
        et=time.time()
        output_len= self.tokenizer(answer, return_tensors="pt").input_ids.shape[-1]
        print("warmup time cost is ,output_len",et-st, output_len)
        # print(answer)
        self.model.reset_session()

        payload = {'text': welcome_words}
        print("")
        requests.post(url, json=payload, headers=url_headers)

    # Streaming mode prefill, support text, single image, audio
    def prefill_model(self, message, type, system_prompt = None):
        prompt = ""
        prompt_msg = []
        if type == "text":
            print("\033[34m*****************Prefill text prompt*****************\033[0m")
            prompt = message
            prompt_msg = [{"role":"user", "content": prompt}]
        elif type == "image":
            print("\033[34m*****************Prefill image prompt*****************\033[0m")
            # reuse image list encoding function, create the list contains only one image. 
            messages = [message,]
            prompt = read_from_images(messages)
            prompt_msg = [{"role":"user", "content": prompt}]
        elif type == "video":
            print(f"\033[34m*****************Prefill {len(message)} seconds video prompt*****************\033[0m")
            prompt = read_from_images(message)
            prompt_msg = [{"role":"user", "content": prompt}]
        elif type == "audio":
            print("\033[34m*****************Prefill audio prompt*****************\033[0m")
            #system_prompt = "请用一句话，以赞美的诗意风格回答下面的问题："
            #system_prompt = "You are an AI assistant, can answer questions based on user input, for example, if you heard some keywords about *light up screen*, you will answer with format: {lighten}"
            # system_prompt = "You are an AI assistant, can answer questions based on user input, for example, if someone ask you to light up the screen, you need answer {lighten} only"
            prompt, _ = librosa.load(message, sr=16000, mono=True)
            prompt_msg = [{"role":"user", "content": [system_prompt, prompt]}]

        res = self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=prompt_msg, 
            tokenizer=self.tokenizer
        )
        # print("finished prefill")
        return res

    # submit question and send text output to TTS model in streaming mode
    def query_model(self):
        res = self.model.streaming_generate(
            session_id=self.session_id,
            tokenizer=self.tokenizer,
            temperature=0.3,
            sampling=True,
            generate_audio=self.generate_audio,
            stopping_criteria=[self.stop_criteria]
        )

        text = ""
        # default sleep time (about one word speaking duration)
        sleep_time = 0.3
        if self.generate_audio:
            try:
                for r in res:
                    # Create and send chunk with both audio and text
                    chunk = StreamChunk(
                        text=r.text,
                        audio_wav=r.audio_wav,
                        sampling_rate=r.sampling_rate
                    )
                    data_queue.put(chunk)
                    
                # Signal completion
                data_queue.put(StreamChunk(is_end=True))
                
            except Exception as e:
                print(f"Error in processing: {e}")
                # Make sure to signal completion even on error
                data_queue.put(StreamChunk(is_end=True))
        else:
            for r in res:
                text = r['text']
                print(text, end="", flush=True)
                payload = {'text': text}
                # Take model streaming output and send through http directly
                response = requests.post(url, json=payload, headers=url_headers)
                # find last TTS output feedback, calculate sleep time to avoid TTS output captured by ASR
                if "<|tts_eos|>" in text:
                    print("\n", end="")
                    sleep_time = int(response.json()['length']) / 5  # Estimate 3 words per second.
                    break
        
        return sleep_time

    def reset_model(self):
        self.model.reset_session()


# model = MiniCPM()
# model.load_model()
# # system message
# model.prefill_model("你是一个AI助手。你能接受视频，音频和文本输入并输出简短的语音和文本。", "text")

# # prompt stream
# # print(f"Before image prefill: {time.time()}")
# model.prefill_model("bilibili.png", "image")
# # torch.xpu.synchronize()
# # ct = time.time()
# # print(f"finish asking time: {ct}")

# model.prefill_model("record.wav", "audio")
# # torch.xpu.synchronize()
# # print(f"after prefill {time.time()}")

# #model.prefill_model("描述这张图片", "text")
# model.query_model()

# if __name__ == '__main__':
#     # multiprocessing.freeze_support()
#     # process = multiprocessing.Process(target=screen_shot, args=(screen_shots_path, screen_shot_time, screen_shot_cadence))
#     # process.start()
#     print("*********************************************************")
#     # app.run(debug=False, threaded=True)
