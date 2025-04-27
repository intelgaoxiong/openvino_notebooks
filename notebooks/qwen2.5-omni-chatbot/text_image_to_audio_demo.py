import requests
from pathlib import Path
from notebook_utils import device_widget
from qwen2_5_omni_helper import OVQwen2_5OmniModel
from transformers import Qwen2_5OmniProcessor

model_id = "Qwen/Qwen2.5-Omni-7B"
model_dir = Path(model_id.split("/")[-1])

thinker_device = device_widget(default="GPU", exclude=["NPU"], description="Thinker device")
talker_device = device_widget(default="GPU", exclude=["NPU"], description="Talker device")
token2wav_device = device_widget(default="CPU", exclude=["NPU"], description="Token2Wav device")

ov_model = OVQwen2_5OmniModel(model_dir, thinker_device=thinker_device.value, talker_device=talker_device.value, token2wav_device=token2wav_device.value)
processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)

#######################################################

from qwen_omni_utils import process_mm_info
import soundfile as sf
import IPython
from transformers import TextStreamer
from PIL import Image
from io import BytesIO
from notebook_utils import download_file
from IPython.display import display

# Demo: text-image input and Audio output

image_path = Path("cat.png")

if not image_path.exists():
    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    image = Image.open(BytesIO(requests.get(url).content))
    image.save(image_path)
else:
    image = Image.open(image_path)

print("Question:\nWhat is unusual on this picture?")
display(image)
print("Answer:")

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "cat.png"},
            {"type": "text", "text": "What is unusual on this picture?"},
        ],
    },
]

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
inputs = processor(text=text, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
text_ids, audio = ov_model.generate(
    **inputs, stream_config=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True), return_audio=True, thinker_max_new_tokens=256
)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)

display(IPython.display.Audio("output.wav"))
