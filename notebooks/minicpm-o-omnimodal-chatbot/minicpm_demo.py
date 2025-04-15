import os
import threading
import pyaudio
import wave
import time
import hashlib
from PIL import ImageGrab, Image, ImageTk  
import webrtcvad
import pyperclip
import keyboard
import playsound
from miniCPM_model import MiniCPM
import asyncio
import torch
import pyaudio
import numpy as np
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import tkinter as tk
import sys
import cv2 

system_prompt = "You are an AI assistant, can answer questions based on user input,"
#system_prompt = "你是一个AI夸人助手。你能接受视频，音频和文本输入并输出简短的语音和文本，请用热情洋溢的赞美口吻，中文回答问题，最好是用夸人的预期赞美你看到的。"
start_image_path = "resource\\start.png"
stop_image_path = "resource\\stop.png"
camera_capture = "captured_data\\camera_capture.png"
video_frame_folder = "screenshots\\"

mimick_prompt = """If question contain specific requirements, you should answer stable formats:
 for example:
    if someone ask you to light up the screen, you need answer {lighten} only
    if someone ask you to light down the screen or make the screen darker, you need answer {darker} only.
For rest question, please answer the question with the short answer in one or two sentences"""

class MultiListener:
    def __init__(self, output_dir='captured_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize GUI
        self.root = None
        self.label = None
        self.start_img = None
        self.stop_img = None

        self.vad = webrtcvad.Vad()
        self.audio = pyaudio.PyAudio()
        self.stream = None
        # ASR parameter initialization
        self.sample_rate = 16000
        self.frame_duration = 8000
        self.channels = 1
        self.format = pyaudio.paInt16
        
        self.running = True
        self.last_text_hash = None
        self.last_image_hash = None
        self.recording = False
        self.audio_input = True
        self.clipboard_input = True
        self.video_frame_list = []


        self.cpmmodel = MiniCPM()
        self.cpmmodel.load_model()
        self.cpmmodel.prefill_model(system_prompt, "text")

        #Initialize VAD model
        self.vad = load_silero_vad()
        self.init_gui()

        #Initialize camera
        self.capture = cv2.VideoCapture(0)
        self.need_camera = True
        self.need_background = False

    def init_gui(self):
        """初始化图片显示窗口"""
        self.root = tk.Tk()
        self.root.title("状态指示器")
        self.root.overrideredirect(True)  # no boarder
        
        # Load image (300*300)
        self.start_img = ImageTk.PhotoImage(Image.open(start_image_path))
        self.stop_img = ImageTk.PhotoImage(Image.open(stop_image_path))
        
        # set window position (right-bottom)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_size = 300
        x = screen_width - window_size - 10  # reserve 10 pixels
        y = screen_height - window_size - 10
        self.root.geometry(f"{window_size}x{window_size}+{x}+{y}")
        
        # Show picture's label
        self.label = tk.Label(self.root, image=self.stop_img)
        self.label.pack()
        
        # Set window to the top
        self.root.attributes('-topmost', True)
        self.root.update()

    def update_status_image(self, is_running):
        # update image
        new_image = self.start_img if is_running else self.stop_img
        self.label.config(image=new_image)
        self.root.update()

    def hide_image(self):
        self.root.withdraw()

    def resume_image(self):
        self.root.deiconify()

    def calculate_hash(self, content):
        return hashlib.md5(str(content).encode()).hexdigest() if content else None

    def audio_listener(self):
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_duration
        )

        print("Start to monitor human voice...")
        bg_buffer = []
        frames = []
        is_speech = 0
        is_background = 0
        try:
            while True:
                if self.running and self.audio_input:
                    # Read audio data from microphone
                    data = self.stream.read(self.frame_duration, exception_on_overflow=False)
                    audio_frame = np.frombuffer(data, dtype=np.int16).astype(np.float32)

                    # Check whether audio frame contains speech
                    speech_prob = get_speech_timestamps(audio_frame, self.vad, sampling_rate = self.sample_rate, return_seconds = False)
                    if len(speech_prob) > 0:  
                        print("detected human voice")
                        frames.append(data)
                        is_speech = is_speech + 1
                        is_background = 0
                    else:
                        if is_speech > 1: # if only one audio frame, it maybe some unexpected noise.
                            # Save audio frame with the first silent frame after speech.
                            frames.append(data)
                            wav_path = os.path.join(self.output_dir, f'speech_{int(time.time())}.wav')
                            with wave.open(wav_path, 'wb') as wf:
                                wf.setnchannels(self.channels)
                                wf.setsampwidth(self.audio.get_sample_size(self.format))
                                wf.setframerate(self.sample_rate)
                                wf.writeframes(b''.join(frames))

                            if self.need_camera:
                                for _ in range(3):
                                    self.capture.read()
                                ret, vframe = self.capture.read()
                                # resized_frame = cv2.resize(vframe, (640, 480), interpolation=cv2.INTER_LINEAR)
                                cv2.imwrite(camera_capture, vframe)
                                self.cpmmodel.prefill_model(camera_capture, "image")

                            # self.cpmmodel.prefill_model(wav_path, "audio", mimick_prompt)
                            self.cpmmodel.prefill_model(wav_path, "audio")
                            self.need_camera = True
                            # hold on ASR to avoid capturing last TTS output sentence

                            self.clipboard_input = False # suspend clipboard listening
                            sleep_time = self.cpmmodel.query_model()
                            time.sleep(sleep_time)
                            # print(f"Finished Last sentence sending with {sleep_time}s")
                            # Reset model to avoid large context and make model confused
                            self.cpmmodel.reset_model()
                            self.cpmmodel.prefill_model(system_prompt, "text")
                            self.clipboard_input = True # restart clipboard listening

                            # Flush out both unexpected noise and prefilled speech
                            frames = []
                        else:
                            if self.need_background:
                                is_background += 1
                                bg_buffer.append(data)
                                if is_background > 3:
                                    bg_wav_path = os.path.join(self.output_dir, f'background_{int(time.time())}.wav')
                                    with wave.open(bg_wav_path, 'wb') as wf:
                                        wf.setnchannels(self.channels)
                                        wf.setsampwidth(self.audio.get_sample_size(self.format))
                                        wf.setframerate(self.sample_rate)
                                        wf.writeframes(b''.join(bg_buffer))
                                    self.cpmmodel.prefill_model(bg_wav_path, "audio")
                                    is_background = 0
                                    bg_buffer = []
                            # save the last audio frame before detecting speech
                            frames = []
                            frames.append(data)
                        is_speech = 0
                else:
                    time.sleep(0.1)

        except Exception as e:
            print(f"Audio listening error: {e}")
            exit()

    def clipboard_listener(self):
        # Not prefill existing clipboard before listening
        self.last_text_hash = self.calculate_hash(pyperclip.paste())
        last_image = ImageGrab.grabclipboard()
        if last_image and not isinstance(last_image, list):
            # print(last_image)
            self.last_image_hash = self.calculate_hash(last_image.tobytes())

        while True:
            try:
                if self.running and self.clipboard_input:
                    # Check text clip board first
                    current_text = pyperclip.paste()
                    current_text_hash = self.calculate_hash(current_text)               
                    if current_text and current_text_hash != self.last_text_hash:
                        # print(f"current text {current_text}")
                        self.cpmmodel.prefill_model(current_text, "text")
                        # print("Text clip board prefilled")
                        self.last_text_hash = current_text_hash
                        self.need_camera = False

                    # Check image clip board
                    image = ImageGrab.grabclipboard()
                    if image and not isinstance(image, list):
                        image_hash = self.calculate_hash(image.tobytes())
                        if image_hash != self.last_image_hash:
                            # print(f"{image_hash} vs. {self.last_image_hash}")
                            # Save image and prefill   
                            img_path = os.path.join(self.output_dir, f'image_{int(time.time())}.png')
                            image.save(img_path)
                            self.cpmmodel.prefill_model(img_path, "image")
                            print(f"Image saved: {img_path}")

                            #update last image hash, wait for next 
                            self.last_image_hash = image_hash
                            print("finished copy image") 
                            self.need_camera = False                          
                time.sleep(0.1)
            
            except Exception as e:
                print(f"Clipboard listening error: {e}")

    def screenshot_listener(self):
        frame_index = 0
        while True:
            if self.running and self.recording:
                screen_shot = ImageGrab.grab()
                target_size = (960, 540)
                resized_screenshot = screen_shot.resize(target_size, Image.Resampling.LANCZOS)
                file_name = video_frame_folder + "screenshot_" + "{:04d}".format(frame_index) + ".jpg"
                resized_screenshot.save(file_name, "PNG")
                self.video_frame_list.append(file_name)
                frame_index = frame_index + 1
                time.sleep(1)
            else:
                time.sleep(0.1)
                frame_index = 0

    def start(self):
        print("Start listening")
        self.update_status_image(True) 
        audio_thread = threading.Thread(target=self.audio_listener)
        clipboard_thread = threading.Thread(target=self.clipboard_listener)
        screenshot_thread = threading.Thread(target=self.screenshot_listener)
        
        audio_thread.start()
        clipboard_thread.start()
        screenshot_thread.start()

        keyboard.add_hotkey('esc', self.stop)
        keyboard.add_hotkey('L', self.suspend)
        keyboard.add_hotkey('V', self.record)
        
        try:
            self.root.mainloop()  
            audio_thread.join()
            clipboard_thread.join()
            screenshot_thread.join()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        print("程序正在终止...")
        self.running = False
        self.root.quit()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        keyboard.unhook_all()
        self.capture.release()
        os._exit(0)
    
    def suspend(self):
        self.running = not self.running
        self.root.after(0, lambda: self.update_status_image(self.running))

    def record(self):
        if self.running:
            if not self.recording:
                # self.close_status_image()
                self.audio_input = False
                self.recording = True
                self.hide_image()
            else:
                self.recording = False
                if self.video_frame_list:
                    self.cpmmodel.prefill_model(self.video_frame_list, "video")
                    self.video_frame_list = []
                    self.need_camera = False
                self.resume_image()
                self.audio_input = True
            # self.init_gui()

def main():
    listener = MultiListener()
    os.system("cls")
    keyboard.wait ("L")
    listener.start()

if __name__ == '__main__':
    main()