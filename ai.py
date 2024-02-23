import speech_recognition as sr
from openai import OpenAI
import torch 
import sounddevice as sd 
import time 
import speech_recognition as sr


# создаем экземпляр класса Recognizer
recognizer = sr.Recognizer()

# записываем звук с микрофона
with sr.Microphone() as source:
    print("Скажите что-нибудь: ")
    audio_data = recognizer.listen(source)

# распознаем речь с помощью Google Speech Recognition
try:
    text = recognizer.recognize_google(audio_data, language="ru-RU")
    print("Вы сказали: " + text)

    client = OpenAI(api_key="Key", base_url="https://api.deepseek.com/v1")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": text},
            {"role": "user", "content": text},
        ]
    )

    print(response.choices[0].message.content)

    language = 'ru'
    model_id = 'ru_v3'
    sample_rate = 48000
    speaker = 'baya' #aidar, baya, xenia, random
    put_accent = True 
    put_yo = True
    device = torch.device('cpu') 
    text = (response.choices[0].message.content)

    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', 
                            model='silero_tts', 
                            language=language, 
                            speaker=model_id)

    model.to(device)

    audio = model.apply_tts(text=text,
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent, 
                            put_yo=put_yo)

    print(text)

    def play():
        sd.play(audio, sample_rate)
        time.sleep(len(audio) / sample_rate)
        sd.stop()
    play()
except sr.UnknownValueError:
    print("Извините, не удалось распознать речь")
except sr.RequestError as e:
    print("Ошибка сервиса распознавания речи; {0}".format(e))