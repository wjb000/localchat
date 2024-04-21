from transformers import pipeline
import sounddevice as sd
import numpy as np
from llama_cpp import Llama
import subprocess
import threading
import soundfile as sf
import os

whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

# Load the GGML model using llama_cpp for generating responses
model_path = "path/to/model.gguf"
llm = Llama(model_path=model_path)

system_message = "only respond to User: statements with statements that start with AI:, You are an intelligent sentient AI who gives useful responce."


def record_audio(duration=5, samplerate=16000):
    """Record audio from the microphone."""
    print("Recording...")
    audio = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32"
    )
    sd.wait()
    print("Recording stopped.")
    return audio


def transcribe_audio(audio):
    """Transcribe audio to text using Whisper."""
    temp_audio_path = "temp_audio.wav"
    sf.write(temp_audio_path, audio, 16000)
    transcription = whisper(temp_audio_path)
    os.remove(temp_audio_path)
    return transcription["text"]


def generate_response(user_input):
    """Generate a response from the model."""
    prompt = f"{system_message}\nUser: {user_input}\nAI:"
    response = llm(prompt, max_tokens=100)
    return response["choices"][0]["text"].strip()


def synthesize_speech(text, voice="Karen"):
    """Use subprocess to call 'say' command for MacOS with a specific voice."""
    subprocess.run(["say", "-v", voice, text])


def wait_for_activation(activation_phrase="Activate."):
    while True:
        audio = record_audio()
        user_input = transcribe_audio(audio)
        print("Transcribed text:", user_input)
        if activation_phrase.lower() in user_input.lower():
            synthesize_speech("Activation phrase detected. Starting conversation.")
            break


def main():
    while True:
        print("Waiting for activation phrase...")
        wait_for_activation()
        print("Please speak into the microphone.")
        while True:
            audio = record_audio()
            print("Processing your voice...")
            user_input = transcribe_audio(audio)
            print("User:", user_input)

        
            user_input_lower = user_input.lower()
            exit_phrases = ["bye", "goodbye", "exit", "quit"]
            if any(phrase in user_input_lower for phrase in exit_phrases):
                print("Goodbye! Going back to dormant mode.")
                synthesize_speech("Goodbye! Going back to dormant mode.")
                break

            ai_response = generate_response(user_input)
            print("AI:", ai_response)
            synthesis_thread = threading.Thread(
                target=synthesize_speech, args=(ai_response,)
            )
            synthesis_thread.start()
            synthesis_thread.join()  # Wait for the TTS to complete before continuing


if __name__ == "__main__":
    main()
