import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

def record_audio(filename, duration=5, fs=16000):
    """Records audio for a specified duration and saves it as a WAV file."""
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    write(filename, fs, recording)
    print(f"Audio recorded and saved as {filename}")

def transcribe_audio(filename):
    """Uses Whisper model to transcribe audio from a file."""
    model = whisper.load_model("base")
    try:
        result = model.transcribe(filename)
    except RuntimeError as e:
        print(f"Failed to transcribe audio: {e}")
        return None
    return result["text"]

def save_transcription_to_file(transcription, output_file="text.txt"):
    """Saves the transcribed text to a file."""
    if transcription:
        with open(output_file, "w") as f:
            f.write(transcription)
        print(f"Transcription saved to {output_file}")
    else:
        print("No transcription available to save.")

def text_to_llm(input_file='text.txt'):
    """Generates a response using a language model."""
    model_name = "crumb/nano-mistral"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1
    model.to(device)

    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    with open(input_file, 'r') as file:
        prompt = file.read().strip()

    if not prompt:
        raise ValueError("The input prompt is empty. Please provide valid input text.")

    response = text_generator(
        prompt,
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    generated_text = response[0]['generated_text'].strip()

    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    def is_meaningful(response):
        if len(response.split()) < 2:
            return False
        if response.lower() == prompt.lower():
            return False
        return True

    attempts = 3
    while not is_meaningful(generated_text) and attempts > 2:
        response = text_generator(
            prompt,
            max_length=200,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        generated_text = response[0]['generated_text'].strip()
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        attempts -= 1

    if is_meaningful(generated_text):
        with open('response.txt', 'w') as file:
            file.write(generated_text)
        print("Response has been written to response.txt")
    else:
        print("Failed to generate a meaningful response after multiple attempts.")

def read_text_from_file(file_path):
    """Reads and returns text from the specified file."""
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def text_to_speech_and_play(text, language='en', slow=False, audio_file='output.mp3'):
    """Converts text to speech using gTTS and plays the audio."""
    print(f"Text to be spoken:\n{text}\n")

    tts = gTTS(text=text, lang=language, slow=slow)
    tts.save(audio_file)
    print(f"Audio saved to {audio_file}")

    # Load and play audio file
    try:
        audio = AudioSegment.from_mp3(audio_file)
        play(audio)
        print("Playing audio...")
    except Exception as e:
        print(f"An error occurred while playing audio: {e}")

    # Remove the audio file after playing
    if os.path.exists(audio_file):
        os.remove(audio_file)
        print("Temporary audio file removed.")

def main():
    print("Choose an option:")
    print("1. Record audio")
    print("2. Upload an audio file")
    print("3. Use an existing audio file in the project directory")
    choice = input("Enter the option number: ")

    if choice == "1":
        filename = "recorded_audio.wav"
        duration = int(input("Enter the duration of the recording in seconds: "))
        record_audio(filename, duration)
    elif choice == "2":
        filename = input("Enter the path to the audio file (including extension): ")
        if not os.path.isfile(filename):
            print("File not found. Exiting.")
            return
    elif choice == "3":
        filename = input("Enter the name of the audio file in the project directory (including extension): ")
        file_path = os.path.join(os.getcwd(), filename)
        if not os.path.isfile(file_path):
            print("File not found in the project directory. Exiting.")
            return
        filename = file_path
    else:
        print("Invalid option. Exiting.")
        return

    print("Transcribing audio...")
    transcript = transcribe_audio(filename)
    if transcript:
        print("Transcript:", transcript)
        save_transcription_to_file(transcript)
    else:
        print("No transcript generated.")
        return

    print("Generating text...")
    text_to_llm()

    print("Converting text to speech...")
    text_response = read_text_from_file('response.txt')
    text_to_speech_and_play(text_response)

if __name__ == "__main__":
    main()
