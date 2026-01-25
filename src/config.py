import os
from dotenv import load_dotenv

# Carrega o .env da raiz
load_dotenv()

class Config:
    # Saída
    OUTPUT_SUFFIX = os.getenv("OUTPUT_SUFFIX", "_DUB_PRO")
    AUDIO_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "44100"))
    
    # Whisper
    WHISPER_MODEL = os.getenv("WHISPER_MODEL_SIZE", "medium")
    WHISPER_LANG = os.getenv("WHISPER_SOURCE_LANG", "en")
    WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    WHISPER_BEAM = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    
    # Tradução
    TRANS_MODEL = os.getenv("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-tc-big-en-pt")
    TRANS_BATCH = int(os.getenv("TRANSLATION_BATCH_SIZE", "32"))
    
    # TTS
    TTS_VOICE = os.getenv("TTS_VOICE", "pt-BR-AntonioNeural")
    TTS_WORKERS = int(os.getenv("TTS_MAX_WORKERS", "5"))
    
    # Ducking
    DUCK_THRESH = os.getenv("DUCKING_THRESHOLD", "0.05")
    DUCK_RATIO = os.getenv("DUCKING_RATIO", "5")
    DUCK_ATTACK = os.getenv("DUCKING_ATTACK", "20")
    DUCK_RELEASE = os.getenv("DUCKING_RELEASE", "200")