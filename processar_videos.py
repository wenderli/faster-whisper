import os
import rarfile
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel

# Diretórios
BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
AUDIOS_DIR = BASE_DIR / "audios"
TRANSCRICOES_DIR = BASE_DIR / "transcricoes"

# Certifique-se de que os diretórios existem
for d in [VIDEOS_DIR, AUDIOS_DIR, TRANSCRICOES_DIR]:
    d.mkdir(exist_ok=True)

# 1. Descompactar arquivos .rar
def descompactar_rar(caminho_rar):
    with rarfile.RarFile(caminho_rar) as rf:
        rf.extractall(VIDEOS_DIR)
        print(f"✔ Descompactado: {caminho_rar.name}")

# 2. Extrair áudio de vídeo e dividir em partes de 30 minutos
def extrair_audio_em_partes(video_path, tempo_segmento=1800):
    nome_base = video_path.stem
    output_pattern = AUDIOS_DIR / f"{nome_base}_%03d.wav"

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-f", "segment",
        "-segment_time", str(tempo_segmento),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        str(output_pattern)
    ]
    subprocess.run(cmd, check=True)
    print(f"✔ Áudio extraído e dividido: {video_path.name}")

# 3. Transcrever áudios WAV
def transcrever_audios():
    model = WhisperModel("large-v2", device="cuda", compute_type="float16")

    for wav in sorted(AUDIOS_DIR.glob("*.wav")):
        print(f"⏳ Transcrevendo {wav.name}...")
        segments, info = model.transcribe(str(wav))

        txt_path = TRANSCRICOES_DIR / f"{wav.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Idioma detectado: {info.language}\n\n")
            for seg in segments:
                f.write(f"[{seg.start:.2f} -> {seg.end:.2f}] {seg.text}\n")

        print(f"✔ Transcrição salva: {txt_path.name}")

# 4. Pipeline principal
def processar_videos():
    for rar in sorted(VIDEOS_DIR.glob("*.rar")):
        descompactar_rar(rar)

    for mkv in sorted(VIDEOS_DIR.glob("*.mkv")):
        extrair_audio_em_partes(mkv)

    transcrever_audios()

if __name__ == "__main__":
    processar_videos()
