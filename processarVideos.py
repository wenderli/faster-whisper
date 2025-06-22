import os
import subprocess
from pathlib import Path
import whisperx
from datetime import timedelta
import torch

# Diretórios
BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
AUDIOS_DIR = BASE_DIR / "audios"
TRANSCRICOES_DIR = BASE_DIR / "transcricoes"

for d in [VIDEOS_DIR, AUDIOS_DIR, TRANSCRICOES_DIR]:
    d.mkdir(exist_ok=True)

# Função para converter segundos para hh:mm:ss
def formatar_tempo(segundos):
    return str(timedelta(seconds=int(segundos))).zfill(8)

# 1. Extrair áudio de vídeo e dividir em partes
def extrair_audio_em_partes(video_path, tempo_segmento=1800):
    nome_base = video_path.stem
    output_pattern = AUDIOS_DIR / f"{nome_base}_%03d.wav"

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-f", "segment",
        "-segment_time", str(tempo_segmento),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        str(output_pattern)
    ]
    subprocess.run(cmd, check=True)
    print(f"✔ Áudio extraído e dividido: {video_path.name}")

# 2. Transcrever e identificar falantes

def transcrever_audios():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("large-v2", device)

    for wav in sorted(AUDIOS_DIR.glob("*.wav")):
        print(f"⏳ Transcrevendo {wav.name}...")
        audio = whisperx.load_audio(str(wav))
        result = model.transcribe(audio, batch_size=16)

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        diarize_model = whisperx.DiarizationPipeline(use_auth_token=True, device=device)
        diarize_segments = diarize_model(str(wav))

        result_aligned = whisperx.merge_text_diarization(result_aligned, diarize_segments)

        txt_path = TRANSCRICOES_DIR / f"{wav.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Idioma detectado: {result['language']}\n\n")
            for seg in result_aligned["segments"]:
                start = formatar_tempo(seg['start'])
                end = formatar_tempo(seg['end'])
                speaker = seg.get("speaker", "SPEAKER_00")
                text = seg["text"].strip()
                f.write(f"[{start} -> {end}] {speaker}: {text}\n")

        print(f"✔ Transcrição salva: {txt_path.name}")

# 3. Pipeline principal
def processar_videos():
    for mkv in sorted(VIDEOS_DIR.glob("*.mkv")):
        extrair_audio_em_partes(mkv)

    transcrever_audios()

if __name__ == "__main__":
    processar_videos()