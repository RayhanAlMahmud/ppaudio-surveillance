
import json, numpy as np, librosa, torch
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier

BASE = Path(__file__).resolve().parent
SUSPECTS_DIR = BASE / "data" / "suspects"
ARTIFACTS = BASE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True, parents=True)

encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

def load_audio_16k(path, target_sr=16000):
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y

def wav_to_embedding(path):
    y = load_audio_16k(path)
    sig = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(sig).squeeze(0).squeeze(0).cpu().numpy()
    return emb

def main():
    speaker_embs = {}
    for speaker_dir in sorted(p for p in SUSPECTS_DIR.iterdir() if p.is_dir()):
        wavs = sorted(speaker_dir.glob("*.wav"))
        if not wavs:
            print(f"[WARN] No wavs in {speaker_dir.name}, skipping.")
            continue
        embs = [wav_to_embedding(w) for w in wavs]
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
        speaker_embs[speaker_dir.name] = mean_emb.tolist()
        print(f"[OK] {speaker_dir.name}: {len(embs)} files -> enrolled.")
    out = ARTIFACTS / "suspects_embeddings.json"
    json.dump(speaker_embs, open(out, "w"))
    print(f"Saved {len(speaker_embs)} to {out}")

if __name__ == "__main__":
    main()
