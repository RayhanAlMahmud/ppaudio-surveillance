

# enroll.py  (dual backend: cosine | plda)
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # always use local cache
os.environ.setdefault("SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY", "copy")  # avoid symlinks on Windows

import argparse
import json
from pathlib import Path
import torchaudio, soundfile
import librosa
import numpy as np
import torch, speechbrain
from speechbrain.inference import EncoderClassifier
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)

print("librosa OK:", hasattr(librosa, "load"))
print("speechbrain OK:", hasattr(speechbrain, "__version__"))

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "suspects"
ART = BASE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

encoder = EncoderClassifier.from_hparams(
      source=str((ART if 'ART' in globals() else (Path(__file__).resolve().parent / "artifacts")) / "ecapa-voxceleb"),
    savedir=str((ART if 'ART' in globals() else (Path(__file__).resolve().parent / "artifacts")) / "ecapa-voxceleb"),
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

print("Model dir:", ART / "ecapa-voxceleb")
print("HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"))
print("SB_STRATEGY    =", os.environ.get("SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY"))


def load_audio_16k(p: Path, target_sr: int = 16000) -> np.ndarray:
    y, sr = librosa.load(p, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y

def wav_to_emb(p: Path) -> np.ndarray:
    y = load_audio_16k(p)
    sig = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        emb = encoder.encode_batch(sig).squeeze(0).squeeze(0).cpu().numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb

def main(backend: str) -> None:
    if not DATA.exists():
        print(f"[ERR] Missing folder: {DATA}")
        return

    # Optional PLDA projection params
    if backend == "plda":
        bk_path = ART / "plda_backend.npz"
        if not bk_path.exists():
            print(f"[ERR] {bk_path} not found. Run: python train_plda_simplified.py")
            return
        bk = np.load(bk_path)
        gmean = bk["mean"]
        V = bk["V"]

    out = {}
    for spk_dir in sorted([d for d in DATA.iterdir() if d.is_dir()]):
        wavs = sorted(spk_dir.glob("*.wav"))
        if not wavs:
            print(f"[WARN] No wavs in {spk_dir.name}, skipping.")
            continue

        embs = [wav_to_emb(w) for w in wavs]
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)

        if backend == "cosine":
            z = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
            out[spk_dir.name] = z.tolist()
            print(f"[OK] {spk_dir.name}: {len(wavs)} files -> enrolled (cosine).")
        else:
            # PLDA space: u = V^T (x - global_mean), then L2 normalize
            u = V.T @ (mean_emb - gmean)
            u = u / (np.linalg.norm(u) + 1e-9)
            out[spk_dir.name] = u.tolist()
            print(f"[OK] {spk_dir.name}: {len(wavs)} files -> enrolled (PLDA).")

    suffix = "cosine" if backend == "cosine" else "plda"
    out_path = ART / f"suspects_embeddings.{suffix}.json"
    json.dump(out, open(out_path, "w"))
    print(f"Saved {len(out)} speakers to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cosine", "plda"], default="cosine")
    args = ap.parse_args()
    main(args.backend)
