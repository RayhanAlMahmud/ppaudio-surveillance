import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # always use local HF cache

import json, numpy as np, librosa, torch, argparse
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
PROBES = BASE / "data" / "probes"

encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
     savedir=str(ARTIFACTS / "ecapa-voxceleb"),
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

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def main(probe_wav, threshold=0.55):
    enrolled = json.load(open(ARTIFACTS / "suspects_embeddings.json"))
    probe_emb = wav_to_embedding(probe_wav)
    probe_emb = probe_emb / (np.linalg.norm(probe_emb) + 1e-9)
    best_name, best_score = None, -1.0
    for spk, emb in enrolled.items():
        s = cosine(probe_emb, np.array(emb))
        if s > best_score:
            best_name, best_score = spk, s
    verdict = best_name if best_score >= threshold else "NO MATCH"
    print(f"Best={best_name}  Score={best_score:.3f}  Verdict={verdict}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True, help="Path to probe wav")
    ap.add_argument("--th", type=float, default=0.55, help="Cosine threshold")
    args = ap.parse_args()
    main(args.wav, args.th)
