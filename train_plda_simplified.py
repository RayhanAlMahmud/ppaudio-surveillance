# train_plda_simplified.py
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # always use local HF cache
os.environ.setdefault("SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY", "copy")


import numpy as np, torch
from pathlib import Path
import torchaudio
from speechbrain.inference import EncoderClassifier
from scipy.linalg import eigh as geigh

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"; ART.mkdir(parents=True, exist_ok=True)
DATA_ROOT = BASE / "data" / "librispeech"

def load_utt(ds, idx, target_sr=16000):
    wav, sr, *_ = ds[idx]
    if wav.ndim > 1: wav = wav.mean(0)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.numpy()

def embed(wav, enc):
    sig = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        e = enc.encode_batch(sig).squeeze(0).squeeze(0).cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)

def compute_sw_sb(by_spk_vecs):
    allv = np.concatenate([np.stack(v,0) for v in by_spk_vecs.values()],0)
    m = allv.mean(0); d = m.size
    Sw = np.zeros((d,d)); Sb = np.zeros((d,d)); total = 0
    for vecs in by_spk_vecs.values():
        X = np.stack(vecs,0); mu = X.mean(0)
        Xc = X - mu; Sw += Xc.T @ Xc; total += X.shape[0]
        diff = (mu - m).reshape(-1,1); Sb += X.shape[0] * (diff @ diff.T)
    Sw /= total; Sb /= total
    return m, Sw, Sb

def main():
    ds = torchaudio.datasets.LIBRISPEECH(str(DATA_ROOT), url="test-clean", download=False)
    by_spk_idx = {}
    for i in range(len(ds)):
        _, _, _, spk, _, _ = ds[i]
        by_spk_idx.setdefault(spk, []).append(i)
    speakers = sorted(by_spk_idx.keys())[:20]         # ~20 spk
    enc = EncoderClassifier.from_hparams(ART / "ecapa-voxceleb",
     savedir=str(ART / "ecapa-voxceleb"),                          
     run_opts={"device":"cuda" if torch.cuda.is_available() else "cpu"})
    by_spk_vecs = {}
    for spk in speakers:
        idxs = by_spk_idx[spk][:10]                   # ~10 utt/spk
        by_spk_vecs[spk] = [embed(load_utt(ds,k), enc) for k in idxs]

    m, Sw, Sb = compute_sw_sb(by_spk_vecs)
    reg = 1e-3 * np.trace(Sw) / Sw.shape[0]
    w, V = geigh(Sb, Sw + reg*np.eye(Sw.shape[0]))    # generalized eig
    order = np.argsort(w)[::-1]
    psi = np.clip(w[order], 0.0, None).astype(np.float32)
    V = V[:, order].astype(np.float32)
    np.savez(ART/"plda_backend.npz", mean=m.astype(np.float32), V=V, psi=psi)
    print("Saved", ART/"plda_backend.npz")

if __name__ == "__main__": main()
