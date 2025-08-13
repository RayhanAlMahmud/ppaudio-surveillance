import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # always use local cache
import json, random, argparse, requests
from pathlib import Path
import numpy as np
import librosa, torch
from speechbrain.inference import EncoderClassifier
from phe import paillier
import sys
import torch

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
# ----------------- audio / embeddings -----------------
# Load pretrained speaker encoder (ECAPA)
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

def load_audio_16k(path: Path, target_sr=16000):
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y

def wav_to_embedding(path: Path):
    y = load_audio_16k(path)
    sig = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(sig).squeeze(0).squeeze(0).cpu().numpy()
    # L2 normalize
    return emb / (np.linalg.norm(emb) + 1e-9)


# ----------------- server helpers -----------------
def get_pubkey(server_url: str):
    try:
        resp = requests.get(f"{server_url}/pubkey", timeout=10)
    except requests.RequestException as e:
        print(f"[ERR] Could not reach server {server_url}: {e}")
        sys.exit(1)
    ct = (resp.headers.get("content-type") or "").lower()
    if "application/json" not in ct:
        print(f"[ERR] /pubkey returned non-JSON ({resp.status_code}):")
        print(resp.text[:400])
        sys.exit(1)
    data = resp.json()
    return paillier.PaillierPublicKey(n=int(data["n"]))


# ----------------- scoring -----------------
def cosine(a: np.ndarray, b: np.ndarray):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

    
def plda_params(psi: np.ndarray, n: int):
    theta = 1.0 / (1.0 + psi)                    # Θ
    xi    = 1.0 / (1.0 + psi / (n * psi + 1.0))  # Ξ
    return theta, xi

def plda_score_float(u_probe: np.ndarray, ubar_g: np.ndarray, psi: np.ndarray, n: int):
    theta, xi = plda_params(psi, n)
    m = (n * psi / (n * psi + 1.0)) * ubar_g     # elementwise
    up2 = u_probe * u_probe
    diff2 = (u_probe - m) * (u_probe - m)
    return float(-np.sum(xi * diff2) + np.sum(theta * up2))

# ----------------- main -----------------
def main(wav_path: Path, server_url: str, backend: str, th: float):
    # load gallery per backend
    suffix = "cosine" if backend == "cosine" else "plda"
    enrolled_path = ARTIFACTS / f"suspects_embeddings.{suffix}.json"
    if not enrolled_path.exists():
        print(f"[ERR] {enrolled_path} not found. Run enroll.py --backend {backend} first.")
        sys.exit(1)
    enrolled = json.load(open(enrolled_path))
    names = sorted(enrolled.keys())
    gallery = [np.array(enrolled[n]) for n in names]

    # probe embedding (ECAPA, L2)
    probe_raw = wav_to_embedding(wav_path)

    # build integer scores where NEGATIVE => "match"
    if backend == "cosine":
        scores_float = [cosine(probe_raw, g) for g in gallery]
        tau = th  # cosine threshold from CLI
        K = 1000
        s_int = [int(round(K * (tau - s))) for s in scores_float]
    else:
        # PLDA: transform probe to PLDA space, score vs PLDA-space class means
        bk = np.load(ARTIFACTS / "plda_backend.npz")
        gmean = bk["mean"]; V = bk["V"]; psi = bk["psi"]
        n_enroll = 2  # how many enrollment utts per suspect you used; change if needed

        u_probe = V.T @ (probe_raw - gmean)
        u_probe = u_probe / (np.linalg.norm(u_probe) + 1e-9)

        scores_float = [plda_score_float(u_probe, g, psi, n_enroll) for g in gallery]
        # simple pivot; if NO MATCH often, try: tau = np.median(scores_float)
        tau = 0.0
        K = 1000
        s_int = [int(round(K * (tau - s))) for s in scores_float]

    # encrypt + obfuscate + permute
    pk = get_pubkey(server_url)
    r1 = 4096  # scalar multiplier to hide raw magnitude; preserves sign
    enc_list = []
    for s in s_int:
        enc_s = pk.encrypt(s)    # Enc(s)
        delta = enc_s * r1       # Enc(r1 * s)
        enc_list.append(delta)

    idx = list(range(len(enc_list)))
    random.shuffle(idx)
    perm_enc = [enc_list[i] for i in idx]

    payload = {"scores": [{"c": str(x.ciphertext()), "e": x.exponent} for x in perm_enc]}
    rr = requests.post(f"{server_url}/match", json=payload, timeout=30)
    rr.raise_for_status()
    res = rr.json()

    if res.get("match") and res.get("index", -1) >= 0:
        real_idx = idx[res["index"]]
        print(f"[SERVER] MATCH => suspect={names[real_idx]}")
    else:
        print("[SERVER] NO MATCH")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cosine", "plda"], default="cosine")
    ap.add_argument("--wav", required=True, help="Probe wav, e.g., data\\probes\\probe1.wav")
    ap.add_argument("--server", default="http://127.0.0.1:8008", help="Server URL")
    ap.add_argument("--th", type=float, default=0.55, help="Cosine threshold (ignored for PLDA)")
    args = ap.parse_args()
    main(Path(args.wav), args.server, args.backend, args.th)