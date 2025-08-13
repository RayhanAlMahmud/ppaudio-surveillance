import json, random, argparse, requests
from pathlib import Path
import numpy as np
import librosa, torch
from speechbrain.inference import EncoderClassifier
from phe import paillier

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"

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

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def main(wav_path: Path, server_url: str, threshold: float):
    # Load enrolled suspects
    enrolled_path = ARTIFACTS / "suspects_embeddings.json"
    if not enrolled_path.exists():
        print("[ERR] Run enroll.py first.")
        return
    enrolled = json.load(open(enrolled_path))
    names = sorted(enrolled.keys())
    gallery = [np.array(enrolled[n]) for n in names]

    # Probe embedding
    probe = wav_to_embedding(wav_path)

    # Build integer scores where NEGATIVE means "match"
    # s_int = round(K * (threshold - cosine))  => if cosine >= threshold -> s_int <= 0
    K = 1000
    s_int = [int(round(K * (threshold - cosine(probe, g)))) for g in gallery]

    # Get public key from server
    r = requests.get(f"{server_url}/pubkey", timeout=10)
    pk = paillier.PaillierPublicKey(n=int(r.json()["n"]))

    # Encrypt + obfuscate (δ = r1*s + r2). We keep r2=0 so the sign is preserved.
    r1 = 4096
    enc_list = []
    for s in s_int:
        enc_s = pk.encrypt(s)      # Enc(s)
        delta = enc_s * r1         # r1 * Enc(s)  => Enc(r1*s)
        # Optionally add small r2: delta += pk.encrypt(0)
        enc_list.append(delta)

    # Permute (and remember the mapping)
    idx = list(range(len(enc_list)))
    random.shuffle(idx)
    perm_enc = [enc_list[i] for i in idx]

    # Serialize ciphertexts for JSON
    payload = {
        "scores": [{"c": str(x.ciphertext()), "e": x.exponent} for x in perm_enc]
    }

    # Send to server
    rr = requests.post(f"{server_url}/match", json=payload, timeout=30)
    rr.raise_for_status()
    res = rr.json()
    if res["match"] and res["index"] >= 0:
        # Map back from permuted index to suspect name
        real_idx = idx[res["index"]]
        print(f"[SERVER] MATCH => suspect={names[real_idx]}")
    else:
        print("[SERVER] NO MATCH")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Probe wav path, e.g., data\\probes\\probe1.wav")
    ap.add_argument("--server", default="http://127.0.0.1:8008", help="Server base URL")
    ap.add_argument("--th", type=float, default=0.55, help="Cosine threshold (default 0.55)")
    args = ap.parse_args()
    main(Path(args.wav), args.server, args.th)
