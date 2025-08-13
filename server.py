from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from phe import paillier  
#server.py
app = FastAPI(title="PP-Audio Server")

# Generate Paillier keypair at startup
PUB, PRIV = paillier.generate_paillier_keypair() #Fixed during debugging...Priv, pUB was flipped and In phe, generate_paillier_keypair() returns (public_key, private_key),So PUB was actually the private key, which is why PUB.n exploded.
print("Paillier keys ready:", type(PUB).__name__, type(PRIV).__name__)

class EncScore(BaseModel):
    c: str     # ciphertext as decimal string
    e: int     # exponent

class MatchRequest(BaseModel):
    scores: List[EncScore]  # encrypted & obfuscated scores in a random order

@app.get("/pubkey")
def pubkey():
    # only need 'n' to build PaillierPublicKey on the client
    return {"n": str(PUB.n)}

@app.post("/match")
def match(req: MatchRequest):
    # Reconstruct EncryptedNumbers and decrypt to plaintext ints
    plains = []
    for item in req.scores:
        enc = paillier.EncryptedNumber(PUB, int(item.c), item.e)
        val = PRIV.decrypt(enc)  # float due to exponent
        plains.append(int(round(val)))
    # A match is indicated by a NEGATIVE value (by our construction)
    best_idx = None
    best_val = None
    for i, v in enumerate(plains):
        if (best_val is None) or (v < best_val):
            best_val, best_idx = v, i
    found = (best_val is not None) and (best_val < 0)
    return {"match": bool(found), "index": (best_idx if found else -1)}
