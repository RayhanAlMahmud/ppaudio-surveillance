# ppaudio-surveillance

Proof-of-concept privacy-preserving audio surveillance
This project aims to build a scalable surveillance system







Privacy-Preserving Audio Surveillance (Client–Server, Cosine \& PLDA)

This project is a coursework-scale reproduction of a privacy-preserving speaker search system inspired by “Towards a Scalable and Privacy-Preserving Audio Surveillance System”.

It runs a device → server pipeline where the device extracts a speaker embedding, computes per-suspect scores, obfuscates \& encrypts them, permutes the list, and sends to a server.

The server only learns if any suspect matches (negative score) and which permuted index—no raw audio, embeddings, or clear scores leave the device.



Features

ECAPA-TDNN embeddings (SpeechBrain)



Two backends:



Cosine (simple, fast; good baseline)



PLDA (paper-style; trained from LibriSpeech)



Encrypted client→server scoring (Paillier HE) + simple obfuscation



Runs locally; works offline on Windows (no symlinks) ✅



1\) Environment

Windows + Anaconda recommended (Python 3.10+)



bat

Copy code

conda create -n ppaudio python=3.10 -y

conda activate ppaudio



pip install fastapi uvicorn phe requests numpy scipy torch torchaudio librosa soundfile speechbrain

Optional Windows runtimes (if DLL popups appear)

bat

Copy code

conda install -y vs2015\_runtime ucrt   :: or: conda install -y -c conda-forge vc14\_runtime ucrt

Also install Microsoft “Visual C++ Redistributable 2015–2022 (x64)” from the official site if needed.



2\) Repo layout

pgsql

Copy code

ppaudio-surveillance/

├─ artifacts/

│  ├─ ecapa-voxceleb/           # local copy of SpeechBrain ECAPA model (no internet needed)

│  ├─ plda\_backend.npz          # PLDA transform (trained once)

│  ├─ suspects\_embeddings.cosine.json

│  └─ suspects\_embeddings.plda.json

├─ data/

│  ├─ librispeech/              # auto-downloaded by torchaudio (ignored in git)

│  ├─ suspects/                 # wavs per enrolled person (created by prep script)

│  └─ probes/                   # probe wavs

├─ device.py                    # local-only baseline matcher (no server)

├─ device\_client.py             # encrypted client (cosine | plda)

├─ enroll.py                    # enroll suspects (cosine | plda)

├─ prepare\_librispeech.py       # make small suspects+probes set from LibriSpeech

├─ server.py                    # FastAPI server (HE decrypt + match)

├─ train\_plda\_simplified.py     # train PLDA backend (from LibriSpeech)

└─ README.md

3\) Model cache (offline \& Windows-safe)

We run fully offline and avoid symlinks:



enroll.py / device\_client.py set:



HF\_HUB\_OFFLINE=1 (use local HF cache only)



SPEECHBRAIN\_LOCAL\_DOWNLOAD\_STRATEGY=copy (no symlinks)



ECAPA model is loaded from a fixed local folder: artifacts/ecapa-voxceleb/



If this folder isn’t present, copy it once from your Hugging Face cache (path may vary):



bat

Copy code

xcopy /E /I /Y "C:\\Users\\<YOU>\\.cache\\huggingface\\hub\\models--speechbrain--spkrec-ecapa-voxceleb\\snapshots\\<hash>" "D:\\ppaudio-surveillance\\artifacts\\ecapa-voxceleb\\"

4\) Data prep (tiny subset for demo)

Creates a few suspects (folders with wavs) and one or two probes from LibriSpeech test-clean.



bat

Copy code

conda activate ppaudio

python prepare\_librispeech.py                :: default 2 suspects x 2 utts + 1 probe

:: examples:

:: python prepare\_librispeech.py --suspects 5 --per\_speaker 3 --probes 2

5\) Train PLDA backend (once)

bat

Copy code

python train\_plda\_simplified.py

This produces artifacts\\plda\_backend.npz (global mean, transform V, eigenvalues ψ).



6\) Enroll suspects

Two DBs are created—one for each backend.



bat

Copy code

:: cosine DB

python enroll.py --backend cosine



:: PLDA DB (after training step above)

python enroll.py --backend plda

Output:



artifacts\\suspects\_embeddings.cosine.json



artifacts\\suspects\_embeddings.plda.json



7\) Run the encrypted client–server pipeline

Start the server

bat

Copy code

uvicorn server:app --host 127.0.0.1 --port 8008 --reload

Run the client (pick backend)

bat

Copy code

:: Cosine (uses --th threshold; NEGATIVE after int transform means match)

python device\_client.py --backend cosine --wav data\\probes\\probe1.wav --th 0.55 --server http://127.0.0.1:8008



:: PLDA (float scoring; simple tau=0.0; you can set tau = median for robustness)

python device\_client.py --backend plda   --wav data\\probes\\probe1.wav --server http://127.0.0.1:8008

Expected console:



ini

Copy code

\[SERVER] MATCH => suspect=personX

Tip: For PLDA, if you ever see “NO MATCH”, open device\_client.py and change:

tau = 0.0 → tau = np.median(scores\_float) to auto-center.



8\) Local-only baseline (no server)

For quick sanity checks:



bat

Copy code

python device.py --wav data\\probes\\probe1.wav --th 0.55

9\) How the privacy works (short)

Device extracts probe embedding.



For each enrolled suspect, compute a score (cosine or PLDA).



Map to an integer where negative ⇒ match; encrypt using Paillier.



Obfuscate (multiply by random scalar r₁) and permute order.



Server decrypts list and returns only: {match: true/false, index} of the most negative item.



No raw audio, embeddings, or clear scores leave the device.



10\) Troubleshooting

A) /pubkey JSONDecodeError



Server not running or port mismatch. Start server on 8008, then hit http://127.0.0.1:8008/pubkey in a browser—should return {"n":"..."}.



B) Windows symlink error (WinError 1314)



We already force copy strategy and fixed local savedir. If you still see it:



Delete half-made folder: rmdir /s /q artifacts\\ecapa-voxceleb



Copy the model snapshot manually into artifacts\\ecapa-voxceleb (see §3).



C) SSL certificate errors (Hugging Face)



We run offline. Ensure HF\_HUB\_OFFLINE=1 is set (already in code).



If you do need online download: pip install -U certifi requests huggingface\_hub and set SSL\_CERT\_FILE to ...site-packages\\certifi\\cacert.pem.



D) VCRUNTIME / ucrtbased.dll popups



conda install -y vs2015\_runtime ucrt, install MSVC redist.



Reinstall PyTorch/torchaudio from conda release builds (avoid debug ucrtbased.dll).



E) PLDA says NO MATCH



Use tau = np.median(scores\_float) for a quick, data-driven threshold.



11\) Commands cheat-sheet

bat

Copy code

:: server

uvicorn server:app --host 127.0.0.1 --port 8008 --reload



:: client (cosine)

python device\_client.py --backend cosine --wav data\\probes\\probe1.wav --th 0.55 --server http://127.0.0.1:8008



:: client (plda)

python device\_client.py --backend plda --wav data\\probes\\probe2.wav --server http://127.0.0.1:8008

12\) Acknowledgments

SpeechBrain for ECAPA speaker embeddings



LibriSpeech dataset



phe (Paillier) library for additively homomorphic encryption



FastAPI + Uvicorn for the server



13\) License

MIT (or your choice)



14\) Roadmap (optional)

Quantized integer PLDA s′ (paper-style) end-to-end



ZK proof stub for “server proves a negative score index”



Diarization + VAD front-end



Larger galleries + simple precomputation cache on device

