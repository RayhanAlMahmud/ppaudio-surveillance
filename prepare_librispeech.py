import argparse
from pathlib import Path
import torchaudio
import soundfile as sf
import torch
import numpy as np

def ensure_dirs():
    (Path("data/suspects/person1")).mkdir(parents=True, exist_ok=True)
    (Path("data/suspects/person2")).mkdir(parents=True, exist_ok=True)
    (Path("data/probes")).mkdir(parents=True, exist_ok=True)
    Path("data/librispeech").mkdir(parents=True, exist_ok=True)

def to_mono_16k(waveform: torch.Tensor, sr: int):
    """Convert waveform [channels, time] to mono @ 16 kHz."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    # mixdown to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # resample if needed
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    return waveform.squeeze(0).cpu().numpy(), sr

def main(args):
    ensure_dirs()

    root = Path("data/librispeech")
    print("⏬ Downloading/Preparing LibriSpeech test-clean (first time ≈ 346 MB) ...")
    _ = torchaudio.datasets.LIBRISPEECH(root=str(root), url="test-clean", download=True)

    # Build an in-memory index grouped by speaker
    ds = torchaudio.datasets.LIBRISPEECH(root=str(root), url="test-clean", download=False)
    by_spk = {}
    for i in range(len(ds)):
        # Each item: (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        _, _, _, speaker_id, _, _ = ds[i]
        by_spk.setdefault(speaker_id, []).append(i)

    # Pick N suspects (default 2) and M files per suspect (default 2)
    speakers = sorted(by_spk.keys())
    if len(speakers) < args.suspects:
        raise RuntimeError(f"Dataset has only {len(speakers)} speakers; reduce --suspects")

    chosen = speakers[:args.suspects]
    print(f"✅ Selected suspects (speaker IDs): {chosen}")

    # Save WAVs into data/suspects/personX and one probe from person1
    saved_counts = {}
    person_dirs = []
    for idx, spk in enumerate(chosen, start=1):
        person_dir = Path(f"data/suspects/person{idx}")
        person_dirs.append(person_dir)
        person_dir.mkdir(parents=True, exist_ok=True)
        # take first K items for this speaker
        chosen_items = by_spk[spk][:args.per_speaker]
        if len(chosen_items) < args.per_speaker:
            print(f"[WARN] Speaker {spk} has only {len(chosen_items)} items; requested {args.per_speaker}.")
        count = 0
        for j, item_idx in enumerate(chosen_items, start=1):
            waveform, sr, _, _, _, _ = ds[item_idx]
            wav, out_sr = to_mono_16k(waveform, sr)
            outpath = person_dir / f"file{j}.wav"
            sf.write(str(outpath), wav, out_sr)
            count += 1
        saved_counts[f"person{idx}"] = count

    # Make a probe from the 3rd utterance of suspect 1 (or fallback if not enough)
    probe_path = Path("data/probes/probe1.wav")
    p1_items = by_spk[chosen[0]]
    probe_idx = p1_items[min(2, len(p1_items)-1)]  # index 2 if exists else last one
    waveform, sr, *_ = ds[probe_idx]
    wav, out_sr = to_mono_16k(waveform, sr)
    sf.write(str(probe_path), wav, out_sr)

    print("\n📂 Files created:")
    for k, v in saved_counts.items():
        print(f"  - data/suspects/{k}/ : {v} wav files")
    print(f"  - data/probes/probe1.wav")

    print("\nAll set! Now run:")
    print("  python enroll.py")
    print("  python device.py --wav data\\probes\\probe1.wav --th 0.55")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suspects", type=int, default=2, help="How many suspects to enroll (default 2)")
    ap.add_argument("--per_speaker", type=int, default=2, help="How many wavs per suspect (default 2)")
    args = ap.parse_args()
    main(args)
