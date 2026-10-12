#!/usr/bin/env python3
"""
codim_investigate_ref.py — point-investigate ONE reference audio file: where
does it actually best match among the pak tracks, and is that track already
claimed by a different reference in an existing matches.json?

Built for exactly the "gaz says good" case: a short reference clip (a
fraction of a second) where the fingerprint pre-filter used by the main
matcher / match_remaining.py is unreliable (coarse 64-bin RMS envelope
carries very little discriminating information over ~200-400ms of audio),
so the true match can rank outside --shortlist and never reach the exact
FFT correlation that would have identified it correctly.

For short clips this doesn't cost anything extra to fix properly: exact
correlation is CHEAP when the audio itself is short (few samples -> fast
FFT), so this tool just skips the fingerprint stage entirely for short refs
and runs the exact correlation against every duration-bucket candidate.

What it reports for the top matches:
  - the score (same exact FFT normalized cross-correlation as the main
    matcher / match_remaining.py — directly comparable)
  - whether that track is already claimed by some OTHER reference in the
    matches.json you pass in (scenario A: hijacked) or free (scenario B:
    the main matcher just never looked at it hard enough)

Usage:
  python3 codim_investigate_ref.py <ref.wav> <pak_dir> [matches.json]
      [--tolerance 0.5] [--top 15] [--min-duration-for-fingerprint 1.0]

  --tolerance is a FRACTION of the reference's own duration (not a fixed
  number of seconds) — default 0.5 means +/-50% of the ref's length, which
  for very short clips still gives a sane absolute window since it's
  combined with a small fixed floor internally.
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np

SR = 8000
MAX_LAG = int(0.08 * SR)
MIN_ABS_TOLERANCE = 0.15  # seconds, floor so very short clips don't get an absurdly tiny window


def decode_pcm(path: str, sr: int = SR):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    if not raw:
        raise RuntimeError(f"ffmpeg produced no audio output for {path}")
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def probe_duration(path: str):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def norm_xcorr_best_fft(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    n = min(len(a), len(b))
    if n < SR * 0.05:
        return 0.0
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    a0 = a - a.mean()
    b0 = b - b.mean()
    fsize = 1
    while fsize < 2 * n:
        fsize *= 2
    fa = np.fft.rfft(a0, fsize)
    fb = np.fft.rfft(b0, fsize)
    xcorr = np.fft.irfft(fa * np.conj(fb), fsize)
    a2 = np.concatenate(([0.0], np.cumsum(a0 ** 2)))
    b2 = np.concatenate(([0.0], np.cumsum(b0 ** 2)))
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            ea = a2[n] - a2[lag]
            eb = b2[n - lag] - b2[0]
            ov = n - lag
        else:
            ea = a2[n + lag] - a2[0]
            eb = b2[n] - b2[-lag]
            ov = n + lag
        if ov < SR * 0.05:
            continue
        denom = np.sqrt(ea * eb)
        if denom < 1e-6:
            continue
        idx = lag if lag >= 0 else fsize + lag
        score = xcorr[idx] / denom
        if score > best:
            best = float(score)
    return max(best, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_wav")
    ap.add_argument("pak_dir")
    ap.add_argument("matches_json", nargs="?", help="optional: existing matches.json, to flag hijacked tracks")
    ap.add_argument("--tolerance", type=float, default=0.5,
                     help="duration window as a FRACTION of the ref's own length (default 0.5 = +/-50%%)")
    ap.add_argument("--top", type=int, default=15, help="how many top candidates to print (default 15)")
    args = ap.parse_args()

    ref_dur = probe_duration(args.ref_wav)
    if ref_dur is None:
        sys.exit(f"could not probe duration of {args.ref_wav}")
    ref_pcm = decode_pcm(args.ref_wav)
    print(f"Reference: {args.ref_wav}  duration={ref_dur:.3f}s")

    tol = max(MIN_ABS_TOLERANCE, ref_dur * args.tolerance)
    print(f"Duration window: +/-{tol:.3f}s  ({ref_dur - tol:.3f}s .. {ref_dur + tol:.3f}s)")

    existing = {}
    reverse_existing = {}
    if args.matches_json:
        existing = json.load(open(args.matches_json, encoding="utf-8"))
        for track, name in existing.items():
            reverse_existing.setdefault(name, []).append(track)
        print(f"Loaded {len(existing)} existing match(es) from {args.matches_json}")

    print("\nScanning pak_dir durations (ffprobe, cheap)...")
    all_tracks = sorted(f for f in os.listdir(args.pak_dir)
                         if f.lower().endswith(".mp3") and "_part" not in f.lower())
    print(f"  {len(all_tracks)} whole track(s) total")

    candidates = []
    for i, fn in enumerate(all_tracks):
        d = probe_duration(os.path.join(args.pak_dir, fn))
        if d is not None and abs(d - ref_dur) <= tol:
            candidates.append(fn)
        if (i + 1) % 2000 == 0:
            print(f"  scanned {i + 1}/{len(all_tracks)}, {len(candidates)} candidate(s) so far")

    print(f"\n{len(candidates)} candidate(s) within the duration window. "
          f"Running EXACT correlation on all of them (no fingerprint pre-filter)...")

    scored = []
    for i, fn in enumerate(candidates):
        pcm = decode_pcm(os.path.join(args.pak_dir, fn))
        score = norm_xcorr_best_fft(ref_pcm, pcm, MAX_LAG)
        claimed_by = existing.get(fn)  # None if unclaimed
        scored.append((score, fn, claimed_by))
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(candidates)}")

    scored.sort(key=lambda s: -s[0])

    print(f"\nTop {min(args.top, len(scored))} match(es):")
    print(f"{'score':>7}  {'track':<20} claimed by (in matches.json)")
    for score, fn, claimed_by in scored[:args.top]:
        ref_name_here = os.path.basename(args.ref_wav)
        if claimed_by is None:
            status = "(unclaimed)"
        elif claimed_by == ref_name_here:
            status = "(already correctly matched to THIS ref)"
        else:
            status = f"-> currently matched to '{claimed_by}'  [HIJACKED if this is the true match]"
        print(f"{score:7.3f}  {fn:<20} {status}")

    if reverse_existing.get(os.path.basename(args.ref_wav)):
        print(f"\nNote: '{os.path.basename(args.ref_wav)}' is ALREADY assigned in matches.json to: "
              f"{reverse_existing[os.path.basename(args.ref_wav)]}")


if __name__ == "__main__":
    main()
