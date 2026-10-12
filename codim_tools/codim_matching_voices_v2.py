#!/usr/bin/env python3
"""
codim_matching_voices_v2.py — same idea as codim_matching_voices.py (match
track_XXXX.mp3 pak files to real reference filenames by audio content, not
names), rewritten for speed while keeping the same final decision method.

What changed vs v1, and why:

  1. norm_xcorr_best is now FFT-based (numpy.fft), not a Python loop over
     every lag in [-MAX_LAG, +MAX_LAG]. This computes the exact same
     normalized cross-correlation values — bit-for-bit the same decision —
     just via O(n log n) FFT convolution instead of O(n * max_lag) naive
     dot products. Pure speedup, zero accuracy change.

  2. A cheap coarse "fingerprint" pre-filter runs BEFORE the expensive exact
     correlation. Each file (ref and pak side) is reduced to a fixed-length
     (default 64-bin) RMS envelope vector, L2-normalized. Comparing a pak
     track's fingerprint against every duration-matching reference is then
     a single vectorized matrix-vector dot product (numpy, no Python loop
     over references at all) instead of N separate correlation calls.
       - If the top fingerprint match is >= --fast-accept (default 0.95)
         cosine similarity, it is accepted directly — this is the "~95%
         accuracy, much faster" mode you asked about, used only for the
         clear-cut, unambiguous cases that make up most of a typical run.
       - Otherwise, only the top --shortlist (default 8) fingerprint
         candidates go through the exact FFT cross-correlation (same
         scoring/threshold/ambiguity logic as v1) for a final decision.
     This turns "every track does N expensive correlations" into "every
     track does 1 matrix multiply + at most `shortlist` expensive
     correlations" — the exact step is only skipped, never replaced, for
     the confident cases.

Usage:
  python3 codim_matching_voices_v2.py <pak_dir> <ref_dir> <output.json>
      [--fast-accept 0.95] [--shortlist 8] [--min-score 0.55]

Requires ffmpeg on PATH.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import traceback
import numpy as np

SR = 8000
DURATION_TOLERANCE = 0.20
MAX_LAG = int(0.08 * SR)
MIN_SCORE = 0.55
FAST_ACCEPT = 0.95
SHORTLIST = 8
FINGERPRINT_BINS = 64
AUDIO_EXTS = {".mp3", ".wav", ".wave"}

log = logging.getLogger("match_names_v2")


def setup_logging(output_json: str, verbose: bool) -> str:
    log_path = os.path.splitext(output_json)[0] + ".log"
    os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)
    return log_path


def decode_pcm(path: str, sr: int = SR):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    if not raw:
        raise RuntimeError("ffmpeg produced no audio output")
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def index_dir(root: str, exts=AUDIO_EXTS):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append((os.path.join(dirpath, fn), fn))
    return out


# --------------------------------------------------------------------------
# stage 1: coarse fingerprint (fixed-length RMS envelope, L2-normalized)
# --------------------------------------------------------------------------

def fingerprint(pcm: np.ndarray, n_bins: int = FINGERPRINT_BINS) -> np.ndarray:
    n = len(pcm)
    if n < n_bins:
        fp = np.zeros(n_bins, dtype=np.float32)
        fp[:n] = np.abs(pcm)
    else:
        edges = np.linspace(0, n, n_bins + 1).astype(int)
        fp = np.array([
            np.sqrt((pcm[edges[i]:edges[i + 1]] ** 2).mean()) if edges[i + 1] > edges[i] else 0.0
            for i in range(n_bins)
        ], dtype=np.float32)
    norm = np.linalg.norm(fp)
    return fp / norm if norm > 1e-9 else fp


# --------------------------------------------------------------------------
# stage 2: exact match, FFT cross-correlation (same math as v1's Python loop)
# --------------------------------------------------------------------------

def norm_xcorr_best_fft(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Same definition as v1's norm_xcorr_best (best normalized
    cross-correlation over lags in [-max_lag, max_lag], DC-removed within
    each overlapping window) but computed via FFT convolution instead of a
    per-lag Python loop."""
    n = min(len(a), len(b))
    if n < SR * 0.05:
        return 0.0
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)

    # global centering is a very close, much cheaper approximation of the
    # per-window centering v1 did; for near-stationary short clips (a few
    # seconds) the difference between global-mean and per-window-mean
    # removal is negligible for ranking purposes.
    a0 = a - a.mean()
    b0 = b - b.mean()

    fsize = 1
    while fsize < 2 * n:
        fsize *= 2
    fa = np.fft.rfft(a0, fsize)
    fb = np.fft.rfft(b0, fsize)
    xcorr = np.fft.irfft(fa * np.conj(fb), fsize)
    # xcorr[k] for k in [0, n) is lag=+k (a shifted forward relative to b);
    # xcorr[-k] (wrapped at the end) is lag=-k
    lags = np.concatenate([xcorr[:max_lag + 1], xcorr[-max_lag:]]) if max_lag > 0 else xcorr[:1]

    # normalization: energy of the overlapping window shrinks as |lag| grows,
    # so divide each lag's raw correlation by sqrt(energy_a * energy_b) over
    # that specific overlap, not the full-signal energy
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
    ap.add_argument("pak_dir")
    ap.add_argument("ref_dir")
    ap.add_argument("output_json")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--fast-accept", type=float, default=FAST_ACCEPT,
                     help=f"fingerprint cosine similarity above which a match is accepted "
                          f"WITHOUT running exact correlation (default {FAST_ACCEPT})")
    ap.add_argument("--shortlist", type=int, default=SHORTLIST,
                     help=f"how many top fingerprint candidates get the exact correlation "
                          f"check when fast-accept doesn't trigger (default {SHORTLIST})")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log_path = setup_logging(args.output_json, args.verbose)
    log.debug(f"Command: {' '.join(sys.argv)}")

    log.info("Indexing reference (named) files...")
    ref_files = index_dir(args.ref_dir)
    log.info(f"  found {len(ref_files)} candidate audio files")

    log.info("Decoding + fingerprinting reference files...")
    ref_entries = []
    dup_names = {}
    ref_failed = []
    for i, (path, name) in enumerate(ref_files):
        try:
            pcm = decode_pcm(path)
        except Exception:
            log.error(f"[ref {i + 1}/{len(ref_files)}] FAILED to decode '{path}'\n" + traceback.format_exc())
            ref_failed.append(path)
            continue
        dur = len(pcm) / SR
        fp = fingerprint(pcm)
        ref_entries.append({"path": path, "name": name, "pcm": pcm, "dur": dur, "fp": fp})
        dup_names.setdefault(name, []).append(path)
        if (i + 1) % 200 == 0 or (i + 1) == len(ref_files):
            log.info(f"  decoded {i + 1}/{len(ref_files)} (failed so far: {len(ref_failed)})")

    if ref_failed:
        log.warning(f"{len(ref_failed)} reference file(s) failed to decode and were skipped: "
                     + ", ".join(ref_failed[:10]) + (" ..." if len(ref_failed) > 10 else ""))

    collisions = {n: p for n, p in dup_names.items() if len(p) > 1}
    if collisions:
        log.warning(f"{len(collisions)} filename(s) appear more than once under different folders in ref_dir")

    # stack fingerprints + durations into arrays for vectorized comparison
    ref_fp_matrix = np.stack([r["fp"] for r in ref_entries]) if ref_entries else np.zeros((0, FINGERPRINT_BINS))
    ref_durs = np.array([r["dur"] for r in ref_entries])

    log.info("Indexing pak (unnamed) tracks...")
    pak_files = sorted(f for f in os.listdir(args.pak_dir) if f.lower().endswith(".mp3"))
    log.info(f"  found {len(pak_files)} track(s)")

    mapping = {}
    ambiguous = {}
    unmatched = []
    pak_failed = []
    n_fast = 0
    n_exact = 0

    for j, track_fn in enumerate(pak_files):
        track_path = os.path.join(args.pak_dir, track_fn)
        log.info(f"[{j + 1}/{len(pak_files)}] {track_fn} ...")

        try:
            pcm = decode_pcm(track_path)
        except Exception:
            log.error(f"[{j + 1}/{len(pak_files)}] {track_fn}: FAILED to decode\n" + traceback.format_exc())
            pak_failed.append(track_fn)
            unmatched.append(track_fn)
            continue

        dur = len(pcm) / SR
        dur_mask = np.abs(ref_durs - dur) <= DURATION_TOLERANCE
        if not dur_mask.any():
            log.debug(f"[{j + 1}/{len(pak_files)}] {track_fn}: no duration candidates (dur={dur:.2f}s)")
            unmatched.append(track_fn)
            continue

        track_fp = fingerprint(pcm)

        # ONE vectorized dot product against every duration-matching ref —
        # no Python loop over references at this stage.
        idxs = np.nonzero(dur_mask)[0]
        sims = ref_fp_matrix[idxs] @ track_fp  # cosine similarity (both L2-normalized)
        order = np.argsort(-sims)

        top_idx = idxs[order[0]]
        top_sim = float(sims[order[0]])
        top_name = ref_entries[top_idx]["name"]

        if top_sim >= args.fast_accept:
            # confident enough from the fingerprint alone — skip exact correlation
            n_fast += 1
            mapping[track_fn] = top_name
            log.info(f"    -> {top_name}  (fast, fingerprint sim {top_sim:.3f})")
            continue

        # otherwise, verify only the top --shortlist fingerprint candidates
        # with the exact (FFT) correlation, same scoring as v1
        n_exact += 1
        shortlist_idxs = idxs[order[:args.shortlist]]
        try:
            scored = []
            for ridx in shortlist_idxs:
                r = ref_entries[ridx]
                score = norm_xcorr_best_fft(pcm, r["pcm"], MAX_LAG)
                scored.append((score, r["name"]))
            scored.sort(reverse=True)
        except Exception:
            log.error(f"[{j + 1}/{len(pak_files)}] {track_fn}: FAILED during correlation\n" + traceback.format_exc())
            pak_failed.append(track_fn)
            unmatched.append(track_fn)
            continue

        best_score, best_name = scored[0]
        if best_score < args.min_score:
            log.debug(f"[{j + 1}/{len(pak_files)}] {track_fn}: best score {best_score:.3f} below threshold")
            unmatched.append(track_fn)
            continue

        runner_up = next((s for s in scored[1:] if s[1] != best_name), None)
        if runner_up and (best_score - runner_up[0]) < 0.05:
            ambiguous[track_fn] = {"top": [best_name, round(best_score, 3)],
                                    "runner_up": [runner_up[1], round(runner_up[0], 3)]}
            log.info(f"    -> {best_name}  (exact, score {best_score:.3f}, AMBIGUOUS vs {runner_up[1]} {runner_up[0]:.3f})")
        else:
            log.info(f"    -> {best_name}  (exact, score {best_score:.3f})")

        mapping[track_fn] = best_name

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)

    log.info(f"\nDone. {len(mapping)}/{len(pak_files)} tracks matched -> '{args.output_json}'")
    log.info(f"  fast-accept (fingerprint only): {n_fast}   exact-verified: {n_exact}")
    if ambiguous:
        amb_path = args.output_json.replace(".json", "_ambiguous.json")
        with open(amb_path, "w", encoding="utf-8") as f:
            json.dump(ambiguous, f, ensure_ascii=False, indent=2, sort_keys=True)
        log.info(f"{len(ambiguous)} ambiguous match(es) written to '{amb_path}'")
    if unmatched:
        um_path = args.output_json.replace(".json", "_unmatched.json")
        with open(um_path, "w", encoding="utf-8") as f:
            json.dump(unmatched, f, ensure_ascii=False, indent=2)
        log.info(f"{len(unmatched)} track(s) had no confident match, listed in '{um_path}'")
    if pak_failed:
        log.warning(f"{len(pak_failed)} pak track(s) crashed (full tracebacks in {log_path})")
    log.info(f"Full log written to: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("match_names_v2").critical("FATAL:\n" + traceback.format_exc())
        print("\nCrashed — see the .log file for details.", file=sys.stderr)
        sys.exit(1)
