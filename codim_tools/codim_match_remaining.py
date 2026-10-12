#!/usr/bin/env python3
"""
match_remaining.py — force a match for EVERY reference (named) audio file
that codim_matching_voices_v2.py did NOT match, without re-running or
disturbing any of the matches it already found. This is the "harsh /
whatever it takes" pass: no duration filter, no score threshold, and an
optimal (not greedy) assignment so that as long as there are at least as
many leftover tracks as leftover references, every single reference gets
SOME track - none are left in "still unmatched" purely because of a
collision with another reference.

Why this is a separate pass and not just "lower --min-score and rerun":
  - Rerunning the full matcher redoes the expensive decode+correlate work
    for all ~19700 tracks again, and risks a track that was correctly
    matched before landing on a different (possibly wrong) reference this
    time, since scores are all close together near the threshold.
  - This script instead: (1) keeps every existing match exactly as-is,
    (2) computes what's left over - references with no match yet, and
    tracks not already claimed by an existing match, (3) searches ONLY
    within that leftover pool, and (4) unlike the main matcher, never
    rejects anything: every remaining reference is FORCED onto its best
    available remaining track. Weak ones are flagged in a separate report
    for manual review, but none are dropped.

Method:
  1. Load existing_matches.json -> matched track filenames and matched
     reference filenames are both excluded from consideration.
  2. Index ref_dir / pak_dir, keep only what's not already matched/claimed.
  3. Decode + fingerprint every remaining track ONCE.
  4. Build the FULL cosine-similarity matrix (fingerprint) between every
     remaining reference and every remaining track - no duration cutoff,
     no shortlist cutoff at this stage. This is one matrix multiply, cheap
     even for a few hundred references x tens of thousands of tracks.
  5. Solve it as an assignment problem (Hungarian algorithm,
     scipy.optimize.linear_sum_assignment) maximizing total similarity,
     one track per reference. This is the "harsh" part: a track that's
     only a mediocre fit for reference A but the only thing left over
     still gets assigned to A rather than being left unmatched, and no
     greedy collision ever silently strands a reference that had a decent
     second-best option.
  6. For just the resulting pairs (small set), run the EXACT FFT
     cross-correlation (same scoring as the main matcher) to get a real
     confidence score for the report / weak-flagging - the assignment
     itself is decided on the fingerprint matrix (fast), the exact
     correlation here is purely for you to review confidence afterwards.

Output:
  <output_json>                       existing matches + newly forced matches, merged
  <output_json>_forced.json           just the newly added pairs, with their score
  <output_json>_still_unmatched.json  references left over only if there were
                                       literally fewer remaining tracks than
                                       remaining references (a hard structural
                                       limit no algorithm can work around)

Usage:
  python3 match_remaining.py existing_matches.json pak_dir ref_dir output.json
      [--weak-below 0.55] [--greedy]

  --greedy uses the old, faster-but-not-guaranteed per-reference top-1
  approach instead of the optimal assignment (only useful if the leftover
  track pool is too large for the assignment solve to be practical).
"""

import argparse
import json
import logging
import os
import sys
import traceback
import numpy as np
from scipy.optimize import linear_sum_assignment

# reuse the exact same building blocks as the main matcher, unmodified
from codim_matching_voices_v2 import (
    SR, decode_pcm, index_dir, fingerprint, norm_xcorr_best_fft, FINGERPRINT_BINS,
)

MAX_LAG = int(0.08 * SR)
DEFAULT_WEAK_BELOW = 0.55  # same value as the main matcher's MIN_SCORE, just no longer a hard cutoff

log = logging.getLogger("match_remaining")


def setup_logging(output_json: str, verbose: bool) -> str:
    log_path = os.path.splitext(output_json)[0] + "_remaining.log"
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("existing_matches", help="the matches json already produced (kept untouched, only added to)")
    ap.add_argument("pak_dir")
    ap.add_argument("ref_dir")
    ap.add_argument("output_json")
    ap.add_argument("--weak-below", type=float, default=DEFAULT_WEAK_BELOW,
                     help="exact-correlation scores below this are still kept, just flagged as weak for review")
    ap.add_argument("--greedy", action="store_true",
                     help="use the old fast-but-not-guaranteed per-reference top-1 approach "
                          "instead of the optimal (Hungarian) assignment")
    ap.add_argument("--max-assignment-cells", type=int, default=200_000_000,
                     help="safety cap on len(remaining_refs)*len(remaining_tracks) before the "
                          "optimal solve is attempted; falls back to --greedy automatically above it")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log_path = setup_logging(args.output_json, args.verbose)
    log.debug(f"Command: {' '.join(sys.argv)}")

    existing = json.load(open(args.existing_matches, encoding="utf-8"))
    matched_tracks = set(existing.keys())
    matched_refs = set(existing.values())
    log.info(f"Existing matches: {len(existing)} (untouched, will not be re-scored)")

    log.info(f"Indexing reference folder '{args.ref_dir}'...")
    all_refs = index_dir(args.ref_dir)
    remaining_refs = [(p, n) for p, n in all_refs if n not in matched_refs]
    log.info(f"  {len(all_refs)} total reference file(s), {len(remaining_refs)} still unmatched")

    log.info(f"Indexing pak folder '{args.pak_dir}'...")
    all_tracks = sorted(f for f in os.listdir(args.pak_dir)
                         if f.lower().endswith(".mp3") and "_part" not in f.lower())
    remaining_tracks = [t for t in all_tracks if t not in matched_tracks]
    log.info(f"  {len(all_tracks)} total whole track(s), {len(remaining_tracks)} still unclaimed")

    if not remaining_refs:
        log.info("Nothing to do - every reference already has a match.")
        json.dump(existing, open(args.output_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
        return

    if len(remaining_tracks) < len(remaining_refs):
        log.warning(f"Only {len(remaining_tracks)} track(s) left for {len(remaining_refs)} reference(s) - "
                    f"{len(remaining_refs) - len(remaining_tracks)} reference(s) CANNOT get a track no "
                    f"matter what algorithm is used (there simply aren't enough unclaimed tracks left). "
                    f"The rest will still all be assigned.")

    if not remaining_tracks:
        log.warning("No unclaimed tracks left in pak_dir - every remaining reference will go unmatched.")
        still = sorted(n for _, n in remaining_refs)
        json.dump(existing, open(args.output_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
        json.dump(still, open(os.path.splitext(args.output_json)[0] + "_still_unmatched.json", "w", encoding="utf-8"),
                   ensure_ascii=False, indent=2)
        return

    log.info(f"Decoding + fingerprinting {len(remaining_tracks)} remaining track(s)...")
    track_entries = []
    for i, fn in enumerate(remaining_tracks):
        try:
            pcm = decode_pcm(os.path.join(args.pak_dir, fn))
        except Exception:
            log.error(f"FAILED to decode '{fn}'\n" + traceback.format_exc())
            continue
        track_entries.append({"file": fn, "pcm": pcm, "fp": fingerprint(pcm)})
        if (i + 1) % 500 == 0 or (i + 1) == len(remaining_tracks):
            log.info(f"  decoded {i + 1}/{len(remaining_tracks)}")

    if not track_entries:
        log.warning("No remaining tracks could be decoded - nothing to match against.")
        still = sorted(n for _, n in remaining_refs)
        json.dump(existing, open(args.output_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2, sort_keys=True)
        json.dump(still, open(os.path.splitext(args.output_json)[0] + "_still_unmatched.json", "w", encoding="utf-8"),
                   ensure_ascii=False, indent=2)
        return

    track_fp_matrix = np.stack([t["fp"] for t in track_entries])  # (n_tracks, FINGERPRINT_BINS)

    log.info(f"Decoding + fingerprinting {len(remaining_refs)} remaining reference(s)...")
    ref_entries = []
    for i, (ref_path, ref_name) in enumerate(remaining_refs):
        try:
            pcm = decode_pcm(ref_path)
        except Exception:
            log.error(f"FAILED to decode reference '{ref_path}'\n" + traceback.format_exc())
            continue
        ref_entries.append({"name": ref_name, "path": ref_path, "pcm": pcm, "fp": fingerprint(pcm)})
    ref_fp_matrix = np.stack([r["fp"] for r in ref_entries])  # (n_refs, FINGERPRINT_BINS)

    n_refs, n_tracks = len(ref_entries), len(track_entries)
    cells = n_refs * n_tracks
    use_greedy = args.greedy or cells > args.max_assignment_cells

    forced = {}
    forced_detail = []
    still_unmatched = [n for _, n in remaining_refs if n not in {r["name"] for r in ref_entries}]

    if use_greedy:
        log.info(f"Using --greedy (per-reference top-1) assignment "
                 f"({'forced by size' if not args.greedy else 'requested'}: {n_refs}x{n_tracks}={cells} cells)")
        sims_full = ref_fp_matrix @ track_fp_matrix.T  # (n_refs, n_tracks)
        candidates = []
        for i, r in enumerate(ref_entries):
            order = np.argsort(-sims_full[i])[:32]
            best_score, best_track = -1.0, None
            for tidx in order:
                score = norm_xcorr_best_fft(r["pcm"], track_entries[tidx]["pcm"], MAX_LAG)
                if score > best_score:
                    best_score, best_track = score, track_entries[tidx]["file"]
            candidates.append((best_score, r["name"], best_track))
            log.info(f"[{i + 1}/{n_refs}] {r['name']} -> {best_track} (score {best_score:.3f})")
        candidates.sort(key=lambda c: -c[0])
        claimed = set()
        for score, ref_name, track_file in candidates:
            if track_file in claimed:
                still_unmatched.append(ref_name)
                continue
            claimed.add(track_file)
            forced[track_file] = ref_name
            forced_detail.append({"track": track_file, "ref": ref_name,
                                   "score": round(float(score), 3), "weak": score < args.weak_below})
    else:
        log.info(f"Building full similarity matrix ({n_refs} refs x {n_tracks} tracks = {cells} cells) "
                 f"and solving optimal assignment (Hungarian algorithm)...")
        sims_full = ref_fp_matrix @ track_fp_matrix.T  # (n_refs, n_tracks), cosine similarity in [-1, 1]
        cost = -sims_full  # linear_sum_assignment minimizes
        ref_idx, track_idx = linear_sum_assignment(cost)
        log.info(f"Assignment solved: {len(ref_idx)} reference(s) got a track "
                 f"(every reference, since n_tracks={n_tracks} >= n_refs={n_refs})"
                 if n_tracks >= n_refs else
                 f"Assignment solved: {len(ref_idx)} reference(s) got a track "
                 f"({n_refs - len(ref_idx)} reference(s) had no track left over to give them)")

        log.info("Computing exact confidence scores for the assigned pairs...")
        for k, (ri, ti) in enumerate(zip(ref_idx, track_idx)):
            r = ref_entries[ri]
            t = track_entries[ti]
            score = norm_xcorr_best_fft(r["pcm"], t["pcm"], MAX_LAG)
            weak = score < args.weak_below
            forced[t["file"]] = r["name"]
            forced_detail.append({"track": t["file"], "ref": r["name"],
                                   "score": round(float(score), 3), "weak": weak})
            if (k + 1) % 50 == 0 or (k + 1) == len(ref_idx):
                log.info(f"  scored {k + 1}/{len(ref_idx)}")

        assigned_ref_names = {r["name"] for r in [ref_entries[i] for i in ref_idx]}
        still_unmatched += [r["name"] for r in ref_entries if r["name"] not in assigned_ref_names]

    merged = dict(existing)
    merged.update(forced)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2, sort_keys=True)

    forced_path = os.path.splitext(args.output_json)[0] + "_forced.json"
    with open(forced_path, "w", encoding="utf-8") as f:
        json.dump(sorted(forced_detail, key=lambda d: -d["score"]), f, ensure_ascii=False, indent=2)

    still_path = os.path.splitext(args.output_json)[0] + "_still_unmatched.json"
    with open(still_path, "w", encoding="utf-8") as f:
        json.dump(sorted(set(still_unmatched)), f, ensure_ascii=False, indent=2)

    n_weak = sum(1 for d in forced_detail if d["weak"])
    log.info(f"\nDone. {len(existing)} pre-existing + {len(forced)} newly forced = {len(merged)} total -> '{args.output_json}'")
    log.info(f"  {n_weak}/{len(forced_detail)} of the forced matches are below the weak-below threshold "
             f"({args.weak_below}) - review these first")
    log.info(f"  {len(set(still_unmatched))} reference(s) still have no match "
             f"(only possible if there were fewer remaining tracks than remaining references)")
    log.info(f"Forced-match detail (with scores, worst first when reviewing): '{forced_path}'")
    log.info(f"Full log: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("match_remaining").critical("FATAL:\n" + traceback.format_exc())
        print("\nCrashed — see the .log file for details.", file=sys.stderr)
        sys.exit(1)
