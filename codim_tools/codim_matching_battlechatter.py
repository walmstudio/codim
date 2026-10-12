#!/usr/bin/env python3
"""
codim_matching_battlechatter.py — match pak tracks to battlechatter reference
files by audio content, SKIPPING every track already claimed by the voiceovers
matching (one or more existing matches json files).

Same core as codim_matching_voices_v2.py (imported, unchanged): duration
window -> 64-bin fingerprint -> exact FFT normalized cross-correlation.

Differences from v2, tuned for battlechatter (many short, similar clips):
  - --skip <json> [<json> ...]: track->ref maps whose tracks are removed from
    the pool before anything is decoded (e.g. voiceovers_final.json).
  - --fast-accept defaults to 1.02 (fingerprint-only acceptance disabled):
    a 64-bin RMS envelope can't tell short similar clips apart.
  - --shortlist defaults to 48.
  - --short-exact SEC (default 1.0): for tracks shorter than SEC the shortlist
    is ignored and EXACT correlation runs against every duration-window
    candidate (cheap for short audio, and the fingerprint is unreliable there).
  - also writes <output>_scores.json with the score of every accepted match,
    and reports references claimed by more than one track.

Usage:
  python3 codim_matching_battlechatter.py <pak_dir> <ref_dir> <output.json>
      --skip voiceovers_final.json [more.json ...]
      [--min-score 0.55] [--fast-accept 1.02] [--shortlist 48] [--short-exact 1.0]
"""

import argparse
import json
import logging
import os
import sys
import traceback
import numpy as np

from codim_matching_voices_v2 import (
    SR, MAX_LAG, DURATION_TOLERANCE, FINGERPRINT_BINS,
    decode_pcm, index_dir, fingerprint, norm_xcorr_best_fft,
)

log = logging.getLogger("match_battlechatter")


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pak_dir")
    ap.add_argument("ref_dir")
    ap.add_argument("output_json")
    ap.add_argument("--skip", nargs="+", required=True,
                    help="existing track->ref json map(s); their tracks are excluded from the pool")
    ap.add_argument("--min-score", type=float, default=0.55)
    ap.add_argument("--fast-accept", type=float, default=1.02)
    ap.add_argument("--shortlist", type=int, default=48)
    ap.add_argument("--short-exact", type=float, default=1.0,
                    help="tracks shorter than this (sec) get exact correlation against ALL duration candidates")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log_path = setup_logging(args.output_json, args.verbose)
    log.debug(f"Command: {' '.join(sys.argv)}")

    claimed = set()
    for p in args.skip:
        m = json.load(open(p, encoding="utf-8"))
        claimed.update(m.keys())
        log.info(f"Skip map '{p}': {len(m)} track(s)")
    log.info(f"Total tracks excluded from pool: {len(claimed)}")

    log.info("Indexing reference (battlechatter) files...")
    ref_files = index_dir(args.ref_dir)
    log.info(f"  found {len(ref_files)} reference file(s)")

    ref_entries, dup_names, ref_failed = [], {}, []
    for i, (path, name) in enumerate(ref_files):
        try:
            pcm = decode_pcm(path)
        except Exception:
            log.error(f"[ref {i + 1}/{len(ref_files)}] FAILED to decode '{path}'\n" + traceback.format_exc())
            ref_failed.append(path)
            continue
        ref_entries.append({"path": path, "name": name, "pcm": pcm,
                            "dur": len(pcm) / SR, "fp": fingerprint(pcm)})
        dup_names.setdefault(name, []).append(path)
        if (i + 1) % 200 == 0 or (i + 1) == len(ref_files):
            log.info(f"  decoded {i + 1}/{len(ref_files)} (failed: {len(ref_failed)})")
    if ref_failed:
        log.warning(f"{len(ref_failed)} reference(s) failed to decode and were skipped")
    coll = {n for n, p in dup_names.items() if len(p) > 1}
    if coll:
        log.warning(f"{len(coll)} reference filename(s) appear in several folders (ambiguous names)")
    if not ref_entries:
        sys.exit("no reference files decoded")

    ref_fp = np.stack([r["fp"] for r in ref_entries])
    ref_durs = np.array([r["dur"] for r in ref_entries])

    all_tracks = sorted(f for f in os.listdir(args.pak_dir)
                        if f.lower().endswith(".mp3") and "_part" not in f.lower())
    pool = [t for t in all_tracks if t not in claimed]
    log.info(f"pak: {len(all_tracks)} whole track(s), {len(pool)} left after skipping claimed ones")

    mapping, scores, ambiguous, unmatched, failed = {}, {}, {}, [], []

    for j, fn in enumerate(pool):
        log.info(f"[{j + 1}/{len(pool)}] {fn} ...")
        try:
            pcm = decode_pcm(os.path.join(args.pak_dir, fn))
        except Exception:
            log.error(f"{fn}: FAILED to decode\n" + traceback.format_exc())
            failed.append(fn)
            unmatched.append(fn)
            continue

        dur = len(pcm) / SR
        idxs = np.nonzero(np.abs(ref_durs - dur) <= DURATION_TOLERANCE)[0]
        if len(idxs) == 0:
            log.debug(f"{fn}: no duration candidates (dur={dur:.2f}s)")
            unmatched.append(fn)
            continue

        sims = ref_fp[idxs] @ fingerprint(pcm)
        order = np.argsort(-sims)

        if sims[order[0]] >= args.fast_accept:
            name = ref_entries[idxs[order[0]]]["name"]
            mapping[fn] = name
            scores[fn] = {"ref": name, "score": None, "mode": "fast"}
            log.info(f"    -> {name}  (fast, sim {sims[order[0]]:.3f})")
            continue

        if dur < args.short_exact:
            cand = idxs[order]
        else:
            cand = idxs[order[:args.shortlist]]

        try:
            scored = sorted(((norm_xcorr_best_fft(pcm, ref_entries[r]["pcm"], MAX_LAG),
                              ref_entries[r]["name"]) for r in cand), reverse=True)
        except Exception:
            log.error(f"{fn}: FAILED during correlation\n" + traceback.format_exc())
            failed.append(fn)
            unmatched.append(fn)
            continue

        best, name = scored[0]
        if best < args.min_score:
            log.debug(f"{fn}: best score {best:.3f} below threshold")
            unmatched.append(fn)
            continue

        runner = next((s for s in scored[1:] if s[1] != name), None)
        if runner and best - runner[0] < 0.05:
            ambiguous[fn] = {"top": [name, round(best, 3)], "runner_up": [runner[1], round(runner[0], 3)]}
            log.info(f"    -> {name}  (score {best:.3f}, AMBIGUOUS vs {runner[1]} {runner[0]:.3f})")
        else:
            log.info(f"    -> {name}  (score {best:.3f})")
        mapping[fn] = name
        scores[fn] = {"ref": name, "score": round(best, 3), "mode": "exact"}

    def dump(path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=isinstance(obj, dict))

    base = os.path.splitext(args.output_json)[0]
    dump(args.output_json, mapping)
    dump(base + "_scores.json", scores)
    if ambiguous:
        dump(base + "_ambiguous.json", ambiguous)
    if unmatched:
        dump(base + "_unmatched.json", unmatched)

    by_ref = {}
    for t, r in mapping.items():
        by_ref.setdefault(r, []).append(t)
    multi = {r: ts for r, ts in by_ref.items() if len(ts) > 1}
    if multi:
        dump(base + "_multi_claimed.json", multi)

    log.info(f"\nDone. {len(mapping)}/{len(pool)} pool tracks matched -> '{args.output_json}'")
    log.info(f"  ambiguous: {len(ambiguous)}   unmatched: {len(unmatched)}   crashed: {len(failed)}")
    log.info(f"  references claimed by >1 track: {len(multi)}"
             + (f" (see {base}_multi_claimed.json)" if multi else ""))
    log.info(f"  references without any track: {len(ref_entries) - len(by_ref)}")
    log.info(f"Full log: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("match_battlechatter").critical("FATAL:\n" + traceback.format_exc())
        print("\nCrashed — see the .log file for details.", file=sys.stderr)
        sys.exit(1)
