#!/usr/bin/env python3
"""
codim_audit_forced.py — audit every "forced" match from match_remaining.py
for a duration sanity mismatch between the reference clip and the ORIGINAL
(pre-recode) track it overwrote, using the .bak files left behind by
codim_recode_matched.py (unless it was run with --no-backup).

Why this matters: match_remaining.py's Hungarian assignment has NO duration
filter by design (it exists to guarantee every reference gets some track,
even a bad one), so it can pair a fraction-of-a-second clip with a track
that was originally many seconds of completely unrelated content. Recoding
that pair then destroys the original content for a wrong, tiny translation.
A >X ratio mismatch is not "maybe wrong" — it's structurally impossible to
be the right content, since duration doesn't change that much between
translations of the same line.

For every entry in <forced.json>:
  1. Find <pak_dir>/<track>.bak (the pre-recode original).
  2. Probe both the .bak's duration and the reference wav's duration.
  3. Flag it BAD if the ratio (longer/shorter) exceeds --max-ratio.
  4. With --restore, copy the .bak back over the recoded file for every
     BAD entry (undoing the damage), leaving good entries untouched.

Two modes:

  POST-recode (--restore): for when recode_matched.py has ALREADY run and
  may have overwritten unrelated content — reads each track's ORIGINAL
  duration from its <track>.bak (left behind unless --no-backup was used),
  and with --restore copies the .bak back over any BAD track, un-doing the
  damage in place.

  PRE-recode (--clean-output): for wiring into the build pipeline BEFORE
  recode_matched.py ever runs, so bad pairs never get written in the first
  place. Reads each track's duration directly from the still-untouched
  pak_dir (no .bak needed, since nothing has been overwritten yet), and
  writes a copy of --matches-json with every BAD forced pair removed —
  feed that cleaned file to recode_matched.py instead of the original.

Usage:
  # post-recode: undo damage already done
  python3 codim_audit_forced.py <forced.json> <pak_dir> <ref_dir> --restore [--max-ratio 3.0]

  # pre-recode: strip bad pairs out before they're ever written
  python3 codim_audit_forced.py <forced.json> <pak_dir> <ref_dir> \\
      --matches-json voiceovers_v2.json --clean-output voiceovers_v2_clean.json [--max-ratio 3.0]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def probe_duration(path: str):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("forced_json")
    ap.add_argument("pak_dir")
    ap.add_argument("ref_dir")
    ap.add_argument("--max-ratio", type=float, default=3.0,
                     help="flag as BAD if longer/shorter duration ratio exceeds this (default 3.0)")
    ap.add_argument("--restore", action="store_true",
                     help="POST-recode mode: copy the .bak back over every BAD track, undoing the damage")
    ap.add_argument("--matches-json",
                     help="PRE-recode mode: the full matches file (e.g. voiceovers_v2.json) to clean")
    ap.add_argument("--clean-output",
                     help="PRE-recode mode: where to write --matches-json with BAD forced pairs removed")
    args = ap.parse_args()

    pre_recode = args.matches_json is not None or args.clean_output is not None
    if pre_recode and (args.matches_json is None or args.clean_output is None):
        sys.exit("--matches-json and --clean-output must be given together (pre-recode mode)")
    if pre_recode and args.restore:
        sys.exit("--restore (post-recode) and --matches-json/--clean-output (pre-recode) are mutually exclusive")

    forced = json.load(open(args.forced_json, encoding="utf-8"))
    print(f"Auditing {len(forced)} forced match(es)...\n")

    # index ref_dir once by basename
    ref_paths = {}
    for dirpath, _, filenames in os.walk(args.ref_dir):
        for fn in filenames:
            ref_paths[fn] = os.path.join(dirpath, fn)

    bad = []
    ok = []
    missing_bak = []
    missing_ref = []

    for entry in forced:
        track = entry["track"]
        ref_name = entry["ref"]
        # pre-recode: pak_dir is still untouched, use the track itself.
        # post-recode: the track has already been overwritten, use its .bak.
        orig_source = (os.path.join(args.pak_dir, track) if pre_recode
                       else os.path.join(args.pak_dir, track + ".bak"))
        ref_path = ref_paths.get(ref_name)

        if not os.path.isfile(orig_source):
            missing_bak.append(track)
            continue
        if ref_path is None:
            missing_ref.append(ref_name)
            continue

        orig_dur = probe_duration(orig_source)
        ref_dur = probe_duration(ref_path)
        if orig_dur is None or ref_dur is None:
            missing_bak.append(track)
            continue

        ratio = max(orig_dur, ref_dur) / max(min(orig_dur, ref_dur), 1e-6)
        row = {"track": track, "ref": ref_name, "score": entry.get("score"),
               "orig_dur": round(orig_dur, 2), "ref_dur": round(ref_dur, 2), "ratio": round(ratio, 1)}
        if ratio > args.max_ratio:
            bad.append(row)
        else:
            ok.append(row)

    bad.sort(key=lambda r: -r["ratio"])

    print(f"{'ratio':>6}  {'orig(bak)':>10}  {'ref':>8}  track / ref")
    for r in bad:
        print(f"{r['ratio']:6.1f}x  {r['orig_dur']:9.2f}s  {r['ref_dur']:7.2f}s  "
              f"{r['track']}  <-  {r['ref']}  (score {r['score']})")

    print(f"\n{len(bad)} BAD (ratio > {args.max_ratio}x) — destroyed unrelated original content")
    print(f"{len(ok)} within tolerance — plausible content match despite low correlation score")
    missing_label = ".bak" if not pre_recode else "track"
    if missing_bak:
        print(f"{len(missing_bak)} {missing_label} file(s) could not be read/probed")
    if missing_ref:
        print(f"{len(missing_ref)} reference name(s) not found under {args.ref_dir}")

    if args.restore:
        print(f"\nRestoring {len(bad)} BAD track(s) from .bak...")
        for r in bad:
            track = r["track"]
            bak_path = os.path.join(args.pak_dir, track + ".bak")
            live_path = os.path.join(args.pak_dir, track)
            shutil.copy2(bak_path, live_path)
        print(f"Restored. These {len(bad)} tracks are back to their original (untranslated) audio "
              f"and should be treated as still-unmatched, not forced onto the wrong content.")

    elif pre_recode:
        bad_tracks = {r["track"] for r in bad}
        matches = json.load(open(args.matches_json, encoding="utf-8"))
        cleaned = {track: ref for track, ref in matches.items() if track not in bad_tracks}
        removed = len(matches) - len(cleaned)
        with open(args.clean_output, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"\nWrote '{args.clean_output}': {len(cleaned)} pair(s) kept, {removed} BAD pair(s) removed.")
        print(f"Those {removed} track(s) are left as their original (untranslated) audio — "
              f"feed '{args.clean_output}' to codim_recode_matched.py instead of '{args.matches_json}'.")

    elif bad:
        print(f"\nRun again with --restore (post-recode) or --matches-json/--clean-output "
              f"(pre-recode) to act on these {len(bad)} track(s).")


if __name__ == "__main__":
    main()
