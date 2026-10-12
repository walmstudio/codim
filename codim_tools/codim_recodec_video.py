#!/usr/bin/env python3
"""
bik_conform_batch.py — batch version of bik_conform.py.

Pairs files by filename between two folders:
  - ps3_dir: the PS3 originals (reference fps/audio params)
  - pc_dir:  the PC files to conform to match each PS3 counterpart
  - out_dir: where results go

Only files whose name exists in BOTH folders are processed (matched by
filename, subfolders are searched recursively but the match key is just
the basename — so ps3_dir/foo/intro.bik pairs with pc_dir/bar/intro.bik).
Files present in only one folder are reported and skipped.

For each matched pair, writes out_dir/<name>.avi — a lossless FFV1 video +
PCM audio file re-timed/resampled to the PS3 file's fps and sample rate,
ready for RAD Video Tools / Bink GUI.

If --radvideo is given, also runs RAD's own command-line encoder right
after conforming, writing out_dir/<name>.bik directly (skips the .avi
being left behind unless --keep-avi is also passed).

Usage:
  python3 bik_conform_batch.py <ps3_dir> <pc_dir> <out_dir>
  python3 bik_conform_batch.py <ps3_dir> <pc_dir> <out_dir> --radvideo "C:\\RADVideo\\radvideo64.exe"
"""

import argparse
import os
import struct
import subprocess
import sys


def read_bik_params(path: str):
    with open(path, "rb") as f:
        data = f.read(4096)
    if data[0:3] != b"BIK":
        raise ValueError(f"'{path}' does not look like a Bink file (no BIK signature)")
    fps_num, fps_den = struct.unpack_from("<II", data, 28)
    num_audio = struct.unpack_from("<I", data, 40)[0]
    if num_audio == 0:
        return {"fps": fps_num / fps_den, "sample_rate": None, "channels": None}
    _, sample_rate, flags = struct.unpack_from("<IHH", data, 44)
    channels = 2 if (flags & 0x2000) else 1
    return {"fps": fps_num / fps_den, "sample_rate": sample_rate, "channels": channels}


def index_by_name(root: str, exts=(".bik",)):
    """basename (case-insensitive) -> (display_name, full path), for every matching
    file under root (recursive). Matching is done on the lowercased filename since
    the two folders in practice differ only in case (e.g. 'aftermath_fade.bik' vs
    'Aftermath_fade.bik' are the same asset on Windows). On collision (same
    lowercased basename appearing twice under one root) keeps the first found and warns."""
    found = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                key = fn.lower()
                if key in found:
                    print(f"warning: duplicate filename '{fn}' under '{root}' "
                          f"('{found[key][1]}' vs '{os.path.join(dirpath, fn)}') — keeping the first one",
                          file=sys.stderr)
                    continue
                found[key] = (fn, os.path.join(dirpath, fn))
    return found


INTERMEDIATE_CODECS = {
    # RAD Video Tools reads the intermediate .avi through the Windows VfW/DirectShow
    # codec layer, NOT through ffmpeg's own decoders. FFV1 has no widely-installed
    # VfW codec, so RAD silently fails to read it (this is almost certainly why
    # radvideo64.exe "succeeds" but writes nothing). rawvideo is uncompressed and
    # therefore always readable by anything, at the cost of large intermediate files.
    # If you have Lagarith or UtVideo's VfW codec installed, "utvideo" is a smaller
    # lossless alternative that RAD can also open.
    "rawvideo": ["-c:v", "rawvideo"],
    "ffv1": ["-c:v", "ffv1"],
    "utvideo": ["-c:v", "utvideo"],
}


def conform_one(pc_path: str, ps3_path: str, avi_out: str, codec: str = "rawvideo") -> tuple[float, int | None]:
    ref = read_bik_params(ps3_path)
    src = read_bik_params(pc_path)
    target_fps, target_rate = ref["fps"], ref["sample_rate"]

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", pc_path,
           "-vf", f"fps={target_fps},format=yuv420p"] + INTERMEDIATE_CODECS[codec]

    if target_rate is not None and src["sample_rate"] is not None:
        # both sides have audio: resample/remux to match the PS3 reference
        cmd += ["-ar", str(target_rate), "-ac", "2", "-c:a", "pcm_s16le"]
    else:
        # reference has no audio track, or the PC source itself has none —
        # either way there is nothing meaningful to resample to, so drop audio
        if target_rate is not None and src["sample_rate"] is None:
            print(f"    note: PS3 reference expects audio ({target_rate} Hz) but "
                  f"the PC source has no audio track — writing video-only output")
        elif target_rate is None and src["sample_rate"] is not None:
            print(f"    note: PS3 reference has no audio track — dropping the PC source's "
                  f"audio, output will be video-only")
        cmd += ["-an"]

    cmd.append(avi_out)
    subprocess.run(cmd, check=True)
    return target_fps, target_rate


def run_radvideo(radvideo: str, avi_out: str, bik_out: str) -> None:
    """radvideo64.exe is a GUI app: it detaches and returns immediately instead
    of blocking until the conversion is actually done, so a plain subprocess
    call "succeeds" before the .bik has even been written. RAD's own FAQ works
    around this with `start /wait` in batch files — we do the same here."""
    if os.name == "nt":
        cmd = f'start /wait "" "{radvideo}" binkc "{avi_out}" "{bik_out}" /#'
        subprocess.run(cmd, shell=True, check=True)
    else:
        # radvideo64.exe only runs under Windows (or Wine); this branch just
        # avoids blowing up the type signature on other platforms.
        subprocess.run([radvideo, "binkc", avi_out, bik_out, "/#"], check=True)

    if not os.path.isfile(bik_out) or os.path.getsize(bik_out) == 0:
        raise RuntimeError(
            f"radvideo64.exe reported success but '{bik_out}' was not created (or is empty) — "
            f"the encode likely failed silently; check that '{avi_out}' opens fine in RAD Video Tools by hand"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ps3_dir", help="folder with PS3 original .bik files (source of truth for fps/Hz)")
    ap.add_argument("pc_dir", help="folder with PC .bik files to conform")
    ap.add_argument("out_dir", help="output folder")
    ap.add_argument("--radvideo", help="path to radvideo64.exe — if given, also encodes straight to .bik "
                                        "via 'radvideo64.exe binkc <avi> <bik> /#' for each matched pair")
    ap.add_argument("--keep-avi", action="store_true",
                     help="with --radvideo, keep the intermediate .avi instead of deleting it after encoding")
    ap.add_argument("--intermediate-codec", choices=sorted(INTERMEDIATE_CODECS), default="rawvideo",
                     help="video codec for the intermediate .avi (default: rawvideo, since RAD Video Tools "
                          "reads AVI via VfW/DirectShow and generally can't decode ffv1). Use 'utvideo' for "
                          "a smaller lossless file if UtVideo's VfW codec is installed on this machine.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ps3_files = index_by_name(args.ps3_dir)
    pc_files = index_by_name(args.pc_dir)

    common = sorted(set(ps3_files) & set(pc_files))
    only_ps3 = sorted(ps3_files[k][0] for k in set(ps3_files) - set(pc_files))
    only_pc = sorted(pc_files[k][0] for k in set(pc_files) - set(ps3_files))

    print(f"PS3 folder: {len(ps3_files)} file(s)")
    print(f"PC folder:  {len(pc_files)} file(s)")
    print(f"Matched by filename (case-insensitive): {len(common)}")
    if only_ps3:
        print(f"  only in PS3 folder (skipped): {', '.join(only_ps3)}")
    if only_pc:
        print(f"  only in PC folder (skipped): {', '.join(only_pc)}")
    if not common:
        sys.exit("nothing to do: no filenames in common between the two folders")

    ok, failed = [], []
    for i, key in enumerate(common, 1):
        ps3_name, ps3_path = ps3_files[key]
        pc_name, pc_path = pc_files[key]
        base = os.path.splitext(ps3_name)[0]  # PS3 is the "source of truth", use its casing for output
        avi_out = os.path.join(args.out_dir, base + ".avi")
        label = ps3_name if ps3_name == pc_name else f"{ps3_name} (PC: {pc_name})"
        print(f"[{i}/{len(common)}] {label} ...")
        try:
            fps, rate = conform_one(pc_path, ps3_path, avi_out, codec=args.intermediate_codec)
            rate_str = f"{rate} Hz" if rate is not None else "no audio"
            print(f"    conformed to fps={fps:.3f}, {rate_str} -> {avi_out}")

            if args.radvideo:
                bik_out = os.path.join(args.out_dir, base + ".bik")
                run_radvideo(args.radvideo, avi_out, bik_out)
                print(f"    encoded -> {bik_out}")
                if not args.keep_avi:
                    os.remove(avi_out)

            ok.append(ps3_name)
        except Exception as e:
            print(f"    FAILED: {e}", file=sys.stderr)
            failed.append(ps3_name)

    print(f"\nDone: {len(ok)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
