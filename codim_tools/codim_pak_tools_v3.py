#!/usr/bin/env python3
"""
pak_tool.py — unpacker/packer for CoD (IW-engine) sound .pak containers,
with an integrated "split check" pass on every extracted track.

Format (reverse-engineered):
  - Whole file is a sequence of 2048-byte sectors.
  - Sector 0 is a header block: 4-byte magic + zero padding to 2048 bytes.
  - Starting at sector 1 (offset 2048), each entry is one MP3 track:
      * begins with an ID3v2.3 tag (with a custom PRIV frame holding a
        fine-grained seek/scrub table — one entry roughly every MP3 frame,
        NOT one entry per spoken line, see notes below), immediately
        followed by the raw MP3 stream.
      * the whole entry (tag + audio) is zero-padded to a multiple of
        2048 bytes — except the very last entry in the file.
  - Entries are placed back-to-back with no gaps other than the padding.

Split-check pass (runs automatically after every track is written, unless
--no-split-check is given):
  Some tracks are actually several spoken lines glued back-to-back by the
  original game asset (not an artifact of extraction). For each track:
    1. Get its real duration via ffprobe.
    2. Skip the check entirely if duration >= --ambient-threshold seconds
       (default 120s) — long tracks are ambient/music loops with thousands
       of natural pauses; running silencedetect on those is slow and
       produces meaningless "splits".
    3. Otherwise run ffmpeg silencedetect to find real pauses.
    4. Convert each pause's midpoint to a byte offset using this track's
       own *actual* average byte-rate (audio_size / real_duration — robust
       for VBR, unlike trusting the first frame's bitrate field, which can
       be a Xing/LAME placeholder).
    5. Snap that byte offset to the nearest entry in the track's own PRIV
       table, landing exactly on an MP3 frame boundary.
    6. If this produces 2+ parts, slice the raw MP3 bytes (no re-encode —
       safe because the cuts are frame-aligned) and write them alongside
       the whole track as track_XXXX_partNN.mp3. The whole (unsplit) track
       file is always kept too, so `pack` round-trips byte-identically
       without needing to know about parts at all — the split files are
       purely an extra, convenient output for the voice-matching step.
  Each track's manifest entry gets a "kind" tag: "ambient" (skipped, too
  long), "single" (checked, no real split found), or "multi" (split into
  N parts, listed under "split_parts").

Usage:
  python3 pak_tool.py extract <input.pak> <output_dir> [split-check options]
  python3 pak_tool.py pack    <input_dir> <output.pak> [--magic HEXBYTES]
"""

import argparse
import bisect
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np

SECTOR = 2048
MAGIC_DEFAULT = bytes.fromhex("6F16D4A1")

AMBIENT_THRESHOLD_SEC = 120.0
SILENCE_NOISE_DB = -30.0
SILENCE_MIN_SEC = 0.12
MIN_PART_SEC = 0.25

# "tail merge" check: a short part immediately following a cut can actually
# be the decaying reverb/echo tail of the PREVIOUS line rather than a new
# replica's onset. Real onsets rise from near-silence; tails start already
# fairly loud and just decay. Only checked for parts shorter than
# TAIL_CHECK_MAX_SEC (longer parts are extremely unlikely to be a stray tail).
TAIL_CHECK_MAX_SEC = 1.0
TAIL_FIRST_WINDOW_RATIO = 0.15  # first 20ms-window loudness vs the part's own peak
TAIL_ENVELOPE_SR = 8000
TAIL_ENVELOPE_WIN_MS = 20

_SILENCE_RE = re.compile(
    r"silence_start:\s*([\d.]+)|silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)"
)


def _id3_tag_size(data: bytes, pos: int):
    """Parse a syncsafe ID3v2 header at pos; return total tag length
    (header + frames) or None if the header looks malformed."""
    if pos + 10 > len(data):
        return None
    if data[pos:pos + 3] != b"ID3":
        return None
    if data[pos + 3] not in (2, 3, 4):
        return None
    size_bytes = data[pos + 6:pos + 10]
    if any(b & 0x80 for b in size_bytes):
        return None
    size = 0
    for b in size_bytes:
        size = (size << 7) | (b & 0x7F)
    return 10 + size


def _looks_like_mpeg_sync(data: bytes, pos: int) -> bool:
    if pos + 2 > len(data):
        return False
    return data[pos] == 0xFF and (data[pos + 1] & 0xE0) == 0xE0


def find_valid_id3_starts(data: bytes, search_from: int = SECTOR) -> list:
    """Scan the WHOLE file for ID3 tags, not just sector boundaries — some
    entries (battlechatter clusters) are packed back-to-back with no sector
    padding at all, so a sector-only scan silently glues many real clips
    into one giant 'track'. Each candidate is validated by checking that a
    real MPEG frame sync immediately follows the parsed tag, which rejects
    accidental 3-byte 'ID3' matches inside unrelated binary data."""
    starts = []
    pos = data.find(b"ID3", search_from)
    while pos != -1:
        total = _id3_tag_size(data, pos)
        if total is not None and _looks_like_mpeg_sync(data, pos + total):
            starts.append(pos)
            pos = data.find(b"ID3", pos + total)
        else:
            pos = data.find(b"ID3", pos + 3)
    return starts


def parse_id3_priv(data: bytes):
    """Returns (audio_start, priv_offsets). priv_offsets is the list of N
    byte offsets (relative to audio_start, always starting at 0) stored in
    the track's PRIV frame, or None if no usable PRIV table was found.
    PRIV payload layout: [4B checksum][4B count N][N x 4B BE offsets]."""
    if data[:3] != b"ID3":
        return None, None

    tag_size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | \
               ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
    audio_start = 10 + tag_size

    fpos = 10
    priv_offsets = None
    while fpos < audio_start - 10:
        fid = data[fpos:fpos + 4]
        if fid == b"\x00\x00\x00\x00":
            break
        fsize = struct.unpack(">I", data[fpos + 4:fpos + 8])[0]
        payload = data[fpos + 10:fpos + 10 + fsize]
        if fid == b"PRIV" and len(payload) >= 8 and len(payload) % 4 == 0:
            n = len(payload) // 4
            vals = struct.unpack(">%dI" % n, payload)
            count = vals[1]
            if 2 + count == n:
                priv_offsets = list(vals[2:2 + count])
        fpos += 10 + fsize

    return audio_start, priv_offsets


def probe_duration_sec(path: str):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(out.stdout.strip())
    except (ValueError, OSError):
        return None


def detect_silence_gaps(path: str, noise_db: float, min_silence: float):
    cmd = ["ffmpeg", "-i", path, "-af",
           f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gaps = []
    pending_start = None
    for line in p.stderr.splitlines():
        m = _SILENCE_RE.search(line)
        if not m:
            continue
        if m.group(1) is not None:
            pending_start = float(m.group(1))
        elif m.group(2) is not None and pending_start is not None:
            gaps.append((pending_start, float(m.group(2))))
            pending_start = None
    return gaps


def snap_to_nearest(offsets, target):
    i = bisect.bisect_left(offsets, target)
    if i == 0:
        return offsets[0]
    if i == len(offsets):
        return offsets[-1]
    before, after = offsets[i - 1], offsets[i]
    return before if (target - before) <= (after - target) else after


def _decode_pcm(mp3_bytes: bytes, sr: int = TAIL_ENVELOPE_SR):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        input=mp3_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if not p.stdout:
        return np.array([], dtype=np.float32)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32)


def looks_like_trailing_tail(mp3_bytes: bytes) -> bool:
    """True if this short clip's loudness envelope peaks almost immediately
    and already starts loud — the signature of catching the decaying tail
    of the PREVIOUS line rather than a fresh onset (real replicas rise from
    near-silence toward their peak)."""
    pcm = _decode_pcm(mp3_bytes)
    win = int(TAIL_ENVELOPE_SR * TAIL_ENVELOPE_WIN_MS / 1000)
    n = len(pcm) // win
    if n < 3:
        return False  # too short to judge, don't merge blindly
    trimmed = pcm[:n * win].reshape(n, win)
    env = np.sqrt((trimmed ** 2).mean(axis=1))
    peak = env.max()
    if peak <= 1e-6:
        return False
    first_win_ratio = env[0] / peak
    return first_win_ratio > TAIL_FIRST_WINDOW_RATIO


def merge_trailing_tails(data: bytes, audio_start: int, boundaries: list,
                          bytes_per_sec: float) -> list:
    """Post-process pass over computed part boundaries: for every part
    (except the very first) shorter than TAIL_CHECK_MAX_SEC, check whether
    it looks like a trailing decay tail of the part before it; if so, drop
    that boundary so the two parts merge back into one."""
    result = [boundaries[0]]
    for j in range(1, len(boundaries) - 1):
        seg_start, seg_end = boundaries[j], boundaries[j + 1]
        seg_dur = (seg_end - seg_start) / bytes_per_sec
        if seg_dur < TAIL_CHECK_MAX_SEC:
            chunk = data[audio_start + seg_start: audio_start + seg_end]
            if looks_like_trailing_tail(chunk):
                continue  # drop this boundary: merges seg into the previous part
        result.append(seg_start)
    result.append(boundaries[-1])
    return result


def split_check_track(track_path: str, out_dir: str,
                       ambient_threshold: float, noise_db: float,
                       min_silence: float, min_part_sec: float) -> dict:
    """Analyzes one already-written track file; writes partNN.mp3 files
    alongside it if it splits. Returns a dict to merge into its manifest
    entry: {"kind": ..., "duration_sec": ..., "split_parts": [...]}."""
    duration = probe_duration_sec(track_path)
    if duration is None:
        return {"kind": "unknown", "duration_sec": None}

    if duration >= ambient_threshold:
        return {"kind": "ambient", "duration_sec": round(duration, 3)}

    with open(track_path, "rb") as f:
        data = f.read()
    audio_start, priv_offsets = parse_id3_priv(data)
    if audio_start is None:
        return {"kind": "single", "duration_sec": round(duration, 3)}
    audio_size = len(data) - audio_start
    if audio_size <= 0:
        return {"kind": "single", "duration_sec": round(duration, 3)}

    bytes_per_sec = audio_size / duration

    gaps = detect_silence_gaps(track_path, noise_db, min_silence)
    cut_times = [(s + e) / 2.0 for s, e in gaps if s > 0.05]
    if not cut_times:
        return {"kind": "single", "duration_sec": round(duration, 3)}

    cut_byte_offsets = []
    for t in cut_times:
        approx = t * bytes_per_sec
        snapped = snap_to_nearest(priv_offsets, approx) if priv_offsets else int(approx)
        cut_byte_offsets.append(snapped)

    min_part_bytes = min_part_sec * bytes_per_sec
    boundaries = [0]
    for off in cut_byte_offsets:
        if off - boundaries[-1] >= min_part_bytes:
            boundaries.append(off)
    boundaries.append(audio_size)
    if boundaries[-1] - boundaries[-2] < min_part_bytes and len(boundaries) > 2:
        boundaries.pop(-2)

    if len(boundaries) > 2:
        boundaries = merge_trailing_tails(data, audio_start, boundaries, bytes_per_sec)

    if len(boundaries) <= 2:
        return {"kind": "single", "duration_sec": round(duration, 3)}

    base = os.path.splitext(os.path.basename(track_path))[0]
    parts = []
    for i in range(len(boundaries) - 1):
        start = audio_start + boundaries[i]
        end = audio_start + boundaries[i + 1]
        chunk = data[start:end]
        part_name = f"{base}_part{i:02d}.mp3"
        with open(os.path.join(out_dir, part_name), "wb") as f:
            f.write(chunk)
        parts.append({
            "file": part_name,
            "size": len(chunk),
            "duration_sec": round(len(chunk) / bytes_per_sec, 3),
        })

    return {
        "kind": "multi",
        "duration_sec": round(duration, 3),
        "cuts_sec": [round(t, 3) for t in cut_times],
        "split_parts": parts,
    }


def extract(pak_path: str, out_dir: str, split_check: bool,
            ambient_threshold: float, noise_db: float,
            min_silence: float, min_part_sec: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    filesize = os.path.getsize(pak_path)

    with open(pak_path, "rb") as f:
        full_data = f.read()

    if len(full_data) < SECTOR:
        raise ValueError("file too small to contain header sector")
    magic = full_data[:4]

    starts = find_valid_id3_starts(full_data, search_from=SECTOR)

    if not starts:
        raise ValueError("no valid ID3 entries found after header sector")

    starts.append(filesize)

    manifest = {"magic": magic.hex(), "sector": SECTOR, "tracks": []}

    n_total = len(starts) - 1
    counts = {"ambient": 0, "single": 0, "multi": 0, "unknown": 0}

    for i in range(n_total):
        entry_start = starts[i]
        entry_end = starts[i + 1]
        data = full_data[entry_start:entry_end]

        name = f"track_{i:04d}.mp3"
        track_path = os.path.join(out_dir, name)
        with open(track_path, "wb") as out:
            out.write(data)

        entry = {"file": name, "size": len(data)}

        if split_check:
            info = split_check_track(
                track_path, out_dir, ambient_threshold, noise_db,
                min_silence, min_part_sec,
            )
            entry.update(info)
            counts[info.get("kind", "unknown")] += 1

            if (i + 1) % 50 == 0 or (i + 1) == n_total:
                print(f"  checked {i + 1}/{n_total} "
                      f"(ambient={counts['ambient']} single={counts['single']} "
                      f"multi={counts['multi']} unknown={counts['unknown']})")

        manifest["tracks"].append(entry)

    with open(os.path.join(out_dir, "manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Extracted {len(manifest['tracks'])} track(s) to '{out_dir}'")
    print(f"Header magic: {magic.hex()}")
    if split_check:
        print(f"Split-check: {counts['single']} single, {counts['multi']} multi "
              f"({sum(len(t.get('split_parts', [])) for t in manifest['tracks'])} total parts written), "
              f"{counts['ambient']} ambient/skipped, {counts['unknown']} unknown")


def pack(in_dir: str, pak_path: str, magic_hex) -> None:
    """
    IMPORTANT FIX (vs the original pack()): no padding is inserted BETWEEN
    tracks here anymore.

    find_valid_id3_starts() scans the *whole* original file for ID3 tags,
    not just sector boundaries. So during extract(), each written track
    file is exactly `data[entry_start:entry_end]` where entry_end is the
    NEXT real tag's start — meaning any gap that existed in the original
    (anywhere from 0 bytes, for the tightly-packed battlechatter/VO
    cluster, up to 2047 bytes of zero padding, for the sector-aligned
    ambient/music tracks) is already baked into that track's own file
    length. Re-adding `pad = (-len(data)) % SECTOR` here as the old code
    did double-pads every sector-aligned track (harmless, since pad=0
    there) but WRONGLY inserts up to 2047 extra bytes after every
    non-sector-aligned track, growing the file and shifting the absolute
    byte offset of every track after it — which breaks any external
    structure that references these tracks by absolute offset (e.g. the
    .ff sound-alias pointers that index straight into this .pak).

    So: just concatenate the header and the tracks, in order, with nothing
    in between. If every track file's length is unchanged from extraction
    (recode_matched.py guarantees this), the result is byte-identical to
    the original .pak.
    """
    manifest_path = os.path.join(in_dir, "manifest.json")
    tracks = None
    magic = MAGIC_DEFAULT

    if os.path.isfile(manifest_path):
        with open(manifest_path) as mf:
            manifest = json.load(mf)
        # pack always uses the WHOLE track file per entry, never the split
        # parts — split_parts exist purely as convenience output for the
        # voice-matching step and are ignored here, so round-trip stays
        # byte-identical to the original regardless of split-check.
        tracks = [t["file"] for t in manifest["tracks"]]
        magic = bytes.fromhex(manifest["magic"])
    else:
        tracks = sorted(
            fn for fn in os.listdir(in_dir)
            if fn.lower().endswith(".mp3") and "_part" not in fn.lower()
        )

    if magic_hex:
        magic = bytes.fromhex(magic_hex)

    if not tracks:
        raise ValueError(f"no tracks found in '{in_dir}'")

    n_resized = 0
    with open(pak_path, "wb") as out:
        header = magic + b"\x00" * (SECTOR - len(magic))
        out.write(header)

        for i, name in enumerate(tracks):
            path = os.path.join(in_dir, name)
            with open(path, "rb") as f:
                data = f.read()

            if data[:3] != b"ID3":
                print(f"warning: '{name}' does not start with an ID3 tag", file=sys.stderr)

            if os.path.isfile(manifest_path):
                orig_size = manifest["tracks"][i]["size"]
                if len(data) != orig_size:
                    n_resized += 1
                    print(f"warning: '{name}' is {len(data)}B but manifest says {orig_size}B — "
                          f"every track after this one will land at a shifted absolute offset",
                          file=sys.stderr)

            # NOTE: deliberately no inter-track padding written here — see
            # the docstring above. Each track's own bytes already include
            # whatever gap (if any) originally followed it.
            out.write(data)

    if n_resized:
        print(f"WARNING: {n_resized} track(s) changed size vs the manifest — "
              f"absolute offsets inside this .pak have shifted for everything "
              f"after the first one of them.")
    print(f"Packed {len(tracks)} track(s) into '{pak_path}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="unpack a .pak into individual .mp3 files, with automatic split-check")
    p_ex.add_argument("input", help="input .pak file")
    p_ex.add_argument("outdir", help="output directory")
    p_ex.add_argument("--no-split-check", action="store_true", help="skip the per-track split-check pass (faster, plain extraction only)")
    p_ex.add_argument("--ambient-threshold", type=float, default=AMBIENT_THRESHOLD_SEC, help=f"tracks with duration >= this many seconds are treated as ambient/music and skipped (default {AMBIENT_THRESHOLD_SEC})")
    p_ex.add_argument("--noise", type=float, default=SILENCE_NOISE_DB, help=f"silencedetect noise floor in dB (default {SILENCE_NOISE_DB})")
    p_ex.add_argument("--min-silence", type=float, default=SILENCE_MIN_SEC, help=f"minimum silence duration in seconds (default {SILENCE_MIN_SEC})")
    p_ex.add_argument("--min-part", type=float, default=MIN_PART_SEC, help=f"minimum resulting part duration in seconds, shorter cuts are merged (default {MIN_PART_SEC})")

    p_pk = sub.add_parser("pack", help="repack a directory of whole track .mp3 files into a .pak (split parts are ignored)")
    p_pk.add_argument("indir", help="input directory (with manifest.json or plain .mp3 files)")
    p_pk.add_argument("output", help="output .pak file")
    p_pk.add_argument("--magic", help="8 hex chars to use as the 4-byte header magic (default: 6F16D4A1)")

    args = ap.parse_args()

    if args.cmd == "extract":
        extract(
            args.input, args.outdir,
            split_check=not args.no_split_check,
            ambient_threshold=args.ambient_threshold,
            noise_db=args.noise,
            min_silence=args.min_silence,
            min_part_sec=args.min_part,
        )
    elif args.cmd == "pack":
        pack(args.indir, args.output, args.magic)


if __name__ == "__main__":
    main()
