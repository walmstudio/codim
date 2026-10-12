#!/usr/bin/env python3
"""
recode_matched.py — re-encode the matched reference audio files (found by
codim_matching_voices_v2.py, in output.json) so each one drops into the
extracted pak folder (paepak/) as an EXACT byte-for-byte size and format
replacement for the original track it matched. This operates only on the
loose files in the extraction folder (paepak/) — never touches the .pak
itself. Run pak_tool.py's `pack` afterwards to rebuild the .pak from this
folder as usual.

Why exact size matters: downstream, other structures (.ff sound-alias
entries, sector padding, etc.) reference these tracks by absolute byte
length / offset. Changing a track's size shifts everything after it. So
this script never changes a track's file size — it only ever replaces the
audio *content* inside that fixed budget.

How a single track is rebuilt:
  1. Parse the ORIGINAL track's ID3v2 header (paepak/track_XXXX.mp3) to get
     `audio_start` (10-byte header + tag size) and the original file size.
     `target_audio_size = original_size - audio_start` is the exact number
     of bytes the new MP3 audio payload must occupy.
  2. Parse the original's first MPEG frame to read its real sample rate
     and channel mode (mono/stereo), so the replacement is encoded with
     matching parameters, not just matching size.
  3. Decode the matched reference file (output.json's value, looked up by
     filename in ref_dir) via ffmpeg, and re-encode it to MP3 at the same
     sample rate/channel mode, trying a descending list of standard CBR
     bitrates and keeping the highest one whose encoded size still fits
     inside target_audio_size (best quality that fits).
  4. Fit the encoded stream to EXACTLY target_audio_size:
       - if smaller: zero-pad the remainder (silently ignored by decoders
         that stop at the last valid frame, and harmless as trailing
         padding — same trick already used for pak sector padding).
       - if still larger even at the lowest bitrate: trim to the last
         complete MPEG frame that still fits, then zero-pad the remainder.
     Either way the output is a valid, playable MP3 of exactly the right
     length.
  5. Final file = original's untouched ID3 header/tag bytes (0..audio_start,
     byte-for-byte identical — so any PRIV/seek metadata format is kept)
     + the fitted audio payload. Written back over paepak/track_XXXX.mp3
     (a .bak copy of the original is kept unless --no-backup is given).

Usage:
  python3 recode_matched.py output.json paepak/ voiceovers/ [options]

Options:
  --no-backup           don't keep a .bak of each original file before overwriting
  --dry-run             do everything except actually write the replacement files
  --bitrates LIST        comma-separated CBR kbps candidates, high to low
                          (default: 320,256,224,192,160,128,112,96,80,64,56,48,40,32,24,16,8)
  --log PATH             where to write the detailed per-track log (default: recode.log)

Requires ffmpeg/ffprobe on PATH.
"""

import argparse
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import traceback

DEFAULT_BITRATES = [320, 256, 224, 192, 160, 128, 112, 96, 80, 64, 56, 48, 40, 32, 24, 16, 8]
# tempo (speed-up) factors tried, in order, when even the lowest bitrate
# still doesn't fit the byte budget. atempo keeps pitch unchanged; content
# is preserved, it's just played back faster (and thus takes fewer bytes
# at a given bitrate). Kept modest (<=1.5x) since much more starts to sound
# clearly sped-up; ffmpeg's atempo supports up to 2.0 per stage anyway.
DEFAULT_TEMPO_STEPS = [1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]

_MPEG_VERSIONS = {0: 2.5, 2: 2, 3: 1}  # bits '00'=2.5, '10'=2, '11'=1 (01 reserved)
_SR_TABLE = {
    1: [44100, 48000, 32000],
    2: [22050, 24000, 16000],
    2.5: [11025, 12000, 8000],
}
_CHANNEL_MODE = {0: "stereo", 1: "joint_stereo", 2: "dual_channel", 3: "mono"}

log = logging.getLogger("recode_matched")


def setup_logging(log_path, verbose):
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


# --------------------------------------------------------------------------
# ID3 / MPEG frame parsing (original track: figure out the exact byte
# budget and the encoding parameters to match)
# --------------------------------------------------------------------------

def id3_audio_start(data: bytes):
    """Returns byte offset where audio starts (10 + syncsafe tag size), or
    None if this doesn't look like a valid ID3v2 tag."""
    if len(data) < 10 or data[:3] != b"ID3" or data[3] not in (2, 3, 4):
        return None
    size_bytes = data[6:10]
    if any(b & 0x80 for b in size_bytes):
        return None
    size = 0
    for b in size_bytes:
        size = (size << 7) | (b & 0x7F)
    return 10 + size


_BR_TABLE_V1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, None]
_BR_TABLE_V2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, None]


def parse_mpeg_frame_params(data: bytes, pos: int):
    """Read one MPEG frame header at pos, return
    (sample_rate, channels, mpeg_version, bitrate_kbps) or None.

    bitrate_kbps is the ORIGINAL track's CBR bitrate. This matters beyond
    just "matching quality": the pak's ID3 PRIV frame stores a fine-grained
    per-MPEG-frame byte-offset table (see pak_tool.py's docstring) that the
    game engine appears to use for seeking/scrubbing (and possibly
    validates via the PRIV checksum). In CBR MP3, a frame's byte length is
    a pure function of (bitrate, sample_rate) — NOT of its content — so
    re-encoding at a DIFFERENT bitrate than the original silently makes
    every one of those stored offsets point into the middle of a frame
    instead of at a sync word. If the engine trusts that table, this can
    make playback stop partway through instead of hitting a clean EOF —
    matching reports of audio being "cut off" even though the file plays
    and is a valid, complete MP3 on its own. Keeping the original bitrate
    keeps every frame boundary byte-identical to the original, so the
    inherited PRIV table stays valid no matter what content is inside."""
    if pos + 4 > len(data):
        return None
    b0, b1, b2, b3 = data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None
    ver_bits = (b1 >> 3) & 3
    if ver_bits not in _MPEG_VERSIONS:
        return None
    mpeg_ver = _MPEG_VERSIONS[ver_bits]
    layer = (b1 >> 1) & 3
    if layer != 1:  # Layer III
        return None
    br_idx = (b2 >> 4) & 0xF
    sr_idx = (b2 >> 2) & 3
    if sr_idx == 3 or br_idx in (0, 15):
        return None
    sample_rate = _SR_TABLE[mpeg_ver][sr_idx]
    br_table = _BR_TABLE_V1 if mpeg_ver == 1 else _BR_TABLE_V2
    bitrate_kbps = br_table[br_idx]
    if bitrate_kbps is None:
        return None
    chan_bits = (b3 >> 6) & 3
    channels = 1 if _CHANNEL_MODE[chan_bits] == "mono" else 2
    return sample_rate, channels, mpeg_ver, bitrate_kbps


def parse_priv_region(data: bytes, audio_start: int):
    """Locate the PRIV frame inside the leading ID3 tag. Returns
    (frame_start, old_payload_size, capacity_bytes) or None if no PRIV
    frame is present. capacity_bytes is how many bytes are available for
    a NEW payload without moving audio_start at all — the frame's own
    original payload plus whatever zero-padding follows it up to
    audio_start (there's normally nothing else in this tag, see
    pak_tool.py's docstring, but we only ever touch the PRIV frame itself
    regardless)."""
    if data[:3] != b"ID3":
        return None
    fpos = 10
    while fpos < audio_start - 10:
        fid = data[fpos:fpos + 4]
        if fid == b"\x00\x00\x00\x00":
            return None
        fsize = struct.unpack(">I", data[fpos + 4:fpos + 8])[0]
        if fid == b"PRIV":
            capacity_bytes = audio_start - (fpos + 10)
            return fpos, fsize, capacity_bytes
        fpos += 10 + fsize
    return None


def priv_table_capacity(data: bytes, audio_start: int):
    """Max number of frame-offset entries a fresh PRIV table can hold
    without growing the ID3 tag (and thus without moving audio_start),
    or None if there's no PRIV frame to rebuild."""
    region = parse_priv_region(data, audio_start)
    if region is None:
        return None
    _, _, capacity_bytes = region
    return max(0, (capacity_bytes - 8) // 4)


def samples_per_frame_for(sample_rate: int) -> int:
    """MPEG1 Layer III = 1152 samples/frame; MPEG2/2.5 Layer III = 576.
    ffmpeg picks the MPEG version from the sample rate automatically."""
    return 1152 if sample_rate in (44100, 48000, 32000) else 576


def walk_frame_offsets(data: bytes, limit: int = None):
    """Walk real MPEG frames from byte 0, return the list of each frame's
    START offset (so offsets[0] == 0), stopping at the first byte that
    isn't a valid frame header (e.g. our own trailing zero padding) or at
    `limit`. This is exactly the real-audio frame table the PRIV frame
    needs to describe."""
    if limit is None:
        limit = len(data)
    offsets = []
    pos = 0
    while pos < limit:
        fl = mp3_frame_len(data, pos)
        if fl is None or fl < 21 or pos + fl > limit:
            break
        offsets.append(pos)
        pos += fl
    return offsets


def build_priv_payload(offsets: list, sample_rate: int) -> bytes:
    spf = samples_per_frame_for(sample_rate)
    checksum = len(offsets) * spf  # matches the original format exactly:
    # verified against a real extracted track where count=142, spf=576
    # and the stored "checksum" field was precisely 142*576.
    return struct.pack(">II", checksum, len(offsets)) + struct.pack(">%dI" % len(offsets), *offsets)


def rebuild_priv_header(orig: bytes, audio_start: int, offsets: list, sample_rate: int):
    """Return a new header (same total length as orig[:audio_start]) with
    the PRIV frame's payload replaced to describe `offsets`, tail
    zero-padded to fill out to the original audio_start. Raises ValueError
    if the new table doesn't fit in the space available (caller should
    keep len(offsets) <= priv_table_capacity(...) to avoid this)."""
    region = parse_priv_region(orig, audio_start)
    if region is None:
        raise ValueError("no PRIV frame found to rebuild")
    frame_start, old_payload_size, capacity_bytes = region
    new_payload = build_priv_payload(offsets, sample_rate)
    if len(new_payload) > capacity_bytes:
        raise ValueError(f"new PRIV payload ({len(new_payload)}B) exceeds available space "
                          f"({capacity_bytes}B) without moving audio_start")
    flags = orig[frame_start + 8:frame_start + 10]
    new_frame = b"PRIV" + struct.pack(">I", len(new_payload)) + flags + new_payload
    pad_len = audio_start - (frame_start + 10 + len(new_payload))
    header = orig[:frame_start] + new_frame + b"\x00" * pad_len
    assert len(header) == audio_start
    return header


def probe_duration_sec(path: str):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None
def mp3_frame_len(data: bytes, pos: int):
    if pos + 4 > len(data):
        return None
    b0, b1, b2, b3 = data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None
    ver_bits = (b1 >> 3) & 3
    if ver_bits not in _MPEG_VERSIONS:
        return None
    mpeg_ver = _MPEG_VERSIONS[ver_bits]
    layer = (b1 >> 1) & 3
    if layer != 1:
        return None
    br_idx = (b2 >> 4) & 0xF
    sr_idx = (b2 >> 2) & 3
    pad = (b2 >> 1) & 1
    if br_idx in (0, 15) or sr_idx == 3:
        return None
    if mpeg_ver == 1:
        br_table = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, None]
        base = 144
    else:
        br_table = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, None]
        base = 72
    br = br_table[br_idx]
    if br is None or br == 0:
        return None
    sr = _SR_TABLE[mpeg_ver][sr_idx]
    return base * br * 1000 // sr + pad


def last_full_frame_end(data: bytes, start: int, limit: int):
    """Walk MPEG frames from `start`, return the offset of the end of the
    last COMPLETE frame that still fits within `limit`."""
    pos = start
    last_ok = start
    while True:
        fl = mp3_frame_len(data, pos)
        if fl is None or fl < 21:
            break
        if pos + fl > limit:
            break
        pos += fl
        last_ok = pos
    return last_ok


# --------------------------------------------------------------------------
# ffmpeg helpers
# --------------------------------------------------------------------------

def _atempo_chain(factor: float) -> str:
    """atempo only accepts 0.5-2.0 per instance; chain stages for anything
    outside that (not needed at our modest factors, but safe either way)."""
    stages = []
    f = factor
    while f > 2.0:
        stages.append(2.0)
        f /= 2.0
    while f < 0.5:
        stages.append(0.5)
        f /= 0.5
    stages.append(f)
    return ",".join(f"atempo={s:.6f}" for s in stages)


def encode_mp3(src_path: str, sample_rate: int, channels: int, bitrate_kbps: int,
                tempo: float = 1.0) -> bytes:
    """Decode src_path and re-encode to a CBR MP3 with given params.
    tempo > 1.0 speeds the audio up (pitch-preserving) via atempo, used to
    shrink a clip's byte size WITHOUT cutting any content off the end.
    Returns just the raw bytes ffmpeg produced - caller extracts the audio
    payload."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src_path]
    if tempo != 1.0:
        cmd += ["-filter:a", _atempo_chain(tempo)]
    cmd += [
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-codec:a", "libmp3lame",
        "-b:a", f"{bitrate_kbps}k",
        "-write_xing", "0",
        "-id3v2_version", "0",
        "-f", "mp3", "pipe:1",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(f"ffmpeg encode failed ({bitrate_kbps}kbps, tempo={tempo}): "
                            f"{p.stderr.decode(errors='replace')[:400]}")
    return p.stdout


def strip_leading_id3(buf: bytes) -> bytes:
    """ffmpeg with id3v2_version 0 shouldn't write a leading tag, but be
    defensive in case it (or a future ffmpeg default) does."""
    start = id3_audio_start(buf)
    return buf[start:] if start is not None else buf


# --------------------------------------------------------------------------
# core per-track logic
# --------------------------------------------------------------------------

def index_ref_dir(ref_dir: str):
    idx = {}
    dupes = set()
    for dirpath, _, filenames in os.walk(ref_dir):
        for fn in filenames:
            if fn in idx:
                dupes.add(fn)
            else:
                idx[fn] = os.path.join(dirpath, fn)
    if dupes:
        log.warning(f"{len(dupes)} filename(s) appear more than once in ref_dir "
                     f"(first occurrence used): {sorted(dupes)[:10]}"
                     + (" ..." if len(dupes) > 10 else ""))
    return idx


def fit_to_size(audio_payload: bytes, target_size: int, allow_trim: bool):
    """Return (fitted_bytes, note) — audio_payload padded to exactly
    target_size bytes. Only trims (cutting audio content off the end) if
    allow_trim is True; otherwise raises if it's still too big, so the
    caller can try speeding the audio up instead of ever cutting it."""
    if len(audio_payload) == target_size:
        return audio_payload, "exact"
    if len(audio_payload) < target_size:
        pad = target_size - len(audio_payload)
        return audio_payload + b"\x00" * pad, f"padded +{pad}B"
    if not allow_trim:
        raise ValueError(f"{len(audio_payload)}B still exceeds {target_size}B target "
                          f"and trimming is disabled")
    # too big: trim to the last complete frame that fits, then zero-pad
    cut = last_full_frame_end(audio_payload, 0, target_size)
    fitted = audio_payload[:cut] + b"\x00" * (target_size - cut)
    trimmed = len(audio_payload) - cut
    return fitted, f"trimmed -{trimmed}B (frame-aligned) then padded +{target_size - cut}B"


def process_track(track_fn: str, ref_name: str, paepak_dir: str, ref_index: dict,
                   bitrates: list, tempo_steps: list, allow_trim: bool,
                   backup: bool, dry_run: bool) -> dict:
    orig_path = os.path.join(paepak_dir, track_fn)
    if not os.path.isfile(orig_path):
        return {"track": track_fn, "status": "error", "reason": "original track file missing"}

    orig = open(orig_path, "rb").read()
    audio_start = id3_audio_start(orig)
    if audio_start is None:
        return {"track": track_fn, "status": "error", "reason": "original has no valid ID3 header"}

    orig_size = len(orig)
    target_audio_size = orig_size - audio_start
    if target_audio_size <= 0:
        return {"track": track_fn, "status": "error", "reason": "no audio payload in original (empty)"}

    params = parse_mpeg_frame_params(orig, audio_start)
    if params is None:
        return {"track": track_fn, "status": "error", "reason": "could not read original's MPEG frame header"}
    sample_rate, channels, mpeg_ver, orig_bitrate = params

    ref_path = ref_index.get(ref_name)
    if ref_path is None:
        return {"track": track_fn, "status": "error", "reason": f"reference file '{ref_name}' not found in ref_dir"}

    # The pak's inherited ID3 tag carries a PRIV frame: a fine-grained
    # per-MPEG-frame byte-offset table (see pak_tool.py's docstring), which
    # the game engine appears to use for seeking/scrubbing. It only
    # describes the ORIGINAL audio's frame layout — if we drop in different
    # content without updating it, the table is nonsense for the new
    # stream. Worse: the original source is very often VBR (bitrate varies
    # frame-to-frame), so there's no bitrate we could pick that would keep
    # old offsets valid anyway — the table has to be regenerated to match
    # whatever we actually encode, every time.
    #
    # That table lives in a FIXED number of bytes (the original PRIV
    # frame's payload, at most reused up to the original audio_start) so it
    # can only hold so many frame entries without moving audio_start. Frame
    # COUNT for a given duration doesn't depend on bitrate at all (frame
    # time span = samples_per_frame/sample_rate is fixed) — only the
    # spoken-audio duration matters. So if the reference line is too long
    # for the table's capacity, no bitrate choice fixes it: only speeding
    # the audio up (tempo) shortens it enough to fit.
    priv_capacity = priv_table_capacity(orig, audio_start)
    spf = samples_per_frame_for(sample_rate)
    max_duration = (priv_capacity * spf / sample_rate) if priv_capacity else None

    ref_duration = probe_duration_sec(ref_path)
    if ref_duration is None:
        return {"track": track_fn, "status": "error", "reason": f"could not probe duration of '{ref_name}'"}

    all_tempos = sorted(set([1.0] + list(tempo_steps)))
    if max_duration is not None:
        eligible_tempos = [t for t in all_tempos if ref_duration / t <= max_duration + 1e-6]
    else:
        eligible_tempos = all_tempos  # no PRIV frame found — can't enforce/fix this constraint

    priv_frame_trimmed = 0  # frames chopped off the END purely to fit table capacity (allow_trim path)
    if not eligible_tempos:
        if not allow_trim:
            return {"track": track_fn, "status": "error",
                    "reason": f"reference is {ref_duration:.2f}s, but the inherited PRIV seek "
                              f"table only has room for {priv_capacity} frames "
                              f"({max_duration:.2f}s) even at {all_tempos[-1]}x tempo — "
                              f"pass --allow-trim to cut the tail instead of skipping, or "
                              f"raise --tempo-steps"}
        eligible_tempos = [all_tempos[-1]]  # best effort: fastest available, trim the rest below

    # Bitrate is now free to pick purely for quality (highest that fits the
    # byte budget) — PRIV safety no longer depends on matching the
    # original's bitrate, since we rebuild the table from whatever we
    # actually encode.
    chosen = None  # (bitrate, tempo, payload)
    smallest_overshoot = None
    for tempo in eligible_tempos:
        for br in bitrates:
            try:
                raw = encode_mp3(ref_path, sample_rate, channels, br, tempo=tempo)
            except Exception as e:
                log.debug(f"{track_fn}: encode at {br}kbps tempo={tempo} failed: {e}")
                continue
            payload = strip_leading_id3(raw)
            if len(payload) <= target_audio_size:
                chosen = (br, tempo, payload)
                break
            if smallest_overshoot is None or len(payload) < len(smallest_overshoot[2]):
                smallest_overshoot = (br, tempo, payload)
        if chosen is not None:
            break

    trimmed_used = False
    if chosen is None:
        if not allow_trim:
            return {"track": track_fn, "status": "error",
                    "reason": f"doesn't fit even at {bitrates[-1]}kbps + "
                              f"{eligible_tempos[-1]}x tempo, and trimming is disabled "
                              f"(--allow-trim to permit it)"}
        if smallest_overshoot is None:
            return {"track": track_fn, "status": "error", "reason": "all encode attempts failed"}
        chosen = smallest_overshoot
        trimmed_used = True

    br, tempo, payload = chosen
    fitted, fit_note = fit_to_size(payload, target_audio_size, allow_trim=trimmed_used)
    assert len(fitted) == target_audio_size

    # Regenerate the PRIV table to describe the REAL frames actually in
    # `fitted` (this naturally excludes any trailing zero-padding, since
    # that isn't a valid frame sync and walk_frame_offsets stops there).
    priv_note = ""
    if priv_capacity is not None:
        offsets = walk_frame_offsets(fitted)
        if len(offsets) > priv_capacity:
            # only reachable via the allow_trim "best effort" path above —
            # cut the extra frames' worth of audio and re-pad, so the table
            # we write always matches what's really there.
            priv_frame_trimmed = len(offsets) - priv_capacity
            offsets = offsets[:priv_capacity]
            cut_at = offsets[-1] if offsets else 0
            # find the real end of the last kept frame, not just its start
            fl = mp3_frame_len(fitted, cut_at) or 0
            fitted = fitted[:cut_at + fl] + b"\x00" * (target_audio_size - (cut_at + fl))
            fit_note += f", +table-trimmed {priv_frame_trimmed} frame(s) to fit PRIV capacity"
        header = rebuild_priv_header(orig, audio_start, offsets, sample_rate)
        priv_note = f"{len(offsets)}/{priv_capacity} frames"
        priv_safe = True
    else:
        header = orig[:audio_start]  # unchanged — nothing we can safely rebuild
        priv_safe = False
        priv_note = "no PRIV frame found, left untouched"

    new_file = header + fitted
    assert len(new_file) == orig_size

    if not dry_run:
        if backup:
            bak_path = orig_path + ".bak"
            if not os.path.exists(bak_path):
                shutil.copy2(orig_path, bak_path)
        with open(orig_path, "wb") as f:
            f.write(new_file)

    return {
        "track": track_fn,
        "status": "ok",
        "ref": ref_name,
        "sample_rate": sample_rate,
        "channels": channels,
        "orig_bitrate_kbps": orig_bitrate,
        "bitrate_kbps": br,
        "tempo": tempo,
        "priv_safe": priv_safe,
        "priv_table": priv_note,
        "target_audio_size": target_audio_size,
        "fit": fit_note,
        "final_size": len(new_file),
    }


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matches_json", help="output.json from codim_matching_voices_v2.py (track_fn -> ref_fn)")
    ap.add_argument("paepak_dir", help="extracted pak folder (files are overwritten in place here)")
    ap.add_argument("ref_dir", help="folder containing the replacement reference audio files")
    ap.add_argument("--no-backup", action="store_true", help="don't keep .bak copies of overwritten originals")
    ap.add_argument("--dry-run", action="store_true", help="do everything except write files")
    ap.add_argument("--bitrates", default=",".join(str(b) for b in DEFAULT_BITRATES),
                     help="comma-separated CBR kbps candidates, high to low")
    ap.add_argument("--tempo-steps", default=",".join(str(t) for t in DEFAULT_TEMPO_STEPS),
                     help="comma-separated speed-up factors tried (pitch-preserving) when no "
                          "bitrate fits at normal speed, so the full line still plays without "
                          "cutting anything off the end (default: %(default)s)")
    ap.add_argument("--allow-trim", action="store_true",
                     help="last-resort: if even the slowest bitrate + fastest tempo step still "
                          "doesn't fit the byte budget, OR the reference is too long for the "
                          "inherited PRIV seek table's capacity even at max tempo, cut audio off "
                          "the end (frame-aligned) instead of skipping the track. OFF by default.")
    ap.add_argument("--log", default="recode.log")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.log, args.verbose)
    log.debug(f"Command: {' '.join(sys.argv)}")

    bitrates = [int(x) for x in args.bitrates.split(",") if x.strip()]
    tempo_steps = [float(x) for x in args.tempo_steps.split(",") if x.strip()]

    matches = json.load(open(args.matches_json, encoding="utf-8"))
    log.info(f"Loaded {len(matches)} matched track(s) from '{args.matches_json}'")

    log.info(f"Indexing reference folder '{args.ref_dir}'...")
    ref_index = index_ref_dir(args.ref_dir)
    log.info(f"  found {len(ref_index)} unique reference filename(s)")

    if args.dry_run:
        log.info("DRY RUN — no files will be modified")

    results = []
    ok = 0
    errors = 0
    for i, (track_fn, ref_name) in enumerate(sorted(matches.items())):
        log.info(f"[{i + 1}/{len(matches)}] {track_fn} <- {ref_name} ...")
        try:
            r = process_track(track_fn, ref_name, args.paepak_dir, ref_index,
                               bitrates, tempo_steps, allow_trim=args.allow_trim,
                               backup=not args.no_backup, dry_run=args.dry_run)
        except Exception:
            log.error(f"{track_fn}: CRASHED\n" + traceback.format_exc())
            r = {"track": track_fn, "status": "error", "reason": "exception, see log"}
        results.append(r)
        if r["status"] == "ok":
            ok += 1
            tempo_note = f", tempo x{r['tempo']:.2f}" if r["tempo"] != 1.0 else ""
            priv_note = f"  [PRIV: {r['priv_table']}]" if r.get("priv_safe") else "  [!] no PRIV table rebuilt"
            log.info(f"    -> {r['bitrate_kbps']}kbps {r['sample_rate']}Hz "
                      f"{'mono' if r['channels'] == 1 else 'stereo'}{tempo_note}, "
                      f"{r['fit']}, final size {r['final_size']}B (target was {r['target_audio_size']}B audio)"
                      f"{priv_note}")
        else:
            errors += 1
            log.warning(f"    -> SKIPPED: {r['reason']}")

    report_path = os.path.splitext(args.log)[0] + "_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    priv_broken = [r["track"] for r in results if r["status"] == "ok" and not r.get("priv_safe", True)]

    log.info(f"\nDone. {ok}/{len(matches)} recoded successfully, {errors} skipped/failed.")
    if priv_broken:
        log.warning(f"{len(priv_broken)} track(s) had no PRIV frame to rebuild at all (left as-is) — "
                     f"these were never protected by any bitrate trick either, old or new: "
                     + ", ".join(priv_broken[:20]) + (" ..." if len(priv_broken) > 20 else ""))
    log.info(f"Per-track report: {report_path}")
    log.info(f"Full log: {args.log}")
    if ok and not args.dry_run:
        log.info(f"Next: run pak_tool.py pack on '{args.paepak_dir}' to rebuild the .pak.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("recode_matched").critical("FATAL:\n" + traceback.format_exc())
        print("\nCrashed — see the .log file for details.", file=sys.stderr)
        sys.exit(1)
