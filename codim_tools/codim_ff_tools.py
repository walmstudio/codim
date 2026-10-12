#!/usr/bin/env python3
"""
ff_tool.py - CoD4 (IW3) PS3 .ff FastFile: decompress/recompress + extract/repack
the embedded ID3v2+PRIV+MP3 sound clusters (sfx/ambient/dialogue) found inline
inside the decompressed zone data.

Format notes (reverse-engineered, PS3 CoD4/WaW):
  - .ff container: 8-byte magic "IWffu100" + 4-byte version (BE), then a stream
    of raw-deflate blocks, each: 2-byte big-endian compressed size + that many
    bytes of raw deflate data (no zlib header, wbits=-15). Decompress each block
    independently and concatenate -> the "raw zone".
  - Somewhere inside the raw zone there can be a cluster of back-to-back audio
    entries, each starting with a normal ID3v2.3 tag (with a custom PRIV frame
    holding a seek/loop table) immediately followed by a raw MP3 stream. There
    is no explicit index/table for this cluster - entries just run back to back
    until the pattern stops. We find them by scanning for valid ID3v2 headers.

Commands:
    unpack  <in.ff> <outdir>          decompress + extract all audio entries + manifest.json
    repack  <outdir> <in.ff> <out.ff> rebuild a .ff from an unpack outdir (uses manifest.json)
    raw-decompress <in.ff> <out.bin>  just dump the decompressed zone
    raw-compress   <in.bin> <out.ff>  just recompress a zone dump back into a .ff

Usage examples:
    python3 ff_tool.py unpack ambush.ff ambush_unpacked/
    # ... replace/edit ambush_unpacked/audio/track_003.mp3 ...
    python3 ff_tool.py repack ambush_unpacked/ ambush.ff ambush_modded.ff
"""

import sys
import os
import json
import struct
import zlib
import hashlib

BLOCK_SIZE = 65536
MAGIC = b"IWffu100"


# --------------------------------------------------------------------------
# Container-level (de)compression: IWffu100 raw-deflate block stream
# --------------------------------------------------------------------------

def decompress_ff(path):
    """Returns (version_bytes, raw_zone_bytes)."""
    data = open(path, "rb").read()
    if data[:8] != MAGIC:
        raise ValueError(f"Not an unsigned CoD4/WaW PS3 .ff (bad magic {data[:8]!r})")
    version = data[8:12]

    pos = 12
    out = bytearray()
    while pos < len(data):
        if pos + 2 > len(data):
            break
        size = struct.unpack(">H", data[pos:pos + 2])[0]
        pos += 2
        if size == 0:
            break
        chunk = data[pos:pos + size]
        pos += size
        d = zlib.decompressobj(-15)
        dec = d.decompress(chunk)
        dec += d.flush()
        out += dec

    return version, bytes(out)


def compress_ff(version, raw_zone, out_path):
    """Recompress raw zone bytes into a valid IWffu100 .ff, same block scheme."""
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(version)
        for i in range(0, len(raw_zone), BLOCK_SIZE):
            block = raw_zone[i:i + BLOCK_SIZE]
            co = zlib.compressobj(9, zlib.DEFLATED, -15)
            comp = co.compress(block) + co.flush()
            if len(comp) > 0xFFFF:
                # Extremely unlikely (block is only 64KB), but guard anyway.
                raise ValueError("compressed block exceeds 65535 bytes")
            f.write(struct.pack(">H", len(comp)))
            f.write(comp)


# --------------------------------------------------------------------------
# ID3v2 scanning inside the raw zone
# --------------------------------------------------------------------------

def id3_tag_size(hdr10):
    """Syncsafe 4-byte size field (bytes 6-9), excludes the 10-byte header itself."""
    b = hdr10
    return ((b[6] & 0x7F) << 21) | ((b[7] & 0x7F) << 14) | ((b[8] & 0x7F) << 7) | (b[9] & 0x7F)


def find_id3_entries(data):
    """
    Scan the whole buffer for byte-perfect ID3v2 tag starts ("ID3" + sane
    version/flags), then group consecutive hits that are close enough together
    into one cluster (real clusters have entries packed tightly; a lone false
    "ID3" match inside unrelated binary data will be isolated / have garbage
    version bytes and gets filtered out here).

    Returns list of absolute offsets (start of "ID3") that are the real
    cluster of audio entries, in order.
    """
    import re
    candidates = [m.start() for m in re.finditer(b"ID3", data)]

    valid = []
    for off in candidates:
        hdr = data[off:off + 10]
        if len(hdr) < 10:
            continue
        ver_major, ver_minor = hdr[3], hdr[4]
        flags = hdr[5]
        if ver_major in (2, 3, 4) and ver_minor == 0 and flags in (0, 0x80):
            valid.append(off)

    if not valid:
        return []

    # Keep only the largest tightly-packed run (real cluster). "Tight" = each
    # tag is followed by another real tag within a generous window relative
    # to its own declared size, OR gaps stay small. We use a simple adaptive
    # rule: build runs where consecutive gap < max(1_000_000, 4 * tag_size).
    runs = []
    current = [valid[0]]
    for prev, cur in zip(valid, valid[1:]):
        hdr = data[prev:prev + 10]
        sz = id3_tag_size(hdr)
        gap_limit = max(1_000_000, sz * 8 + 100_000)
        if cur - prev <= gap_limit:
            current.append(cur)
        else:
            runs.append(current)
            current = [cur]
    runs.append(current)

    best = max(runs, key=len)
    return best


def compute_track_bounds(data, entry_offsets):
    """
    Given ordered ID3 start offsets, the last track's true end is found by
    walking forward from the last tag until MP3 sync clearly breaks down (or
    a sane cap), since there's no following tag to bound it.
    Returns list of (start, end) tuples.
    """
    bounds = []
    for i in range(len(entry_offsets) - 1):
        bounds.append((entry_offsets[i], entry_offsets[i + 1]))

    if entry_offsets:
        last = entry_offsets[-1]
        end = find_track_end(data, last)
        bounds.append((last, end))

    return bounds


_BITRATES_V1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, None]
_BITRATES_V2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, None]
_SR_V1 = [44100, 48000, 32000, None]
_SR_V2 = [22050, 24000, 16000, None]
_SR_V25 = [11025, 12000, 8000, None]


def mp3_frame_len(data, pos):
    if pos + 4 > len(data):
        return None
    b0, b1, b2, b3 = data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None
    ver = (b1 >> 3) & 3       # 0=2.5, 1=reserved, 2=MPEG2, 3=MPEG1
    layer = (b1 >> 1) & 3     # 1=Layer3
    if ver == 1 or layer != 1:
        return None
    br_idx = (b2 >> 4) & 0xF
    sr_idx = (b2 >> 2) & 3
    pad = (b2 >> 1) & 1
    if br_idx in (0, 15) or sr_idx == 3:
        return None
    if ver == 3:
        br = _BITRATES_V1[br_idx]
        sr = _SR_V1[sr_idx]
        base = 144
    elif ver == 2:
        br = _BITRATES_V2[br_idx]
        sr = _SR_V2[sr_idx]
        base = 72
    else:  # ver == 0, MPEG2.5
        br = _BITRATES_V2[br_idx]
        sr = _SR_V25[sr_idx]
        base = 72
    if br is None or sr is None or br == 0:
        return None
    return base * br * 1000 // sr + pad


def find_track_end(data, tag_off, max_scan=8_000_000):
    hdr = data[tag_off:tag_off + 10]
    sz = id3_tag_size(hdr)
    pos = tag_off + 10 + sz
    limit = min(len(data), tag_off + max_scan)
    last_good = pos
    while pos + 4 <= limit:
        fl = mp3_frame_len(data, pos)
        if fl is None or fl < 21:
            break
        pos += fl
        last_good = pos
    return last_good


# --------------------------------------------------------------------------
# unpack / repack
# --------------------------------------------------------------------------

def cmd_unpack(ff_path, outdir):
    version, raw = decompress_ff(ff_path)
    print(f"[unpack] decompressed zone: {len(raw)} bytes (version={version.hex()})")

    entries = find_id3_entries(raw)
    print(f"[unpack] found {len(entries)} audio entries")

    os.makedirs(outdir, exist_ok=True)
    audio_dir = os.path.join(outdir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    bounds = compute_track_bounds(raw, entries)

    manifest = {
        "source_file": os.path.basename(ff_path),
        "version_hex": version.hex(),
        "zone_size": len(raw),
        "zone_sha256": hashlib.sha256(raw).hexdigest(),
        "tracks": [],
    }

    for i, (start, end) in enumerate(bounds):
        blob = raw[start:end]
        fname = f"track_{i:04d}.mp3"
        with open(os.path.join(audio_dir, fname), "wb") as f:
            f.write(blob)
        manifest["tracks"].append({
            "index": i,
            "file": fname,
            "zone_offset": start,
            "orig_size": len(blob),
        })

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Also keep the full raw zone dump so repack can splice against untouched
    # bytes for everything outside the audio cluster.
    with open(os.path.join(outdir, "zone_original.bin"), "wb") as f:
        f.write(raw)

    print(f"[unpack] wrote {len(bounds)} tracks to {audio_dir}")
    print(f"[unpack] manifest: {os.path.join(outdir, 'manifest.json')}")


def cmd_repack(outdir, orig_ff_path, out_ff_path):
    manifest_path = os.path.join(outdir, "manifest.json")
    zone_path = os.path.join(outdir, "zone_original.bin")
    if not os.path.exists(manifest_path) or not os.path.exists(zone_path):
        raise FileNotFoundError(
            "repack needs manifest.json and zone_original.bin from a prior 'unpack' run in outdir"
        )

    manifest = json.load(open(manifest_path, encoding="utf-8"))
    raw = bytearray(open(zone_path, "rb").read())

    if hashlib.sha256(raw).hexdigest() != manifest["zone_sha256"]:
        print("[repack] WARNING: zone_original.bin does not match manifest hash "
              "(was it edited?) - continuing anyway")

    version = bytes.fromhex(manifest["version_hex"])
    audio_dir = os.path.join(outdir, "audio")

    tracks = sorted(manifest["tracks"], key=lambda t: t["zone_offset"])

    # Splice from the end backwards so earlier offsets stay valid as we edit
    # the bytearray, in case any replacement changes length.
    resized_count = 0
    for t in reversed(tracks):
        fpath = os.path.join(audio_dir, t["file"])
        new_blob = open(fpath, "rb").read()
        start = t["zone_offset"]
        end = start + t["orig_size"]

        if len(new_blob) != t["orig_size"]:
            resized_count += 1
            print(f"[repack] track {t['index']} ({t['file']}) size changed: "
                  f"{t['orig_size']} -> {len(new_blob)} bytes "
                  f"(downstream zone offsets for OTHER assets will shift - "
                  f"only safe if this is the last/only thing referencing "
                  f"absolute positions, or if the engine doesn't care)")

        raw[start:end] = new_blob

    if resized_count:
        print(f"[repack] {resized_count} track(s) changed size - "
              f"round-trip is no longer guaranteed byte-identical to the "
              f"original layout past the first resized track. Proceeding on "
              f"a best-effort basis, as requested.")

    compress_ff(version, bytes(raw), out_ff_path)
    print(f"[repack] wrote {out_ff_path} ({os.path.getsize(out_ff_path)} bytes)")


def cmd_raw_decompress(ff_path, out_path):
    version, raw = decompress_ff(ff_path)
    with open(out_path, "wb") as f:
        f.write(raw)
    print(f"[raw-decompress] {ff_path} -> {out_path} ({len(raw)} bytes, version={version.hex()})")


def cmd_raw_compress(bin_path, out_path, version_hex="00000001"):
    raw = open(bin_path, "rb").read()
    version = bytes.fromhex(version_hex)
    compress_ff(version, raw, out_path)
    print(f"[raw-compress] {bin_path} -> {out_path} ({os.path.getsize(out_path)} bytes)")


# --------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "unpack" and len(args) == 2:
        cmd_unpack(args[0], args[1])
    elif cmd == "repack" and len(args) == 3:
        cmd_repack(args[0], args[1], args[2])
    elif cmd == "raw-decompress" and len(args) == 2:
        cmd_raw_decompress(args[0], args[1])
    elif cmd == "raw-compress" and len(args) == 2:
        cmd_raw_compress(args[0], args[1])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
