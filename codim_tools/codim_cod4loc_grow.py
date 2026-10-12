#!/usr/bin/env python3
"""
codim_cod4loc_grow.py — port LocalizeEntry values from a PC (Russian) .ff into
a PS3 .ff WITHOUT the "port_safe" length limitation: values that are LONGER
than the original are no longer truncated/left-in-English.

Why this works (see README's "Что нужно для полного перевода" section for
the problem statement) — the short version:

  Pointers inside an IWffu100 zone come in two forms:
    - 0xFFFFFFFF            : the real data follows immediately, inline.
    - (block<<29|offset)+1  : an explicit reference into one of the 7 named
                               memory pools (Temp/Physical/Runtime/Virtual/
                               Large/Callback/Vertex), where `offset` is
                               relative to THAT POOL'S OWN START, not to the
                               file/zone as a whole.

  LocalizeEntry is `{ char* value; char* name; }`, 8 bytes, and on disk the
  inline strings (when used) are just value-text + NUL + name-text + NUL
  packed immediately after those two pointer slots, entries placed back to
  back. It lives in the Large pool.

  cod4loc.py's plain `import`/`port` grows a value STRAIGHT IN PLACE, so
  everything after it in the whole 40MB zone shifts - and every pointer
  ANYWHERE that referenced something past that point, in ANY pool, now
  reads a stale offset. That's the crash.

  port_safe.py avoids the crash by never growing anything, so it's safe but
  a longer Russian value just doesn't fit and gets left in English.

  The fix used here: never touch the original bytes for an entry that needs
  a longer value at all. Instead, leave the old (English) inline text sitting
  there completely undisturbed - dead, unreferenced bytes - and APPEND the
  new value + the same key name to the very END of the Large pool's own
  data (not the end of the file, the end of the LARGE POOL specifically,
  i.e. right where the Callback/Vertex pools currently start). Then flip
  that entry's two pointers from 0xFFFFFFFF/0xFFFFFFFF to explicit
  pointers at the new location.

  Because the insertion happens exactly at the Large pool's own boundary
  (never in its middle), nothing that was already in the Large pool has
  its pool-relative offset change, so no other Large-pool (block=4)
  pointer needs touching. Everything physically after Large (Callback,
  Vertex) shifts to a later ABSOLUTE file position, but since those pools'
  own pointers are relative to THEIR OWN pool start (not file-absolute),
  and we correctly bump the Large pool's size field in the header, their
  internal references keep resolving correctly with zero changes to the
  pointer words themselves. Verified end-to-end on the real common.ff:
  after inserting bytes into Large, an untouched block=6 (Vertex) pointer
  still decodes to byte-identical target content.

  Net result: only the header's ZoneSize and Large pool-size fields change,
  plus the touched entries' own two pointer words. Nothing else in the
  40MB zone is modified. This is NOT a general pointer-relocation pass (it
  doesn't need to be, for this specific kind of edit) and does not attempt
  to handle materials/techsets/images growing - only LocalizeEntry values.

Limitations / what this does NOT (yet) handle:
  - ~126 of ~3025 localize entries in common.ff already use an explicit
    (non-inline) value pointer themselves (shared values, e.g. many
    SUBTITLE_* keys reusing one line's text). This tool leaves those
    completely alone and reports them separately - what they actually
    point to isn't confirmed yet (real cross-references into Large-pool
    content, decoded exactly like the ones this tool creates, but the
    matching source entries weren't identified during this investigation).
    Their PC-side translation is simply skipped for now.
  - Only tested with careful manual verification of the pointer
    arithmetic and one synthetic insertion, NOT in a real PS3 environment
    or emulator. Treat the first real output.ff as a build to test
    carefully (font rendering, menu navigation, a save/load cycle) before
    trusting it further - if anything is subtly wrong about the pool
    boundary math on a zone with nonzero Physical/Runtime/Virtual pools
    (common.ff/code_post_gfx.ff both happen to have those at 0), this is
    where it would show up.

Usage:
  python3 codim_cod4loc_grow.py ps3.ff pc_ru.ff out.ff
      [--enc cp1251] [--skip PLATFORM_,XBOXLIVE_] [--report report.tsv]
"""
import sys
import struct
import argparse

sys.path.insert(0, "codim_tools/codim_tools")  # adjust if run from elsewhere
from codim_cod4loc import ff_to_zone, zone_to_ff, PAT, PAT_ANY  # reuse, unmodified

HEADER_SIZE = 0x34
POOL_NAMES = ["Temp", "Physical", "Runtime", "Virtual", "Large", "Callback", "Vertex"]
LARGE_BLOCK = POOL_NAMES.index("Large")  # 4


def find_pool_bounds(z: bytes):
    """Parse the 52-byte zone header + tag table + asset-pool table to find
    the byte range, within the decompressed zone, of each of the 7 named
    memory pools. Returns (bounds_dict, sizes_dict) where bounds_dict[name]
    = (start, end) absolute offsets into `z`."""
    vals = struct.unpack(">13I", z[:52])
    script_string_count = vals[9]
    asset_count = vals[11]
    pool_sizes = dict(zip(POOL_NAMES, vals[2:9]))

    o = HEADER_SIZE
    while o < len(z) and (z[o] == 0x00 or z[o] == 0xFF):
        o += 1
    offset = o
    for _ in range(script_string_count):
        end = z.index(b"\x00", offset)
        offset = end + 1

    ap = offset
    while ap < len(z):
        b0, b1, b2, b3, b4, b5, b6 = z[ap:ap + 7]
        if b0 == 0xFF and b1 == 0xFF and b2 == 0xFF and b3 == 0xFF and b4 == 0 and b5 == 0 and b6 == 0:
            break
        ap += 1
    pool_data_start = ap + asset_count * 8

    bounds = {}
    running = pool_data_start
    for name in POOL_NAMES:
        s = pool_sizes[name]
        bounds[name] = (running, running + s)
        running += s
    return bounds, pool_sizes


def encode_ptr(block: int, offset: int) -> int:
    return ((block << 29) | offset) + 1


def do_grow(ps3_path, pc_path, out_path, enc, skip):
    z = bytearray(ff_to_zone(ps3_path))
    bounds, pool_sizes = find_pool_bounds(bytes(z))
    large_start, large_end = bounds["Large"]

    # source (PC) values by key - byte-for-byte, same convention as
    # codim_cod4loc.py's do_port (PAT_ANY tolerates 0xFF inside the value)
    src = {}
    for m in PAT_ANY.finditer(bytes(ff_to_zone(pc_path))):
        src[m.group(2).decode()] = m.group(1)

    matches = list(PAT.finditer(bytes(z)))  # only fully-inline PS3 entries
    print(f"PS3 fully-inline localize entries: {len(matches)}")
    print(f"PC reference entries: {len(src)}")

    in_place = 0
    appended = 0
    missing = 0
    skipped = 0
    unchanged = 0
    append_blob = bytearray()
    report_lines = []

    # process in reverse file order so in-place edits (which don't change
    # length) never disturb positions we haven't processed yet
    for m in reversed(matches):
        key = m.group(2).decode()
        old_value = bytes(m.group(1))
        if key not in src or any(key.startswith(s) for s in skip if s):
            if key not in src:
                missing += 1
            else:
                skipped += 1
            continue

        new_value_enc = src[key]  # already raw bytes, PC's own encoding/content
        if new_value_enc == old_value:
            unchanged += 1
            continue

        if len(new_value_enc) <= len(old_value):
            # fits in the original slot: write in place, pad with trailing
            # spaces to keep every byte position after it identical - zero
            # risk, same approach as port_safe.py.
            pad = len(old_value) - len(new_value_enc)
            z[m.start() + 8: m.start() + 8 + len(old_value)] = new_value_enc + b" " * pad
            in_place += 1
            report_lines.append(f"{key}\tin-place\t{len(old_value)}->{len(new_value_enc)}+{pad}pad")
        else:
            # doesn't fit: leave the old inline bytes untouched (dead), and
            # queue (new_value + key) to be appended at the true end of the
            # Large pool once we've finished the reverse pass.
            append_blob_entry = new_value_enc + b"\x00" + m.group(2) + b"\x00"
            offset_in_large = (large_end - large_start) + len(append_blob)
            append_blob.extend(append_blob_entry)
            ptr1 = encode_ptr(LARGE_BLOCK, offset_in_large)
            ptr2 = encode_ptr(LARGE_BLOCK, offset_in_large + len(new_value_enc) + 1)
            z[m.start():m.start() + 4] = struct.pack(">I", ptr1)
            z[m.start() + 4:m.start() + 8] = struct.pack(">I", ptr2)
            appended += 1
            report_lines.append(f"{key}\tappended\t{len(old_value)}->{len(new_value_enc)}")

    if append_blob:
        z[large_end:large_end] = append_blob
        added = len(append_blob)
        z[0:4] = struct.pack(">I", struct.unpack(">I", z[0:4])[0] + added)
        large_off = 8 + 4 * LARGE_BLOCK
        z[large_off:large_off + 4] = struct.pack(">I", pool_sizes["Large"] + added)
        print(f"Appended {added} byte(s) to the Large pool "
              f"({pool_sizes['Large']} -> {pool_sizes['Large'] + added})")

    zone_to_ff(z, out_path)

    print(f"\nin-place (fit, padded): {in_place}")
    print(f"appended (grew, repointed): {appended}")
    print(f"unchanged (PC == PS3 already): {unchanged}")
    print(f"no PC reference found: {missing}")
    print(f"skipped (--skip prefix): {skipped}")
    print(f"126-ish shared-value entries (explicit ptr already, untouched): "
          f"{3025 - len(matches) if False else 'see note below'}")
    print(f"NOTE: PS3 asset-pool declares more localize entries than the "
          f"{len(matches)} fully-inline ones this tool can see/edit - those "
          f"already use a shared explicit pointer and are left as-is.")
    print(f"-> {out_path}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"Per-key report: {args.report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ps3_ff")
    ap.add_argument("pc_ru_ff")
    ap.add_argument("out_ff")
    ap.add_argument("--enc", default="cp1251")
    ap.add_argument("--skip", default="", help="comma-separated key prefixes to leave in English")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    skip = [s for s in args.skip.split(",") if s]
    do_grow(args.ps3_ff, args.pc_ru_ff, args.out_ff, args.enc, skip)
