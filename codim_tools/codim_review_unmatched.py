#!/usr/bin/env python3
"""
codim_review_unmatched.py — find out WHICH reference (PC) files still have no
pak track, look at the best candidate tracks for each one, listen, and pick
by ear. One tool, three subcommands:

  list    what is still unmatched (and which forced pairs look like noise)
  review  per reference: rank candidate tracks, play them, choose (or export
          everything to a folder if you'd rather listen in your own player)
  apply   merge your choices into a matches json (feed the result to recode)

Typical flow (run on the pristine paepak, BEFORE recode):

  python3 codim_review_unmatched.py list  <ref_dir> voiceovers_v2_clean.json \\
          --forced voiceovers_v2_forced.json -o unmatched_refs.json
  python3 codim_review_unmatched.py review unmatched_refs.json paepak/ voiceovers_v2_clean.json
  python3 codim_review_unmatched.py apply voiceovers_v2_clean.json review_choices.json voiceovers_final.json

Notes
  * A pair is track -> ref. One ref may have SEVERAL tracks (the same line
    stored twice in the pak) — in `review`, accept several: `a 1 3`.
  * Candidates are ranked by the same exact normalized cross-correlation the
    matcher uses (no fingerprint pre-filter, so short words are not lost),
    over every track whose duration is close to the reference's. Durations
    come from paepak/manifest.json (no ffprobe pass over 17k files).
  * In `review`, if none of the shown candidates fits, type `m` to reveal the
    next batch (another --top, default 10) — numbers keep counting (11, 12, ...),
    and you can repeat `m` until the duration window is exhausted.
  * Decoded audio is cached in .review_cache/ (per file, keyed by size+mtime),
    so re-running review is fast.
  * choices file: {"ref.wav": ["track_1.mp3", ...]}  -> use these tracks
                  {"ref.wav": null}                  -> no match in the pak (drop current pair)
                  {"ref.wav": []}                    -> undecided, ignored by apply

Requires ffmpeg / ffprobe / ffplay on PATH, numpy.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SR = 8000
MIN_OVERLAP = int(SR * 0.05)
AUDIO_EXTS = {".wav", ".wave", ".mp3"}
CACHE_DIR_DEFAULT = ".review_cache"


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------

def probe_duration(path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    out = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def decode_pcm_i16(path):
    # -nostdin + stdin=DEVNULL: by default ffmpeg reads the terminal's stdin
    # for interactive keys ('q' quits it!) and would swallow what you type
    # at the review prompt.
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def cached_pcm(path, cache_dir):
    st = os.stat(path)
    key = hashlib.md5(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
    cp = os.path.join(cache_dir, key + ".npy")
    if os.path.isfile(cp):
        try:
            return np.load(cp)
        except Exception:
            pass
    pcm = decode_pcm_i16(path)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cp, pcm)
    return pcm


def best_xcorr(a, b, max_lag):
    """Best normalized cross-correlation over lags in [-max_lag, +max_lag].
    Same definition as norm_xcorr_best_fft in codim_matching_voices_v2.py
    (globally-centered signals, per-overlap energy normalization) but the
    per-lag loop is vectorized with numpy, so a wide lag search is cheap."""
    n = min(len(a), len(b))
    if n < MIN_OVERLAP:
        return 0.0
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    a0 = a - a.mean()
    b0 = b - b.mean()

    fsize = 1
    while fsize < 2 * n:
        fsize *= 2
    xc = np.fft.irfft(np.fft.rfft(a0, fsize) * np.conj(np.fft.rfft(b0, fsize)), fsize)
    A2 = np.concatenate(([0.0], np.cumsum(a0 * a0)))
    B2 = np.concatenate(([0.0], np.cumsum(b0 * b0)))

    L = min(max_lag, n - MIN_OVERLAP)
    lags = np.arange(-L, L + 1)
    al = np.abs(lags)
    pos = lags >= 0
    ea = np.where(pos, A2[n] - A2[al], A2[n - al] - A2[0])
    eb = np.where(pos, B2[n - al] - B2[0], B2[n] - B2[al])
    denom = np.sqrt(ea * eb)
    idx = np.where(pos, lags, fsize + lags)
    valid = denom > 1e-6
    if not valid.any():
        return 0.0
    scores = xc[idx][valid] / denom[valid]
    return max(float(scores.max()), 0.0)


def play(path):
    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                       stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("  (ffplay не найден — он входит в ffmpeg)")
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------
# pool of pak tracks (durations from manifest.json)
# --------------------------------------------------------------------------

def load_pool(pak_dir, include_parts):
    """name -> duration for every candidate track in pak_dir."""
    pool = {}
    mpath = os.path.join(pak_dir, "manifest.json")
    if os.path.isfile(mpath):
        manifest = json.load(open(mpath, encoding="utf-8"))
        for t in manifest.get("tracks", []):
            if t.get("duration_sec") is not None:
                pool[t["file"]] = float(t["duration_sec"])
            if include_parts:
                for p in t.get("split_parts", []):
                    if p.get("duration_sec") is not None:
                        pool[p["file"]] = float(p["duration_sec"])
        pool = {k: v for k, v in pool.items() if os.path.isfile(os.path.join(pak_dir, k))}
        return pool

    print(f"[!] {mpath} не найден — беру длительности через ffprobe (медленно для больших паков)")
    for fn in sorted(os.listdir(pak_dir)):
        if not fn.lower().endswith(".mp3"):
            continue
        if not include_parts and "_part" in fn.lower():
            continue
        d = probe_duration(os.path.join(pak_dir, fn))
        if d is not None:
            pool[fn] = d
    return pool


def find_candidates(ref_pcm, ref_dur, pool, pak_dir, cache_dir, args, extra=()):
    """Returns (ranked [(score, name, dur)], n_in_window). `extra` = track
    names that must be scored even if outside the duration window."""
    tol = max(args.tol_abs, args.tol_rel * ref_dur)
    window = [(n, d) for n, d in pool.items() if abs(d - ref_dur) <= tol]
    window.sort(key=lambda x: abs(x[1] - ref_dur))
    window = window[:args.cap]
    have = {n for n, _ in window}
    for n in extra:
        if n in pool and n not in have:
            window.append((n, pool[n]))

    max_lag = int(SR * args.max_lag_ms / 1000)
    ref_f = ref_pcm.astype(np.float32)

    def work(item):
        name, dur = item
        try:
            pcm = cached_pcm(os.path.join(pak_dir, name), cache_dir)
        except Exception:
            return (0.0, name, dur)
        return (best_xcorr(ref_f, pcm.astype(np.float32), max_lag), name, dur)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(work, window))
    results.sort(key=lambda r: -r[0])
    return results, len(window)


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------

def index_refs(ref_dir):
    refs = {}
    for dirpath, _, files in os.walk(ref_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS and fn not in refs:
                refs[fn] = os.path.join(dirpath, fn)
    return refs


def cmd_list(args):
    matches = json.load(open(args.matches, encoding="utf-8"))
    refs = index_refs(args.ref_dir)
    matched = set(matches.values())
    ref_to_tracks = defaultdict(list)
    for t, r in matches.items():
        ref_to_tracks[r].append(t)

    entries = []
    for name in sorted(n for n in refs if n not in matched):
        entries.append({"ref": name, "path": refs[name], "duration": probe_duration(refs[name]),
                        "reason": "unmatched"})

    n_unmatched = len(entries)
    n_weak = 0
    if args.forced:
        forced = json.load(open(args.forced, encoding="utf-8"))
        for f in sorted(forced, key=lambda f: f["score"]):
            r = f["ref"]
            if r in matched and f["score"] < args.weak_below and r in refs:
                entries.append({"ref": r, "path": refs[r], "duration": probe_duration(refs[r]),
                                "reason": "weak", "score": f["score"], "current": ref_to_tracks[r]})
                n_weak += 1

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)

    print(f"Референсов в {args.ref_dir}: {len(refs)}")
    print(f"Сопоставлено (уникальных ref в {args.matches}): {len(matched & set(refs))}")
    print(f"БЕЗ ПАРЫ: {n_unmatched}")
    if args.forced:
        print(f"С подозрительной парой (forced, score < {args.weak_below}): {n_weak}")

    def group(name):
        return "pain/death/melee" if re.search(r"(pain|death|melee)", name, re.I) else "диалоги/прочее"
    groups = defaultdict(list)
    for e in entries:
        groups[(e["reason"], group(e["ref"]))].append(e["ref"])
    print()
    for (reason, g), names in sorted(groups.items()):
        print(f"  {reason:<9} {g:<18} {len(names):>4}   напр.: {', '.join(names[:3])}")
    print(f"\nСписок записан: {args.output}")
    print(f"Дальше: python3 {sys.argv[0]} review {args.output} <pak_dir> {args.matches}")


# --------------------------------------------------------------------------
# review (interactive) / export
# --------------------------------------------------------------------------

def load_choices(path):
    if os.path.isfile(path):
        return json.load(open(path, encoding="utf-8"))
    return {}


def save_choices(path, choices):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(choices, fh, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def build_top(cands, shown, current):
    """First `shown` ranked candidates, plus the current (suspicious) pair
    appended at the end if it ranks lower, so it always stays visible."""
    top = cands[:shown]
    for c in cands:
        if c[1] in current and c not in top:
            top.append(c)
    return top


def print_candidates(cands, ref_dur, claimed, current, shown):
    print(f"   {'#':>2} {'score':>6}  {'track':<24} {'dur':>7} {'Δdur':>7}  статус")
    for i, (score, name, dur) in enumerate(cands[:shown], 1):
        if name in current:
            status = "ТЕКУЩАЯ ПАРА (подозрительная)"
        elif name in claimed:
            status = f"занят: {claimed[name]}"
        else:
            status = "свободен"
        print(f"   {i:>2} {score:6.3f}  {name:<24} {dur:6.2f}s {dur - ref_dur:+6.2f}s  {status}")


HELP = """  команды:
    r            переслушать референс
    <номер>      прослушать кандидата (например: 1)
    m            (more) если эти не подходят — показать следующие кандидаты (ещё --top штук, нумерация продолжается)
    a <номера>   принять кандидата(ов), можно несколько: a 1 3 (одна фраза лежит в паке дважды)
    n            (именно буква n) в паке этого нет; текущая пара, если была, будет снята
    s            пропустить (решишь потом)
    q            сохранить и выйти"""


def cmd_review(args):
    entries = json.load(open(args.refs, encoding="utf-8"))
    only = set(args.only or [])
    if only:
        known = {e["ref"] for e in entries}
        extra_names = [n for n in only if n not in known]
        if extra_names:
            if not args.ref_dir:
                sys.exit(f"{extra_names} нет в {args.refs}; укажи --ref-dir, чтобы найти их там")
            idx = index_refs(args.ref_dir)
            for n in extra_names:
                if n not in idx:
                    sys.exit(f"'{n}' не найден в {args.ref_dir}")
                entries.append({"ref": n, "path": idx[n], "duration": probe_duration(idx[n]),
                                "reason": "forced-by-hand"})
        entries = [e for e in entries if e["ref"] in only]

    matches = json.load(open(args.matches, encoding="utf-8"))
    claimed = dict(matches)  # track -> ref
    choices = load_choices(args.choices)

    if not args.export and not args.redo:
        entries = [e for e in entries if e["ref"] not in choices]

    if not entries:
        print("Нечего разбирать (все уже в choices; --redo чтобы пройти заново).")
        return

    print(f"Загружаю пул треков из {args.pak_dir} ...")
    pool = load_pool(args.pak_dir, args.include_parts)
    if not pool:
        n_wav = sum(1 for f in os.listdir(args.pak_dir) if f.lower().endswith((".wav", ".wave"))) \
            if os.path.isdir(args.pak_dir) else 0
        hint = (" В этой папке лежат .wav — похоже, это папка с РЕФЕРЕНСАМИ; нужна распакованная папка пака (paepak/)."
                if n_wav else "")
        sys.exit(f"В {args.pak_dir} нет mp3-треков (и нет manifest.json).{hint}\n"
                 f"Порядок аргументов: review <unmatched_refs.json> <paepak/> <matches.json>")
    print(f"  {len(pool)} треков в пуле. К разбору: {len(entries)} реф(ов).\n")

    export_root = args.export
    if export_root:
        os.makedirs(export_root, exist_ok=True)
        template = {}

    for i, e in enumerate(entries, 1):
        ref, rpath = e["ref"], e["path"]
        rdur = e.get("duration") or probe_duration(rpath)
        if rdur is None:
            print(f"[{i}/{len(entries)}] {ref}: не удалось прочитать длительность, пропускаю")
            continue
        current = set(e.get("current", []))
        head = f"[{i}/{len(entries)}] {ref}   {rdur:.2f}s   [{e['reason']}"
        head += f", score {e['score']}]" if "score" in e else "]"
        print(head)

        ref_pcm = decode_pcm_i16(rpath)
        cands, n_win = find_candidates(ref_pcm, rdur, pool, args.pak_dir, args.cache, args, extra=current)
        shown = args.top
        top = build_top(cands, shown, current)
        print(f"   окно длительности: {n_win} треков проверено")

        if export_root:
            d = os.path.join(export_root, re.sub(r"[^\w.\-]", "_", os.path.splitext(ref)[0]))
            os.makedirs(d, exist_ok=True)
            shutil.copy2(rpath, os.path.join(d, "00_REF" + os.path.splitext(rpath)[1]))
            for rank, (score, name, dur) in enumerate(top, 1):
                shutil.copy2(os.path.join(args.pak_dir, name),
                             os.path.join(d, f"{rank:02d}_{os.path.splitext(name)[0]}_s{score:.2f}.mp3"))
            template[ref] = []
            print_candidates(top, rdur, claimed, current, len(top))
            print()
            continue

        print_candidates(top, rdur, claimed, current, len(top))
        if not args.no_autoplay:
            play(rpath)

        while True:
            try:
                raw = input("   > ").strip()
            except (EOFError, KeyboardInterrupt):
                raw = "q"
            if not raw or raw in ("?", "h"):
                print(HELP)
                continue
            parts = raw.split()
            cmd = parts[0]

            if cmd == "r":
                play(rpath)
            elif cmd == "m":
                if shown >= len(cands):
                    print(f"   больше кандидатов нет (все {len(cands)} из окна длительности показаны).")
                    print("   Расширить окно: --tol-abs / --tol-rel / --cap при запуске review.")
                else:
                    shown = min(shown + args.top, len(cands))
                    top = build_top(cands, shown, current)
                    print_candidates(top, rdur, claimed, current, len(top))
                    print(f"   показаны кандидаты 1..{shown} из {len(cands)}")
            elif cmd.isdigit() or (cmd == "p" and len(parts) > 1 and parts[1].isdigit()):
                k = int(cmd if cmd.isdigit() else parts[1])
                if 1 <= k <= len(top):
                    play(os.path.join(args.pak_dir, top[k - 1][1]))
                else:
                    print("   нет такого номера")
            elif cmd == "a" and len(parts) > 1 and all(p.isdigit() and 1 <= int(p) <= len(top) for p in parts[1:]):
                picked = [top[int(p) - 1][1] for p in parts[1:]]
                for t in picked:
                    if t in claimed and claimed[t] != ref and t not in current:
                        print(f"   ! {t} сейчас занят под '{claimed[t]}' — apply откажется без --steal")
                choices[ref] = picked
                save_choices(args.choices, choices)
                print(f"   принято: {', '.join(picked)}\n")
                break
            elif cmd == "n":
                choices[ref] = None
                save_choices(args.choices, choices)
                print("   помечено: в паке нет\n")
                break
            elif cmd == "s":
                print("   пропущено\n")
                break
            elif cmd == "q":
                save_choices(args.choices, choices)
                print(f"Выход. Выбор сохранён в {args.choices}")
                return
            else:
                print(HELP)

    if export_root:
        tpath = os.path.join(export_root, "choices_template.json")
        save_choices(tpath, template)
        print(f"Экспорт готов: {export_root}/<ref>/00_REF + 01..NN кандидаты")
        print(f"Впиши выбранные треки в {tpath} (напр. \"khouse_gaz_4_3.wav\": [\"track_13834.mp3\"]),")
        print(f"\"ref\": null = в паке нет. Потом: python3 {sys.argv[0]} apply {args.matches} {tpath} out.json")
    else:
        save_choices(args.choices, choices)
        print(f"Готово. Выбор: {args.choices}")
        print(f"Дальше: python3 {sys.argv[0]} apply {args.matches} {args.choices} voiceovers_final.json")


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------

def cmd_apply(args):
    matches = json.load(open(args.matches, encoding="utf-8"))
    if os.path.isfile(args.choices):
        choices = json.load(open(args.choices, encoding="utf-8"))
    else:
        choices = {}
        print(f"({args.choices} нет — ручных правок не будет, копирую {args.matches} как есть)")

    for p in args.pair or []:
        if "=" not in p:
            sys.exit(f"--pair ожидает track=ref, получено '{p}'")
        track, ref = p.split("=", 1)
        if choices.get(ref) is None:
            choices[ref] = []
        choices[ref].append(track)

    added = removed = kept_conflicts = 0
    for ref, tracks in choices.items():
        if tracks == []:
            continue
        old = [t for t, r in matches.items() if r == ref]

        if tracks is None:
            for t in old:
                del matches[t]
                removed += 1
                print(f"  - снята пара {t} -> {ref} (в паке нет)")
            continue

        ok = []
        for t in tracks:
            other = matches.get(t)
            if other is not None and other != ref:
                if args.steal:
                    print(f"  ! {t}: забираю у '{other}' — тот реф остаётся без пары")
                else:
                    print(f"  ! {t} занят под '{other}', пропускаю (нужен --steal)")
                    kept_conflicts += 1
                    continue
            ok.append(t)
        if not ok:
            continue
        for t in old:
            if t not in ok:
                del matches[t]
                removed += 1
                print(f"  - снята прежняя пара {t} -> {ref}")
        for t in ok:
            if matches.get(t) != ref:
                matches[t] = ref
                added += 1
                print(f"  + {t} -> {ref}")

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(matches, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\n{args.output}: {len(matches)} пар (добавлено {added}, снято {removed}"
          f"{', конфликтов пропущено ' + str(kept_conflicts) if kept_conflicts else ''}).")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="какие референсы остались без пары")
    p.add_argument("ref_dir")
    p.add_argument("matches", help="текущий matches json (track -> ref)")
    p.add_argument("--forced", help="voiceovers_v2_forced.json — заодно вытащить подозрительные forced-пары")
    p.add_argument("--weak-below", type=float, default=0.55)
    p.add_argument("-o", "--output", default="unmatched_refs.json")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("review", help="кандидаты по каждому рефу: слушать и выбирать (или --export)")
    p.add_argument("refs", help="unmatched_refs.json из `list`")
    p.add_argument("pak_dir")
    p.add_argument("matches")
    p.add_argument("--choices", default="review_choices.json")
    p.add_argument("--only", nargs="+", help="только эти реф(ы) по имени (можно и не из списка, тогда нужен --ref-dir)")
    p.add_argument("--ref-dir", help="папка с референсами (для --only с именами не из списка)")
    p.add_argument("--export", metavar="DIR", help="не интерактивно: выложить реф + кандидатов в папки для прослушивания")
    p.add_argument("--redo", action="store_true", help="пройти заново и те, что уже есть в choices")
    p.add_argument("--top", type=int, default=10, help="сколько кандидатов показывать (10)")
    p.add_argument("--tol-abs", type=float, default=0.4, help="окно длительности, сек, минимум (0.4)")
    p.add_argument("--tol-rel", type=float, default=0.25, help="окно длительности как доля длины рефа (0.25)")
    p.add_argument("--cap", type=int, default=1200, help="максимум треков на реф, ближайшие по длительности (1200)")
    p.add_argument("--max-lag-ms", type=float, default=150.0, help="поиск сдвига ± мс (150)")
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4), help="параллельных ffmpeg")
    p.add_argument("--include-parts", action="store_true",
                   help="добавить в кандидаты куски *_partNN.mp3 (recode/pack с ними не работают — только чтобы узнать, где реплика)")
    p.add_argument("--no-autoplay", action="store_true")
    p.add_argument("--cache", default=CACHE_DIR_DEFAULT)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("apply", help="влить выбор в matches json")
    p.add_argument("matches")
    p.add_argument("choices", help="review_choices.json (если файла нет — matches копируется как есть)")
    p.add_argument("output")
    p.add_argument("--steal", action="store_true", help="разрешить забирать трек, уже занятый другим рефом")
    p.add_argument("--pair", action="append", metavar="track=ref", help="принудительно задать пару руками (можно много раз)")
    p.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
