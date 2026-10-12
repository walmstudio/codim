#!/usr/bin/env python3
"""Переносит кириллицу из ПК-шрифтов CoD4 (русская версия) в PS3 code_post_gfx*.ff.
  build_font.py ps3.ff  pc.ff  gamefonts_pc.iwi  out.ff  [preview.png]
Слоты 0xC0-0xFF (+0xA8,0xB8,0xB9) в таблицах глифов PS3 перезаписываются кириллицей (cp1251),
картинки рисуются в освободившиеся прямоугольники атласа fonts/gamefonts_ps3 (512x1024 DXT5)."""
import sys, struct, re, statistics as st
import numpy as np
from PIL import Image
sys.path.insert(0, '.')
import codim_cod4loc as c
from fontlib import parse, rect, W, H, FONTS

KEEP = {0xA0, 0xA9, 0xAB, 0xAD, 0xAE, 0xB0, 0xB7, 0xBB}
REM = [x for x in range(0xA1, 0x100) if x not in KEEP]
TARGETS = [0xA8, 0xB8, 0xB9] + list(range(0xC0, 0x100))
SRC = {'bigFont': 'extraBigFont', 'extraBigFont': 'extraBigFont', 'boldFont': 'extraBigFont',
       'normalFont': 'extraBigFont', 'smallFont': 'extraBigFont', 'objectiveFont': 'objectiveFont'}
GROUPS = [['bigFont', 'extraBigFont'], ['boldFont'], ['normalFont'], ['smallFont'], ['objectiveFont']]

# ---------- DXT5 (только альфа-половина блока) ----------
def dxt5_alpha_decode(data, w, h):
    a = np.frombuffer(data, np.uint8).reshape(h // 4, w // 4, 16)[:, :, :8].astype(np.int64)
    a0, a1 = a[..., 0], a[..., 1]
    bits = np.zeros(a.shape[:2], np.uint64)
    for i in range(6): bits |= a[..., 2 + i].astype(np.uint64) << np.uint64(8 * i)
    idx = np.stack([(bits >> np.uint64(3 * i)) & np.uint64(7) for i in range(16)], -1).astype(np.int64)
    pal = np.zeros(a.shape[:2] + (8,), np.float64)
    pal[..., 0], pal[..., 1] = a0, a1
    m8 = a0 > a1
    for i in range(2, 8):
        pal[..., i] = np.where(m8, ((8 - i) * a0 + (i - 1) * a1) / 7.0,
                               np.where(i < 6, ((6 - i) * a0 + (i - 1) * a1) / 5.0, 0.0 if i == 6 else 255.0))
    v = np.take_along_axis(pal, idx, -1)
    v = np.floor(v + 0.0).astype(np.uint8)   # декодер GPU: целочисленное деление
    out = np.zeros((h, w), np.uint8)
    for by in range(h // 4):
        for bx in range(w // 4):
            out[by*4:by*4+4, bx*4:bx*4+4] = v[by, bx].reshape(4, 4)
    return out

def dxt5_alpha_block(px):
    px = [int(x) for x in px]; best = None
    for a0, a1 in ((max(px), min(px)), (min(px), max(px))):
        if a0 > a1: pal = [a0, a1] + [((8 - i) * a0 + (i - 1) * a1) // 7 for i in range(2, 8)]
        else: pal = [a0, a1] + [((6 - i) * a0 + (i - 1) * a1) // 5 for i in range(2, 6)] + [0, 255]
        idx = [min(range(8), key=lambda k: abs(pal[k] - p)) for p in px]
        err = sum(abs(pal[i] - p) for i, p in zip(idx, px))
        if best is None or err < best[0]: best = (err, a0, a1, idx)
    _, a0, a1, idx = best; bits = 0
    for i, k in enumerate(idx): bits |= k << (3 * i)
    return bytes([a0, a1]) + bits.to_bytes(6, 'little')

def paste_alpha(buf, base, alpha, blocks):
    """перекодирует блоки 4x4 (альфа-половина), цветовую половину не трогает"""
    for by, bx in sorted(blocks):
        o = base + (by * (W // 4) + bx) * 16
        buf[o:o+8] = dxt5_alpha_block(alpha[by*4:by*4+4, bx*4:bx*4+4].reshape(16))

def main(ps3_ff, pc_ff, iwi, out, preview=None):
    zp = c.ff_to_zone(ps3_ff); zc = bytes(c.ff_to_zone(pc_ff))
    ps = parse(bytes(zp), '>'); pc = parse(zc, '<')
    m = next(x for x in re.finditer(rb'gamefonts_ps3\x00', bytes(zp)) if bytes(zp[x.start()-8:x.start()]) == b'\xff' * 8); tex0 = m.end()
    alpha = dxt5_alpha_decode(bytes(zp[tex0:tex0 + W * H]), W, H)
    d = open(iwi, 'rb').read(); assert d[:3] == b'IWi' and d[8-2] == 13 or True
    pca = dxt5_alpha_decode(d[len(d) - W * H:], W, H)   # IWi хранит мипы от меньшего к большему: верхний мип в конце файла
    orig = alpha.copy()
    keep = np.zeros((H, W), bool); taken = np.zeros((H, W), bool)
    for n in FONTS:
        for g in ps[n]['gl']:
            if g['let'] not in REM: x0, y0, x1, y1 = rect(g); keep[y0:y1, x0:x1] = True
    used_rows = np.where(keep.any(1))[0]
    # самая нижняя занятая строка над пустой областью атласа
    lowest = int(np.where(alpha[:H-4].any(1))[0].max()) + 1   # последний блок-ряд не трогаем: там белый тексель (511,1023)
    free_top = (lowest + 3) // 4 * 4
    shelf = [free_top, 0, 0]   # y, x, высота полки
    touched = set(); report = []

    def blocks_of(r):
        x0, y0, x1, y1 = r
        return {(by, bx) for by in range(y0 // 4, (y1 + 3) // 4) for bx in range(x0 // 4, (x1 + 3) // 4)}

    for grp in GROUPS:
        lead = grp[0]; A = {g['let']: g for g in ps[lead]['gl']}
        B = {g['let']: g for g in pc[SRC[lead]]['gl']}
        flat = [ord(ch) for ch in 'EFHIKLTVXYZ']; lat = list(range(0x41, 0x5b)) + list(range(0x61, 0x7b))
        rh = st.median(A[k]['ph'] / B[k]['ph'] for k in flat)
        rw = st.median(A[k]['pw'] / B[k]['pw'] for k in lat)
        rd = st.median(A[k]['dx'] / B[k]['dx'] for k in lat)
        pool = []
        for k in REM:
            if k in A:
                r = rect(A[k])
                if not keep[r[1]:r[3], r[0]:r[2]].any() and not taken[r[1]:r[3], r[0]:r[2]].any(): pool.append(r)
        pool = list(dict.fromkeys(pool))
        items = []
        for k in TARGETS:
            if k not in A or k not in B: continue
            sr = rect(B[k]); sw, sh = sr[2]-sr[0], sr[3]-sr[1]
            tw, th = max(1, round(sw * rw)), max(1, round(sh * rh)) if sh else 0
            items.append((k, sr, tw, th))
        items.sort(key=lambda t: -t[2] * t[3]); placed = {}
        free = list(pool)
        for k, sr, tw, th in items:
            cand = [p for p in free if p[2]-p[0] >= tw and p[3]-p[1] >= th]
            if cand:
                p = min(cand, key=lambda p: (p[2]-p[0]) * (p[3]-p[1])); free.remove(p)
                placed[k] = (p, (p[0], p[1], p[0]+tw, p[1]+th))
            else:   # в нижнюю пустую область атласа
                if shelf[1] + tw + 1 > W: shelf[0] += shelf[2] + 1; shelf[1] = 0; shelf[2] = 0
                assert shelf[0] + th <= H - 4, 'нет места в атласе'
                r = (shelf[1], shelf[0], shelf[1]+tw, shelf[0]+th); shelf[1] += tw + 1; shelf[2] = max(shelf[2], th)
                placed[k] = (r, r)
        nfb = sum(1 for k, (p, r) in placed.items() if p == r and r[1] >= free_top)
        report.append((grp, 'rh %.3f rw %.3f rd %.3f' % (rh, rw, rd), 'pool', len(pool), 'placed', len(placed), 'в нижней области', nfb))
        # рисуем: сначала чистим все использованные слоты пула, затем кладём глифы
        for k, (p, r) in placed.items():
            alpha[p[1]:p[3], p[0]:p[2]] = 0; taken[p[1]:p[3], p[0]:p[2]] = True; touched |= blocks_of(p)
        for k, sr, tw, th in items:
            p, r = placed[k]
            if tw <= 0 or th <= 0: continue
            crop = Image.fromarray(pca[sr[1]:sr[3], sr[0]:sr[2]])
            im = np.asarray(crop.resize((tw, th), Image.LANCZOS), np.uint8) if (tw, th) != crop.size else np.asarray(crop)
            alpha[r[1]:r[3], r[0]:r[2]] = im; touched |= blocks_of(r)
        # таблицы глифов (в каждом шрифте группы)
        for fn in grp:
            for k, sr, tw, th in items:
                g = {q['let']: q for q in ps[fn]['gl']}[k]; p, r = placed[k]; b = B[k]
                x0 = round(b['x0'] * rw); y0 = round(b['y0'] * rh); dx = round(b['dx'] * rd)
                rec = bytearray(zp[g['off']:g['off']+24])
                rec[2:4] = struct.pack('bb', max(-128, min(127, x0)), max(-128, min(127, y0)))
                rec[4] = max(0, min(255, dx)); rec[5] = min(255, tw); rec[6] = min(255, th)
                rec[8:24] = struct.pack('>4f', r[0] / W, r[1] / H, r[2] / W, r[3] / H)
                zp[g['off']:g['off']+24] = rec
    touched = {(by, bx) for by, bx in touched if not np.array_equal(alpha[by*4:by*4+4, bx*4:bx*4+4], orig[by*4:by*4+4, bx*4:bx*4+4])}
    zb = bytearray(zp); paste_alpha(zb, tex0, alpha, touched)
    c.zone_to_ff(zb, out)
    for r in report: print(*r)
    print('перекодировано блоков:', len(touched), '->', out)
    if preview: make_preview(bytes(zb), tex0, preview)

def make_preview(zone, tex0, path):
    ps = parse(zone, '>'); al = dxt5_alpha_decode(zone[tex0:tex0 + W * H], W, H)
    text = bytes(range(0xC0, 0x100)).decode('cp1251') + ' Ёё№ Новая игра Настройки'
    canvas = Image.new('L', (900, 60 * len(FONTS) + 10), 0); y = 10
    for n in FONTS:
        A = {g['let']: g for g in ps[n]['gl']}; x = 5; base = 45 * 1
        line = Image.new('L', (900, 60), 0)
        sc = 30.0 / ps[n]['pixh']
        for ch in text:
            code = ch.encode('cp1251')[0] if ch != ' ' else 0x20
            g = A.get(code)
            if not g: continue
            r = rect(g)
            if r[2] > r[0] and r[3] > r[1]:
                gi = Image.fromarray(al[r[1]:r[3], r[0]:r[2]])
                gi = gi.resize((max(1, round(gi.width * sc)), max(1, round(gi.height * sc))), Image.LANCZOS)
                line.paste(gi, (x + round(g['x0'] * sc), base + round(g['y0'] * sc)))
            x += round(g['dx'] * sc)
        canvas.paste(line, (0, y - 10)); y += 60
    canvas.save(path)

if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) < 4: print(__doc__); sys.exit()
    main(a[0], a[1], a[2], a[3], a[4] if len(a) > 4 else None)
