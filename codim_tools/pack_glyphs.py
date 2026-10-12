#!/usr/bin/env python3
"""Пакует отредактированный атлас (и таблицы глифов) обратно в code_post_gfx*.ff (CoD4 PS3).
  pack_glyphs.py  оригинал.ff  atlas_edited.png  новый.ff  [glyphs.json]
 - atlas_edited.png: 512x1024, 8 бит серый (альфа = глифы), как отдаёт extract_glyphs.py
 - перекодируются ТОЛЬКО блоки 4x4, которые вы изменили; остальное остаётся байт в байт
 - цветовая половина DXT5 не трогается (там белый), размер зоны не меняется
 - glyphs.json (необязательно): если правили метрики/UV, записи глифов обновляются"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, json
from PIL import Image
import codim_cod4loc as c, codim_fontlib; sys.modules['fontlib'] = codim_fontlib
from codim_fontlib import parse, W, H, FONTS
from codim_build_font import dxt5_alpha_decode
import codim_iwi

src, png, dst = sys.argv[1:4]; jpath = sys.argv[4] if len(sys.argv) > 4 else None
zone = c.ff_to_zone(src); zb = bytes(zone)
m = next(x for x in re.finditer(rb'gamefonts_ps3\x00', zb) if zb[x.start()-8:x.start()] == b'\xff'*8); tex0 = m.end()
orig = dxt5_alpha_decode(zb[tex0:tex0 + W*H], W, H)
im = Image.open(png); assert im.size == (W, H), f'нужен {W}x{H}, а не {im.size}'
new = np.asarray(im.convert('L'), np.uint8)
diff = (new != orig).reshape(H//4, 4, W//4, 4).any((1, 3))
blocks = {(int(by), int(bx)) for by, bx in zip(*np.nonzero(diff))}
codim_iwi.paste_alpha(zone, tex0, new, blocks, W)
print('изменено блоков 4x4:', len(blocks))

if jpath:
    fonts = parse(zb, '>'); js = json.load(open(jpath)); n_rec = 0
    for fn in FONTS:
        by_code = {g['let']: g for g in fonts[fn]['gl']}
        for g in js[fn]['glyphs']:
            o = by_code[g['code']]
            rec = bytearray(zb[o['off']:o['off']+24])
            rec[2:4] = struct.pack('bb', g['x0'], g['y0']); rec[4], rec[5], rec[6] = g['dx'], g['w'], g['h']
            rec[8:24] = struct.pack('>4f', *g['uv'])
            if bytes(rec) != zb[o['off']:o['off']+24]: zone[o['off']:o['off']+24] = rec; n_rec += 1
    print('изменено записей глифов:', n_rec)
c.zone_to_ff(zone, dst); print('->', dst)
