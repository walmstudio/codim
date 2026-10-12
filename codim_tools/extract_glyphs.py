#!/usr/bin/env python3
"""Достаёт атлас глифов и таблицу глифов из code_post_gfx*.ff (CoD4 PS3).
  extract_glyphs.py code_post_gfx.ff  [outdir]
Кладёт рядом с инструментами codim_*.py. Выдаёт: gamefonts_atlas.png (альфа = глифы),
gamefonts_atlas_grid.png (с рамками глифов) и glyphs.json (метрики + UV + прямоугольники)."""
import sys, re, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw
import codim_cod4loc as c, codim_fontlib; sys.modules["fontlib"] = codim_fontlib
from codim_fontlib import parse, rect, W, H, FONTS
from codim_build_font import dxt5_alpha_decode

ff = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else '.'
os.makedirs(out, exist_ok=True)
zone = bytes(c.ff_to_zone(ff))
m = next(x for x in re.finditer(rb'gamefonts_ps3\x00', zone) if zone[x.start()-8:x.start()] == b'\xff'*8)
tex0 = m.end()
alpha = dxt5_alpha_decode(zone[tex0:tex0 + W*H], W, H)
Image.fromarray(alpha).save(f'{out}/gamefonts_atlas.png')

fonts = parse(zone, '>'); grid = Image.fromarray(alpha).convert('RGB'); d = ImageDraw.Draw(grid)
js = {}
for n in FONTS:
    js[n] = {'pixel_height': fonts[n]['pixh'], 'glyphs': []}
    for g in fonts[n]['gl']:
        r = rect(g)
        js[n]['glyphs'].append({'code': g['let'], 'char': chr(g['let']) if 32 <= g['let'] < 127 else None,
            'x0': g['x0'], 'y0': g['y0'], 'dx': g['dx'], 'w': g['pw'], 'h': g['ph'],
            'uv': list(g['uv']), 'rect': list(r)})
        if r[2] > r[0] and r[3] > r[1]: d.rectangle(r, outline=(255, 0, 0))
grid.save(f'{out}/gamefonts_atlas_grid.png')
json.dump(js, open(f'{out}/glyphs.json', 'w'), ensure_ascii=False, indent=1)
print('атлас', W, 'x', H, 'смещение текстуры в зоне', tex0, '| шрифтов:', {n: len(js[n]['glyphs']) for n in FONTS})
