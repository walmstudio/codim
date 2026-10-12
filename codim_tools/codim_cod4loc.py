#!/usr/bin/env python3
"""cod4loc — замена localize-строк в .ff от CoD4 PS3 (IWffu100).

  cod4loc.py export  файл.ff  строки.tsv
  cod4loc.py import  файл.ff  строки.tsv  новый.ff  [--enc cp1251]
  cod4loc.py port    ps3.ff  русский_пк.ff  новый.ff  [--skip PLATFORM_,XBOXLIVE_]
      port: берёт из ПК-ff строки по ключам и вписывает БАЙТ В БАЙТ в PS3-ff
      (кодировку не трогает; печатает, какие байты >0x7F встретились в ПК-тексте).

TSV: КЛЮЧ <TAB> текст. Правь только второй столбец, строки можно удалять —
меняются лишь те ключи, текст которых отличается от оригинала.
Экранирование: \\n \\r \\t \\\\, а любой сырой байт (кнопки \\x01, \\x02, символы >0x7F) как \\xNN.
Обычные (юникодные) символы в тексте кодируются в байты выбранной кодировкой (--enc, по умолчанию cp1251).
Экспортируй всегда из ОРИГИНАЛЬНОГО ff, а не из уже изменённого."""
import sys, re, zlib, struct
CH = 0x10000
PAT = re.compile(rb'\xff{8}([^\x00\xff]*)\x00([A-Z][A-Z0-9_]{2,})\x00')
PAT_ANY = re.compile(rb'\xff{8}([^\x00]*)\x00([A-Z][A-Z0-9_]{2,})\x00')   # значение может содержать 0xFF (я в cp1251)

def ff_to_zone(path):
    d = open(path,'rb').read(); assert d[:8] == b'IWffu100', 'не IWffu100'
    if d[12:13] == b'\x78':                       # ПК: обычный zlib-поток после 12 байт заголовка (НЕ проверено на реальном файле)
        return bytearray(zlib.decompressobj().decompress(d[12:]))
    pos, out = 12, bytearray()
    while pos + 2 <= len(d):
        n = struct.unpack('>H', d[pos:pos+2])[0]
        if n == 0 or pos + 2 + n > len(d): break
        out += zlib.decompressobj(-15).decompress(d[pos+2:pos+2+n]); pos += 2 + n
    return bytearray(out)

def zone_to_ff(z, path):
    z = bytes(z) + b'\x00' * (-len(z) % CH); res = bytearray(b'IWffu100' + struct.pack('>I', 1))
    for i in range(0, len(z), CH):
        c = zlib.compressobj(9, zlib.DEFLATED, -15); b = c.compress(z[i:i+CH]) + c.flush()
        assert len(b) < 0x10000; res += struct.pack('>H', len(b)) + b
    open(path,'wb').write(res + b'\x00\x01')

def esc(b):
    o = []
    for c in b:
        if c == 0x5c: o.append('\\\\')
        elif c == 0x0a: o.append('\\n')
        elif c == 0x0d: o.append('\\r')
        elif c == 0x09: o.append('\\t')
        elif c < 0x20 or c >= 0x7f: o.append('\\x%02x' % c)
        else: o.append(chr(c))
    return ''.join(o)

def unesc(s, enc):
    out = bytearray(); i = 0
    while i < len(s):
        ch = s[i]
        if ch == '\\' and i + 1 < len(s):
            n = s[i+1]
            if n == 'x' and re.fullmatch(r'[0-9a-fA-F]{2}', s[i+2:i+4]): out.append(int(s[i+2:i+4], 16)); i += 4; continue
            m = {'n': 10, 'r': 13, 't': 9, '\\': 92}.get(n)
            if m is not None: out.append(m); i += 2; continue
        out += ch.encode(enc); i += 1
    return bytes(out)

def do_export(ff, tsv):
    z = ff_to_zone(ff); n = 0
    with open(tsv, 'w', encoding='utf-8') as f:
        for m in PAT.finditer(bytes(z)):
            f.write(m.group(2).decode() + '\t' + esc(m.group(1)) + '\n'); n += 1
    print(n, 'строк ->', tsv)

def do_import(ff, tsv, out, enc):
    z = ff_to_zone(ff); recs = [(m.start(), m.group(1), m.group(2).decode()) for m in PAT.finditer(bytes(z))]
    tr = {}
    for line in open(tsv, encoding='utf-8'):
        line = line.rstrip('\n')
        if '\t' in line: k, v = line.split('\t', 1); tr[k] = v
    added = changed = 0
    for p, v, k in reversed(recs):
        if k not in tr or tr[k] == esc(v): continue
        nv = unesc(tr[k], enc)
        if b'\x00' in nv: sys.exit('нулевой байт в значении ' + k)
        z[p+8:p+8+len(v)] = nv; added += len(nv) - len(v); changed += 1
    if added > 0:                        # запись длиннее -> растим size и блоки 4 и 6
        z[0:4] = struct.pack('>I', struct.unpack('>I', z[0:4])[0] + added)
        for i in (4, 6):
            o = 8 + 4*i; b = struct.unpack('>I', z[o:o+4])[0]
            if b: z[o:o+4] = struct.pack('>I', b + added)
    zone_to_ff(z, out)
    print('заменено:', changed, ' прирост байт:', added, ' ->', out)

def do_port(ps3, pc, out, skip):
    src = {}
    for m in PAT_ANY.finditer(bytes(ff_to_zone(pc))): src[m.group(2).decode()] = m.group(1)
    hi = {}
    for v in src.values():
        for c in v:
            if c >= 0x80: hi[c] = hi.get(c, 0) + 1
    print('в ПК-файле строк:', len(src), '; байты >0x7F:', ' '.join('%02x:%d' % kv for kv in sorted(hi.items())[:70]))
    z = ff_to_zone(ps3); recs = [(m.start(), m.group(1), m.group(2).decode()) for m in PAT.finditer(bytes(z))]
    added = changed = miss = 0
    for p, v, k in reversed(recs):
        if k not in src or any(k.startswith(x) for x in skip if x): miss += k not in src; continue
        nv = src[k]
        if nv == v: continue
        z[p+8:p+8+len(v)] = nv; added += len(nv) - len(v); changed += 1
    if added > 0:
        z[0:4] = struct.pack('>I', struct.unpack('>I', z[0:4])[0] + added)
        for i in (4, 6):
            o = 8 + 4*i; b = struct.unpack('>I', z[o:o+4])[0]
            if b: z[o:o+4] = struct.pack('>I', b + added)
    zone_to_ff(z, out)
    print('PS3-строк:', len(recs), ' заменено:', changed, ' нет в ПК-файле:', miss, ' прирост байт:', added, '->', out)

if __name__ == '__main__':
    a = sys.argv[1:]; enc = 'cp1251'; skip = []
    if '--skip' in a: i = a.index('--skip'); skip = a[i+1].split(','); del a[i:i+2]
    if '--enc' in a: i = a.index('--enc'); enc = a[i+1]; del a[i:i+2]
    if a and a[0] == 'export' and len(a) == 3: do_export(a[1], a[2])
    elif a and a[0] == 'import' and len(a) == 4: do_import(a[1], a[2], a[3], enc)
    elif a and a[0] == 'port' and len(a) == 4: do_port(a[1], a[2], a[3], skip)
    else: print(__doc__)
