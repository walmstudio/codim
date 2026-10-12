#!/usr/bin/env python3
"""codim_cod4loc_fit.py — безопасный порт LocalizeEntry из ПК (RU) .ff в PS3 .ff.

Размер зоны, размеры пулов и любые указатели НЕ меняются: русская строка пишется
на место английской только если влезает (добивается пробелами). Не влезшие остаются
английскими и попадают в отчёт (TSV) с длинами — их нужно сократить вручную.

  python3 codim_cod4loc_fit.py ps3.ff pc_ru.ff out.ff --report over.tsv [--skip PLATFORM_,XBOXLIVE_]
  python3 codim_cod4loc_fit.py --override fixed.tsv ...   (TSV КЛЮЧ<TAB>текст cp1251-совместимый, заменяет ПК-значение)
"""
import sys, struct, argparse
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from codim_cod4loc import ff_to_zone, zone_to_ff, PAT, PAT_ANY, esc, unesc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ps3'); ap.add_argument('pc'); ap.add_argument('out')
    ap.add_argument('--skip', default=''); ap.add_argument('--report'); ap.add_argument('--override')
    ap.add_argument('--enc', default='cp1251')
    a = ap.parse_args()
    skip = [s for s in a.skip.split(',') if s]
    z = ff_to_zone(a.ps3); orig_len = len(z)
    src = {m.group(2).decode(): bytes(m.group(1)) for m in PAT_ANY.finditer(bytes(ff_to_zone(a.pc)))}
    if a.override:
        for line in open(a.override, encoding='utf-8'):
            line = line.rstrip('\n')
            if '\t' in line:
                k, v = line.split('\t', 1); src[k] = unesc(v, a.enc)
    fit = same = miss = skipped = 0; over = []
    for m in PAT.finditer(bytes(z)):
        k = m.group(2).decode(); old = bytes(m.group(1))
        if any(k.startswith(s) for s in skip): skipped += 1; continue
        if k not in src: miss += 1; continue
        new = src[k]
        if b'\x00' in new: continue
        if new == old: same += 1; continue
        if len(new) <= len(old):
            z[m.start()+8:m.start()+8+len(old)] = new + b' ' * (len(old) - len(new)); fit += 1
        else:
            over.append((k, len(old), len(new), esc(old), esc(new)))
    assert len(z) == orig_len
    zone_to_ff(z, a.out)
    print(f'влезло и записано: {fit}, уже совпадает: {same}, нет в ПК: {miss}, пропущено: {skipped}')
    print(f'НЕ влезло (осталось английским): {len(over)}, не хватает байт суммарно: {sum(n-o for _,o,n,_,_ in over)}')
    print(f'размер зоны не менялся: {orig_len} -> {len(z)}  ->  {a.out}')
    if a.report:
        with open(a.report, 'w', encoding='utf-8') as f:
            f.write('KEY\torig_len\tru_len\tover\tEN\tRU\n')
            for k, o, n, eo, en in sorted(over, key=lambda r: -(r[2]-r[1])):
                f.write(f'{k}\t{o}\t{n}\t{n-o}\t{eo}\t{en}\n')
        print('отчёт:', a.report)
main()
