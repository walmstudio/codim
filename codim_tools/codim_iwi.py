#!/usr/bin/env python3
"""codim_iwi — чтение и правка текстур IWi (CoD4, версия 6).

  codim_iwi.py info  файл.iwi              — заголовок, формат, мип-уровни
  codim_iwi.py png   файл.iwi  выход.png   — верхний уровень в PNG

Как устроен файл (версия 6):
  0   "IWi"            4  формат (0x01 ARGB32, 0x02 RGB24, 0x03 GA16, 0x04 A8, 0x0B DXT1, 0x0C DXT3, 0x0D DXT5)
  3   версия (=6)      5  флаги
  6   ширина  u16      8  высота u16     10  глубина u16
  12  4 x u32 размеры файла для picmip 0..3     (заголовок = 28 байт)
  дальше мип-уровни ОТ МЕНЬШЕГО К БОЛЬШЕМУ: верхний (полный) уровень лежит в конце файла.

Число мип-уровней определяется по размеру файла (сумма размеров уровней должна сойтись).
Правка (with_plane): DXT5 — перекодируются только изменённые блоки 4x4 альфа-половины, цветовая половина
не трогается; нижние мипы обновляются только там, где изменился верхний уровень. A8 — запись байтов.
Остальные форматы — только просмотр."""
import sys, io, struct
import numpy as np
from PIL import Image

FORMATS = {0x01: 'ARGB32', 0x02: 'RGB24', 0x03: 'GA16', 0x04: 'A8', 0x0B: 'DXT1', 0x0C: 'DXT3', 0x0D: 'DXT5'}
BPP = {0x01: 4, 0x02: 3, 0x03: 2, 0x04: 1}
BLOCK = {0x0B: 8, 0x0C: 16, 0x0D: 16}
HDR = 28


def level_size(fmt, w, h):
    if fmt in BLOCK:
        return ((w + 3) // 4) * ((h + 3) // 4) * BLOCK[fmt]
    return w * h * BPP[fmt]


# ---------------------------------------------------------------- DXT5, альфа-половина блока
def dxt5_alpha_decode(data, w, h):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    a = np.frombuffer(data, np.uint8, bw * bh * 16).reshape(bh, bw, 16)[:, :, :8].astype(np.int64)
    a0, a1 = a[..., 0], a[..., 1]
    bits = np.zeros(a.shape[:2], np.uint64)
    for i in range(6):
        bits |= a[..., 2 + i].astype(np.uint64) << np.uint64(8 * i)
    idx = np.stack([(bits >> np.uint64(3 * i)) & np.uint64(7) for i in range(16)], -1).astype(np.int64)
    pal = np.zeros(a.shape[:2] + (8,), np.int64)
    pal[..., 0], pal[..., 1] = a0, a1
    m8 = a0 > a1
    for i in range(2, 8):
        hi = ((8 - i) * a0 + (i - 1) * a1) // 7
        lo = ((6 - i) * a0 + (i - 1) * a1) // 5 if i < 6 else (0 if i == 6 else 255)
        pal[..., i] = np.where(m8, hi, lo)
    v = np.take_along_axis(pal, idx, -1).astype(np.uint8)
    out = v.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)
    return np.ascontiguousarray(out[:h, :w])


def _palette(a0, a1):
    if a0 > a1:
        return [a0, a1] + [((8 - i) * a0 + (i - 1) * a1) // 7 for i in range(2, 8)]
    return [a0, a1] + [((6 - i) * a0 + (i - 1) * a1) // 5 for i in range(2, 6)] + [0, 255]


def _assign(px, pal):
    best = {}
    for v in set(px):
        k = min(range(8), key=lambda j: abs(pal[j] - v))
        best[v] = (k, (pal[k] - v) ** 2)
    idx = [best[p][0] for p in px]
    return idx, sum(best[p][1] for p in px)


_HAS = None


def _exact_pair(px):
    """Любая пара (a0, a1), палитра которой содержит ВСЕ значения блока (по таблице 256 x 65536)."""
    global _HAS
    if _HAS is None:
        a0, a1 = np.meshgrid(np.arange(256, dtype=np.int16), np.arange(256, dtype=np.int16), indexing='ij')
        cols = [a0, a1]
        for i in range(2, 8):
            hi = ((8 - i) * a0 + (i - 1) * a1) // 7
            lo = ((6 - i) * a0 + (i - 1) * a1) // 5 if i < 6 else np.full_like(a0, 0 if i == 6 else 255)
            cols.append(np.where(a0 > a1, hi, lo))
        pal = np.stack(cols, -1).reshape(65536, 8)
        _HAS = np.zeros((256, 65536), bool)
        for j in range(8):
            _HAS[pal[:, j], np.arange(65536)] = True          # _HAS[v, пара] = «v есть в палитре пары»
    ok = np.ones(65536, bool)
    for v in set(px):
        ok &= _HAS[v]
    hit = np.flatnonzero(ok)
    return (int(hit[0] // 256), int(hit[0] % 256)) if len(hit) else None


_CACHE = {}


def dxt5_alpha_block(px):
    key = tuple(int(x) for x in px)
    r = _CACHE.get(key)
    if r is None:
        if len(_CACHE) > 200000:
            _CACHE.clear()
        r = _CACHE[key] = _encode_block(key)
    return r


def _encode_block(px):
    """Кодирует 16 значений альфы в 8 байт DXT5.
    Кандидаты: крайние значения блока (режим 8 уровней), крайние без 0/255 (режим 6 уровней + 0 и 255).
    Предпочитаются варианты, сохраняющие крайние значения блока точно (чистый белый/чёрный остаётся чистым).
    Если блок можно воспроизвести точно (<= 8 разных значений), ищется пара границ, дающая нулевую ошибку,
    поэтому блоки, полученные из DXT5, перекодируются без потерь."""
    px = list(px)
    hi_all, lo_all = max(px), min(px)
    if hi_all == lo_all:                      # однотонный блок: a0 = a1, все индексы 0
        return bytes([lo_all, lo_all]) + bytes(6)
    mid = [x for x in px if 0 < x < 255]
    cands = [(hi_all, lo_all)]
    cands.append((min(mid), max(mid)) if mid else (0, 255))
    cands.append((lo_all, hi_all))
    best = None
    for a0, a1 in cands:
        pal = _palette(a0, a1)
        idx, err = _assign(px, pal)
        key = (not (hi_all in pal and lo_all in pal), err)
        if best is None or key < best[0]:
            best = (key, a0, a1, idx)
    if best[0][1] > 0 and len(set(px)) <= 8:
        pair = _exact_pair(px)
        if pair:
            idx, err = _assign(px, _palette(*pair))
            best = ((False, err), pair[0], pair[1], idx)
    _, a0, a1, idx = best
    bits = 0
    for i, k in enumerate(idx):
        bits |= k << (3 * i)
    return bytes([a0, a1]) + bits.to_bytes(6, 'little')


def paste_alpha(buf, base, alpha, blocks, width):
    """Перекодирует блоки 4x4 (альфа-половина) в буфере; цветовую половину не трогает."""
    for by, bx in sorted(blocks):
        o = base + (by * (width // 4) + bx) * 16
        buf[o:o + 8] = dxt5_alpha_block(alpha[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4].reshape(16))


def _dds_header(fmt, w, h, size):
    fourcc = {0x0B: b'DXT1', 0x0C: b'DXT3', 0x0D: b'DXT5'}[fmt]
    return (b'DDS ' + struct.pack('<7I', 124, 0x81007, h, w, size, 0, 0) + b'\0' * 44 +
            struct.pack('<2I4s5I', 32, 4, fourcc, 0, 0, 0, 0, 0) + struct.pack('<5I', 0x1000, 0, 0, 0, 0))


class Iwi:
    def __init__(self, data):
        data = bytes(data)
        if len(data) < HDR or data[:3] != b'IWi':
            raise ValueError('Это не IWi: нет заголовка "IWi".\nПервые байты: %s' % data[:16].hex(' '))
        self.data = data
        self.version = data[3]
        if self.version != 6:
            raise ValueError('IWi версии %d. Поддерживается только версия 6 (CoD4).' % self.version)
        self.fmt, self.flags = data[4], data[5]
        self.w, self.h, self.depth = struct.unpack('<3H', data[6:12])
        self.sizes = struct.unpack('<4I', data[12:28])
        hexhead = data[:HDR].hex(' ')
        if self.fmt not in FORMATS:
            raise ValueError('Неизвестный формат пикселей 0x%02X в заголовке IWi.\nЗаголовок: %s' % (self.fmt, hexhead))
        if not (1 <= self.w <= 8192 and 1 <= self.h <= 8192):
            raise ValueError('Странный размер картинки %dx%d — заголовок разобран неверно?\nЗаголовок: %s'
                             % (self.w, self.h, hexhead))
        body = len(data) - HDR
        top = level_size(self.fmt, self.w, self.h)
        if body < top:
            raise ValueError('Файл короче одного изображения %dx%d %s (нужно %d байт данных, есть %d).\nЗаголовок: %s'
                             % (self.w, self.h, FORMATS[self.fmt], top, body, hexhead))
        self.mips, cum, k = None, 0, 0
        while True:
            wk, hk = max(1, self.w >> k), max(1, self.h >> k)
            cum += level_size(self.fmt, wk, hk)
            k += 1
            if cum == body:
                self.mips = k
                break
            if cum > body or (wk == 1 and hk == 1):
                break
        self.top_off = len(data) - top
        self.top_size = top

    # --- устройство файла
    @property
    def fmt_name(self):
        return FORMATS[self.fmt]

    def describe(self):
        mips = ('мипов: %d' % self.mips) if self.mips else 'мипы не распознаны (правится только верхний уровень)'
        return 'IWi v%d, %s, %dx%d, %s' % (self.version, self.fmt_name, self.w, self.h, mips)

    def level_off(self, k):
        if k == 0 or not self.mips:
            return self.top_off
        return HDR + sum(level_size(self.fmt, max(1, self.w >> j), max(1, self.h >> j)) for j in range(k + 1, self.mips))

    def top(self):
        return self.data[self.top_off:]

    # --- просмотр
    def rgba(self):
        top, f, w, h = self.top(), self.fmt, self.w, self.h
        if f in BLOCK:
            im = Image.open(io.BytesIO(_dds_header(f, w, h, len(top)) + top))
            im.load()
            return np.asarray(im.convert('RGBA')).copy()
        a = np.frombuffer(top, np.uint8)
        if f == 0x04:
            g = a.reshape(h, w)
            return np.stack([np.full_like(g, 255)] * 3 + [g], -1)
        if f == 0x01:
            return a.reshape(h, w, 4)[..., [2, 1, 0, 3]].copy()
        if f == 0x02:
            b = a.reshape(h, w, 3)[..., ::-1]
            return np.concatenate([b, np.full((h, w, 1), 255, np.uint8)], -1)
        ga = a.reshape(h, w, 2)
        return np.stack([ga[..., 0]] * 3 + [ga[..., 1]], -1)

    def plane(self):
        """(массив h x w uint8, можно ли править, описание). Для шрифтов это альфа-канал."""
        top, f, w, h = self.top(), self.fmt, self.w, self.h
        if f == 0x0D:
            return dxt5_alpha_decode(top, w, h), (w % 4 == 0 and h % 4 == 0), 'альфа-канал DXT5'
        if f == 0x04:
            return np.frombuffer(top, np.uint8).reshape(h, w).copy(), True, 'альфа-канал A8'
        rgba = self.rgba()
        al = rgba[..., 3]
        if f in (0x01, 0x03, 0x0C) and al.min() < 255:
            return al.copy(), False, 'альфа-канал (только просмотр)'
        return np.asarray(Image.fromarray(rgba[..., :3]).convert('L')).copy(), False, 'яркость (только просмотр)'

    # --- запись
    @staticmethod
    def _box(a, f):
        h, w = a.shape
        return np.floor(a.reshape(h // f, f, w // f, f).astype(np.float64).mean((1, 3)) + 0.5).astype(np.uint8)

    def with_plane(self, new, old):
        """Возвращает (новые байты файла, число перекодированных блоков верхнего уровня)."""
        new = np.ascontiguousarray(new, np.uint8)
        buf = bytearray(self.data)
        w, h = self.w, self.h
        if self.fmt == 0x0D:
            if w % 4 or h % 4:
                raise ValueError('Размер не кратен 4 — править DXT5 нельзя.')
            changed = (new != old).reshape(h // 4, 4, w // 4, 4).any((1, 3))
            n_top = int(changed.sum())
            levels = self.mips or 1
            for k in range(levels):
                f = 1 << k
                wk, hk = w >> k, h >> k
                if k and (w % f or h % f or wk < 4 or hk < 4 or wk % 4 or hk % 4):
                    break
                src = new if k == 0 else self._box(new, f)
                mask = changed if k == 0 else changed.reshape(hk // 4, f, wk // 4, f).any((1, 3))
                base = self.level_off(k)
                for by, bx in zip(*np.nonzero(mask)):
                    o = base + (int(by) * (wk // 4) + int(bx)) * 16
                    buf[o:o + 8] = dxt5_alpha_block(src[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4].reshape(16))
            return bytes(buf), n_top
        if self.fmt == 0x04:
            n_top = int((new != old).sum())
            buf[self.top_off:] = new.tobytes()
            for k in range(1, self.mips or 1):
                f = 1 << k
                if w % f or h % f:
                    break
                o = self.level_off(k)
                lvl = self._box(new, f)
                buf[o:o + lvl.size] = lvl.tobytes()
            return bytes(buf), n_top
        raise ValueError('Формат %s открыт только для просмотра.' % self.fmt_name)


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == 'info':
        i = Iwi(open(a[1], 'rb').read())
        print(i.describe(), '| флаги 0x%02X' % i.flags, '| размеры picmip', i.sizes)
        print('верхний уровень: смещение', i.top_off, 'размер', i.top_size)
    elif len(a) >= 3 and a[0] == 'png':
        i = Iwi(open(a[1], 'rb').read())
        Image.fromarray(i.rgba(), 'RGBA').save(a[2])
        print(i.describe(), '->', a[2])
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
