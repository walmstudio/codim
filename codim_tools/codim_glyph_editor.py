#!/usr/bin/env python3
"""codim_glyph_editor — GTK3-редактор шрифтовых глифов CoD4 PS3 (code_post_gfx*.ff).

  python3 codim_glyph_editor.py [файл.ff]

Лежит рядом с codim_cod4loc.py, codim_fontlib.py и codim_build_font.py (берёт из них
распаковку зоны, разбор таблиц глифов и DXT5-кодек).

Что умеет
  * показывает атлас fonts/gamefonts_ps3 (512x1024, DXT5, глифы в альфа-канале) с зумом и сеткой пикселей;
  * рамки всех глифов выбранного шрифта; клик по атласу выбирает глиф, список слева — по коду;
  * рисование кистью/ластиком прямо в атласе, пипетка, undo/redo;
  * правка метрик (x0, y0, dx, w, h) и прямоугольника глифа в атласе (числами или рамкой мышью);
  * импорт/экспорт PNG: всего атласа и отдельного глифа;
  * предпросмотр произвольного текста выбранным шрифтом (кодировка cp1251);
  * сохранение в .ff: перекодируются только изменённые блоки 4x4, остальное — байт в байт;
  * открывает .iwi (CoD4, v6) как изображение: DXT5/A8 можно править и сохранять обратно в .iwi,
    остальные форматы — только просмотр и экспорт в PNG; атлас из .iwi можно импортировать в .ff.

Управление на холсте: колесо — зум, средняя/правая кнопка — перетаскивание,
клавиши V/B/E/R/I — инструменты, F — вписать, 0 — 100%, +/- — зум."""
import sys, os, re, struct, time, io, json, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
from gi.repository import Gtk, Gdk, GLib
import cairo

import codim_cod4loc as c
import codim_fontlib
sys.modules.setdefault('fontlib', codim_fontlib)      # codim_build_font импортирует его под коротким именем
from codim_fontlib import parse, rect as uv_rect, W, H, FONTS
from codim_build_font import dxt5_alpha_decode
import codim_iwi

META = [('x0', -128, 127, 'x0 (сдвиг по X)'), ('y0', -128, 127, 'y0 (сдвиг по Y)'),
        ('dx', 0, 255, 'dx (шаг)'), ('w', 0, 255, 'w (ширина)'), ('h', 0, 255, 'h (высота)')]
RECT = [('left', 0, W, 'левый'), ('top', 0, H, 'верхний'), ('right', 0, W, 'правый'), ('bottom', 0, H, 'нижний')]
TOOLS = [('select', 'Выбор (V)'), ('pencil', 'Кисть (B)'), ('erase', 'Ластик (E)'),
         ('rect', 'Рамка глифа (R)'), ('pick', 'Пипетка (I)')]
UNDO_LIMIT = 60


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def gfields(g):
    return (g['x0'], g['y0'], g['dx'], g['w'], g['h'], tuple(g['rect']))


def glyph_char(code):
    if code < 0x20 or code == 0x7f:
        return '·'
    if code < 0x7f:
        return chr(code)
    if code < 0x100:
        try:
            return bytes([code]).decode('cp1251')
        except UnicodeDecodeError:
            return '·'
    return chr(code)


def f32(uv):
    return tuple(struct.unpack('>4f', struct.pack('>4f', *uv)))


def glyph_uv(g, w=W, h=H):
    """UV, которые надо записать в запись глифа: исходные, если рамку не трогали,
    UV из применённого набора, если рамка совпадает с ним, иначе пересчёт из пикселей."""
    r = tuple(g['rect'])
    if r == g['orig'][5]:
        return g['uv']
    us = g.get('uv_set')
    if us and uv_rect({'uv': us}) == r:
        return us
    return (r[0] / w, r[1] / h, r[2] / w, r[3] / h)


def text_to_codes(text):
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ord(ch))
        else:
            try:
                out.append(ch.encode('cp1251')[0])
            except UnicodeEncodeError:
                pass
    return out


# ------------------------------------------------------------------ модель

class Doc:
    """Зона .ff + разобранные шрифты + редактируемая копия альфа-канала атласа."""

    def __init__(self, path):
        self.path = path
        self.kind, self.W, self.H, self.editable = 'ff', W, H, True
        z = bytes(c.ff_to_zone(path))
        m = next((x for x in re.finditer(rb'gamefonts_ps3\x00', z) if z[x.start() - 8:x.start()] == b'\xff' * 8), None)
        if not m:
            raise ValueError('В зоне не найдена текстура fonts/gamefonts_ps3.\n'
                             'Это точно code_post_gfx*.ff от CoD4 PS3?')
        self.zone = z
        self.tex0 = m.end()
        self.orig_alpha = dxt5_alpha_decode(z[self.tex0:self.tex0 + W * H], W, H)
        self.alpha = self.orig_alpha.copy()
        names = [n for n in FONTS if re.search(rb'fonts/' + n.encode() + rb'\x00', z)]
        if not names:
            raise ValueError('В зоне не найдено ни одной таблицы глифов (fonts/bigFont и т.п.).')
        raw = parse(z, '>', names)
        self.fonts = {}
        for n in names:
            gl = []
            for g in raw[n]['gl']:
                d = dict(code=g['let'], x0=g['x0'], y0=g['y0'], dx=g['dx'], w=g['pw'], h=g['ph'],
                         uv=tuple(g['uv']), rect=list(uv_rect(g)), off=g['off'])
                d['orig'] = gfields(d)
                gl.append(d)
            self.fonts[n] = dict(pixh=raw[n]['pixh'], gl=gl)
        self.info = 'Загружено: %s  (шрифтов: %d, атлас %dx%d)' % (os.path.basename(path), len(self.fonts), W, H)

    def export_pack(self, path):
        """Набор глифов: zip с atlas.png, glyphs.json (формат extract_glyphs.py) и pack.json."""
        js = {}
        for fn, f in self.fonts.items():
            js[fn] = {'pixel_height': f['pixh'], 'glyphs': [
                {'code': g['code'], 'char': chr(g['code']) if 32 <= g['code'] < 127 else None,
                 'x0': g['x0'], 'y0': g['y0'], 'dx': g['dx'], 'w': g['w'], 'h': g['h'],
                 'uv': list(glyph_uv(g)), 'rect': list(g['rect'])} for g in f['gl']]}
        png = io.BytesIO()
        Image.fromarray(self.alpha).save(png, 'PNG')
        meta = {'format': 1, 'atlas': [W, H], 'source': os.path.basename(self.path),
                'fonts': {fn: len(f['gl']) for fn, f in self.fonts.items()}}
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('pack.json', json.dumps(meta, ensure_ascii=False, indent=1))
            z.writestr('atlas.png', png.getvalue())
            z.writestr('glyphs.json', json.dumps(js, ensure_ascii=False, indent=1))
        return sum(len(f['gl']) for f in self.fonts.values())

    def save(self, path):
        zone = bytearray(self.zone)
        diff = (self.alpha != self.orig_alpha).reshape(H // 4, 4, W // 4, 4).any((1, 3))
        blocks = {(int(a), int(b)) for a, b in zip(*np.nonzero(diff))}
        codim_iwi.paste_alpha(zone, self.tex0, self.alpha, blocks, W)
        n_rec = 0
        for f in self.fonts.values():
            for g in f['gl']:
                if gfields(g) == g['orig']:
                    continue
                rec = bytearray(self.zone[g['off']:g['off'] + 24])
                rec[2:4] = struct.pack('bb', clamp(g['x0'], -128, 127), clamp(g['y0'], -128, 127))
                rec[4], rec[5], rec[6] = clamp(g['dx'], 0, 255), clamp(g['w'], 0, 255), clamp(g['h'], 0, 255)
                rec[8:24] = struct.pack('>4f', *glyph_uv(g))
                zone[g['off']:g['off'] + 24] = rec
                n_rec += 1
        c.zone_to_ff(zone, path)
        # новая «чистая» точка отсчёта: то, что реально лежит в зоне (DXT5 квантует альфу)
        self.zone = bytes(zone)
        self.orig_alpha = dxt5_alpha_decode(self.zone[self.tex0:self.tex0 + W * H], W, H)
        self.alpha = self.orig_alpha.copy()
        for f in self.fonts.values():
            for g in f['gl']:
                if gfields(g) != g['orig']:
                    g['uv'] = f32(glyph_uv(g))
                    g.pop('uv_set', None)
                    g['orig'] = gfields(g)
        return len(blocks), n_rec


class IwiDoc:
    """Текстура .iwi как документ без таблицы глифов. Редактируется альфа-плоскость (DXT5 / A8)."""

    def __init__(self, path):
        self.path = path
        self.kind = 'iwi'
        self.raw = open(path, 'rb').read()
        self.iwi = codim_iwi.Iwi(self.raw)
        self.alpha, self.editable, desc = self.iwi.plane()
        self.orig_alpha = self.alpha.copy()
        self.W, self.H = self.iwi.w, self.iwi.h
        self.fonts = {}
        self.info = '%s — показано: %s%s' % (self.iwi.describe(), desc,
                                              '' if self.editable else '. Правка недоступна, можно экспортировать в PNG')

    def save(self, path):
        if not self.editable:
            raise ValueError('Этот формат (%s) открыт только для просмотра.' % self.iwi.fmt_name)
        data, nb = self.iwi.with_plane(self.alpha, self.orig_alpha)
        with open(path, 'wb') as f:
            f.write(data)
        self.raw, self.iwi = data, codim_iwi.Iwi(data)
        self.alpha = self.iwi.plane()[0]          # то, что реально лежит в файле (DXT5 квантует альфу)
        self.orig_alpha = self.alpha.copy()
        return nb, 0


# ------------------------------------------------------------------ холст

class Canvas(Gtk.DrawingArea):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0
        self.aw, self.ah = W, H
        self._alloc(W, H)
        self.panning = None
        self.stroke_last = None
        self.rect_drag = None
        self.hover = None
        self._fitted = False
        self._masks = {}
        self.set_can_focus(True)
        self.set_size_request(420, 420)
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.POINTER_MOTION_MASK | Gdk.EventMask.SCROLL_MASK |
                        Gdk.EventMask.SMOOTH_SCROLL_MASK | Gdk.EventMask.KEY_PRESS_MASK |
                        Gdk.EventMask.LEAVE_NOTIFY_MASK)
        self.connect('draw', self.on_draw)
        self.connect('button-press-event', self.on_press)
        self.connect('button-release-event', self.on_release)
        self.connect('motion-notify-event', self.on_motion)
        self.connect('scroll-event', self.on_scroll)
        self.connect('key-press-event', self.on_key)
        self.connect('leave-notify-event', self.on_leave)

    # --- данные
    def _alloc(self, w, h):
        self.aw, self.ah = w, h
        stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, w)
        self.a8 = bytearray(stride * h)
        self.view = np.frombuffer(self.a8, np.uint8).reshape(h, stride)[:, :w]
        self.surf = cairo.ImageSurface.create_for_data(self.a8, cairo.FORMAT_A8, w, h, stride)

    def set_atlas(self, alpha):
        h, w = alpha.shape
        if (w, h) != (self.aw, self.ah):
            self._alloc(w, h)
        self.view[:] = alpha
        self.surf.mark_dirty()
        self.queue_draw()

    def refresh_region(self, x0, y0, x1, y1):
        self.view[y0:y1, x0:x1] = self.app.doc.alpha[y0:y1, x0:x1]
        self.surf.mark_dirty()
        self.queue_draw()
        self.app.preview.queue_draw()

    # --- вид
    def to_atlas(self, sx, sy):
        return (sx - self.ox) / self.zoom, (sy - self.oy) / self.zoom

    def fit(self):
        a = self.get_allocation()
        if a.width < 20 or a.height < 20:
            return
        self.zoom = min(a.width / self.aw, a.height / self.ah) * 0.98
        self.ox = (a.width - self.aw * self.zoom) / 2
        self.oy = (a.height - self.ah * self.zoom) / 2
        self.queue_draw()
        self.app.update_zoom_label()

    def set_100(self):
        a = self.get_allocation()
        self.zoom = 1.0
        self.ox, self.oy = (a.width - self.aw) / 2, 0
        self.queue_draw()
        self.app.update_zoom_label()

    def zoom_at(self, sx, sy, f):
        ax, ay = self.to_atlas(sx, sy)
        self.zoom = clamp(self.zoom * f, 0.2, 48.0)
        self.ox, self.oy = sx - ax * self.zoom, sy - ay * self.zoom
        self.queue_draw()
        self.app.update_zoom_label()

    # --- рисование
    def on_draw(self, _w, cr):
        a = self.get_allocation()
        if not self._fitted and a.width > 50:
            self._fitted = True
            self.fit()
        cr.set_source_rgb(0.16, 0.16, 0.18)
        cr.paint()
        app = self.app
        if not app.doc:
            return False
        z = self.zoom
        cr.save()
        cr.translate(self.ox, self.oy)
        cr.scale(z, z)
        cr.rectangle(0, 0, self.aw, self.ah)
        cr.set_source_rgb(0, 0, 0)
        cr.fill()
        pat = cairo.SurfacePattern(self.surf)
        pat.set_filter(cairo.FILTER_NEAREST if z >= 1.5 else cairo.FILTER_GOOD)
        cr.set_source_rgb(1, 1, 1)
        cr.mask(pat)
        # сетка пикселей
        if z >= 8:
            x_lo, x_hi = max(0, int(-self.ox / z)), min(self.aw, int((a.width - self.ox) / z) + 1)
            y_lo, y_hi = max(0, int(-self.oy / z)), min(self.ah, int((a.height - self.oy) / z) + 1)
            cr.set_line_width(1 / z)
            cr.set_source_rgba(0.5, 0.5, 0.6, 0.28)
            for x in range(x_lo, x_hi + 1):
                cr.move_to(x, y_lo); cr.line_to(x, y_hi)
            for y in range(y_lo, y_hi + 1):
                cr.move_to(x_lo, y); cr.line_to(x_hi, y)
            cr.stroke()
        # рамки глифов
        gl = app.cur_gl()
        if app.show_rects and gl:
            cr.set_line_width(1 / z)
            cr.set_source_rgba(0.2, 0.7, 1.0, 0.55)
            for i, g in enumerate(gl):
                l, t, r, b = g['rect']
                if r > l and b > t and i != app.sel:
                    cr.rectangle(l, t, r - l, b - t)
            cr.stroke()
        sg = app.sel_glyph()
        if sg:
            l, t, r, b = sg['rect']
            if r > l and b > t:
                cr.set_line_width(2 / z)
                cr.set_source_rgba(1.0, 0.8, 0.1, 1)
                cr.rectangle(l, t, r - l, b - t)
                cr.stroke()
        if self.rect_drag:
            x0, y0, x1, y1 = self.rect_drag
            cr.set_line_width(2 / z)
            cr.set_source_rgba(0.3, 1.0, 0.4, 1)
            cr.rectangle(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            cr.stroke()
        if self.hover and app.tool in ('pencil', 'erase'):
            hx, hy = int(np.floor(self.hover[0])), int(np.floor(self.hover[1]))
            r = app.brush_size
            cr.set_line_width(1 / z)
            cr.set_source_rgba(1, 0.3, 0.3, 0.9)
            cr.rectangle(hx - r // 2, hy - r // 2, r, r)
            cr.stroke()
        cr.set_line_width(1 / z)
        cr.set_source_rgba(1, 1, 1, 0.35)
        cr.rectangle(0, 0, self.aw, self.ah)
        cr.stroke()
        cr.restore()
        return False

    # --- кисть
    def brush_mask(self, r):
        m = self._masks.get(r)
        if m is None:
            if r <= 2:
                m = np.ones((r, r), bool)
            else:
                yy, xx = np.mgrid[:r, :r]
                cc = (r - 1) / 2
                m = (xx - cc) ** 2 + (yy - cc) ** 2 <= (r / 2) ** 2
            self._masks[r] = m
        return m

    def paint(self, ax0, ay0, ax1, ay1):
        app = self.app
        val = 0 if app.tool == 'erase' else app.brush_val
        r = app.brush_size
        mask = self.brush_mask(r)
        clip = app.sel_rect() if app.clip_to_glyph else None
        if app.clip_to_glyph and clip is None:
            return
        alpha = app.doc.alpha
        n = int(max(abs(ax1 - ax0), abs(ay1 - ay0))) + 1
        bx0, by0, bx1, by1 = self.aw, self.ah, 0, 0
        for i in range(n + 1):
            t = i / n
            ix, iy = int(np.floor(ax0 + (ax1 - ax0) * t)), int(np.floor(ay0 + (ay1 - ay0) * t))
            left, top = ix - r // 2, iy - r // 2
            x0, y0, x1, y1 = max(0, left), max(0, top), min(self.aw, left + r), min(self.ah, top + r)
            if clip:
                x0, y0, x1, y1 = max(x0, clip[0]), max(y0, clip[1]), min(x1, clip[2]), min(y1, clip[3])
            if x1 <= x0 or y1 <= y0:
                continue
            sub = alpha[y0:y1, x0:x1]
            sub[mask[y0 - top:y1 - top, x0 - left:x1 - left]] = val
            bx0, by0, bx1, by1 = min(bx0, x0), min(by0, y0), max(bx1, x1), max(by1, y1)
        if bx1 > bx0:
            self.refresh_region(bx0, by0, bx1, by1)
            app.set_modified(True)

    # --- события
    def on_press(self, _w, ev):
        self.grab_focus()
        app = self.app
        if ev.button in (2, 3):
            self.panning = (ev.x, ev.y, self.ox, self.oy)
            return True
        if ev.button != 1 or not app.doc:
            return False
        ax, ay = self.to_atlas(ev.x, ev.y)
        t = app.tool
        if t == 'select':
            app.pick_at(ax, ay)
        elif t in ('pencil', 'erase'):
            if not app.can_edit():
                return True
            app.push_undo()
            self.stroke_last = (ax, ay)
            self.paint(ax, ay, ax, ay)
        elif t == 'rect':
            if app.sel is None:
                app.status('Сначала выберите глиф (в списке или инструментом «Выбор»)')
                return True
            p = (clamp(int(round(ax)), 0, self.aw), clamp(int(round(ay)), 0, self.ah))
            self.rect_drag = [p[0], p[1], p[0], p[1]]
        elif t == 'pick':
            app.pick_value(ax, ay)
        return True

    def on_release(self, _w, ev):
        if ev.button in (2, 3):
            self.panning = None
        elif ev.button == 1:
            self.stroke_last = None
            if self.rect_drag:
                x0, y0, x1, y1 = self.rect_drag
                self.rect_drag = None
                if abs(x1 - x0) > 0 and abs(y1 - y0) > 0:
                    self.app.apply_rect([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
                self.queue_draw()
        return True

    def on_motion(self, _w, ev):
        ax, ay = self.to_atlas(ev.x, ev.y)
        self.hover = (ax, ay)
        self.app.update_status(ax, ay)
        if self.panning:
            sx, sy, ox, oy = self.panning
            self.ox, self.oy = ox + ev.x - sx, oy + ev.y - sy
        elif self.stroke_last and (ev.state & Gdk.ModifierType.BUTTON1_MASK):
            self.paint(self.stroke_last[0], self.stroke_last[1], ax, ay)
            self.stroke_last = (ax, ay)
        elif self.rect_drag:
            self.rect_drag[2] = clamp(int(round(ax)), 0, self.aw)
            self.rect_drag[3] = clamp(int(round(ay)), 0, self.ah)
        self.queue_draw()
        return True

    def on_leave(self, _w, _ev):
        self.hover = None
        self.queue_draw()

    def on_scroll(self, _w, ev):
        d = 0.0
        if ev.direction == Gdk.ScrollDirection.UP:
            d = 1.0
        elif ev.direction == Gdk.ScrollDirection.DOWN:
            d = -1.0
        elif ev.direction == Gdk.ScrollDirection.SMOOTH:
            d = -ev.delta_y
        if d:
            self.zoom_at(ev.x, ev.y, 1.25 ** d)
        return True

    def on_key(self, _w, ev):
        k = Gdk.keyval_name(ev.keyval) or ''
        if ev.state & Gdk.ModifierType.CONTROL_MASK:
            return False
        tools = {'v': 'select', 'b': 'pencil', 'e': 'erase', 'r': 'rect', 'i': 'pick'}
        a = self.get_allocation()
        if k.lower() in tools:
            self.app.set_tool(tools[k.lower()])
        elif k.lower() == 'f':
            self.fit()
        elif k == '0':
            self.set_100()
        elif k in ('plus', 'equal', 'KP_Add'):
            self.zoom_at(a.width / 2, a.height / 2, 1.25)
        elif k in ('minus', 'KP_Subtract'):
            self.zoom_at(a.width / 2, a.height / 2, 0.8)
        else:
            return False
        return True


# ------------------------------------------------------------------ окно

class Editor(Gtk.Window):
    def __init__(self, path=None):
        super().__init__(title='codim glyph editor')
        self.set_default_size(1280, 820)
        self.doc = None
        self.cur_font = None
        self.sel = None
        self.tool = 'select'
        self.brush_size, self.brush_val = 1, 255
        self.clip_to_glyph, self.show_rects, self.keep_wh = False, True, True
        self.undo, self.redo = [], []
        self._last_key, self._last_t = None, 0.0
        self.modified = False
        self.out_path = None
        self._updating = False
        self.build_ui()
        self.connect('delete-event', self.on_delete)
        if path:
            GLib.idle_add(self.load, path)

    # ---------------------------------------------------------- интерфейс
    def build_ui(self):
        ag = Gtk.AccelGroup()
        self.add_accel_group(ag)
        hb = Gtk.HeaderBar()
        hb.set_show_close_button(True)
        hb.props.title = 'codim glyph editor'
        self.set_titlebar(hb)
        self.hb = hb

        def hbtn(label, cb, key=None, shift=False):
            b = Gtk.Button(label=label)
            b.connect('clicked', lambda _w: cb())
            if key:
                mods = Gdk.ModifierType.CONTROL_MASK
                if shift:
                    mods |= Gdk.ModifierType.SHIFT_MASK
                b.add_accelerator('clicked', ag, ord(key), mods, Gtk.AccelFlags.VISIBLE)
            return b

        hb.pack_start(hbtn('Открыть…', self.cmd_open, 'o'))
        hb.pack_start(hbtn('Сохранить', self.cmd_save, 's'))
        hb.pack_start(hbtn('Сохранить как…', self.cmd_save_as, 's', True))
        self.btn_undo = hbtn('↶ Отмена', self.cmd_undo, 'z')
        self.btn_redo = hbtn('↷ Повтор', self.cmd_redo, 'y')
        hb.pack_start(self.btn_undo)
        hb.pack_start(self.btn_redo)
        menu = Gtk.Menu()
        for label, cb in [('Экспорт атласа в PNG…', self.cmd_export_atlas),
                          ('Импорт атласа из PNG…', self.cmd_import_atlas),
                          ('Импорт атласа из IWI…', self.cmd_import_iwi),
                          ('Экспорт выбранного глифа в PNG…', self.cmd_export_glyph),
                          ('Импорт PNG в выбранный глиф…', self.cmd_import_glyph),
                          ('Экспорт набора глифов (атлас + таблицы)…', self.cmd_export_pack),
                          ('Применить набор глифов…', self.cmd_apply_pack)]:
            it = Gtk.MenuItem(label=label)
            it.connect('activate', lambda _w, cb=cb: cb())
            menu.append(it)
        menu.show_all()
        mb = Gtk.MenuButton()
        mb.set_label('Импорт / экспорт ▾')
        mb.set_popup(menu)
        hb.pack_end(mb)
        self.update_buttons()

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(root)
        main = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main.set_border_width(6)
        root.pack_start(main, True, True, 0)

        # --- слева: шрифт + список глифов
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        left.set_size_request(230, -1)
        main.pack_start(left, False, False, 0)
        left.pack_start(Gtk.Label(label='Шрифт', xalign=0), False, False, 0)
        self.font_combo = Gtk.ComboBoxText()
        self.font_combo.connect('changed', self.on_font_changed)
        left.pack_start(self.font_combo, False, False, 0)
        self.store = Gtk.ListStore(str, str, str, str)
        self.tree = Gtk.TreeView(model=self.store)
        for i, title in enumerate(['Код', 'Знак', 'w×h', '●']):
            col = Gtk.TreeViewColumn(title, Gtk.CellRendererText(), text=i)
            self.tree.append_column(col)
        self.tree.set_enable_search(False)
        self.tree.get_selection().connect('changed', self.on_tree_sel)
        sw = Gtk.ScrolledWindow()
        sw.add(self.tree)
        left.pack_start(sw, True, True, 0)
        left.pack_start(Gtk.Label(label='● — глиф изменён с последнего сохранения', xalign=0,
                                  wrap=True), False, False, 0)

        # --- центр: панель инструментов + холст
        centre = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        main.pack_start(centre, True, True, 0)
        row1 = Gtk.Box(spacing=4)
        centre.pack_start(row1, False, False, 0)
        self.tool_btns = {}
        first = None
        for name, label in TOOLS:
            b = Gtk.RadioButton.new_with_label_from_widget(first, label)
            b.set_mode(False)
            if first is None:
                first = b
            b.connect('toggled', self.on_tool_toggled, name)
            self.tool_btns[name] = b
            row1.pack_start(b, False, False, 0)
        row2 = Gtk.Box(spacing=8)
        centre.pack_start(row2, False, False, 0)
        row2.pack_start(Gtk.Label(label='Кисть, px:'), False, False, 0)
        sp = Gtk.SpinButton.new_with_range(1, 32, 1)
        sp.connect('value-changed', lambda s: setattr(self, 'brush_size', s.get_value_as_int()))
        row2.pack_start(sp, False, False, 0)
        row2.pack_start(Gtk.Label(label='Яркость:'), False, False, 0)
        self.val_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 255, 1)
        self.val_scale.set_value(255)
        self.val_scale.set_size_request(140, -1)
        self.val_scale.set_digits(0)
        self.val_scale.connect('value-changed', lambda s: setattr(self, 'brush_val', int(s.get_value())))
        row2.pack_start(self.val_scale, False, False, 0)
        ck = Gtk.CheckButton(label='Только внутри рамки глифа')
        ck.connect('toggled', lambda s: setattr(self, 'clip_to_glyph', s.get_active()))
        row2.pack_start(ck, False, False, 0)
        ck2 = Gtk.CheckButton(label='Рамки')
        ck2.set_active(True)
        ck2.connect('toggled', self.on_show_rects)
        row2.pack_start(ck2, False, False, 0)
        row3 = Gtk.Box(spacing=4)
        centre.pack_start(row3, False, False, 0)
        row3.pack_start(Gtk.Label(label='Масштаб:'), False, False, 0)
        for label, cb in [('−', lambda: self.zoom_btn(0.8)), ('+', lambda: self.zoom_btn(1.25)),
                          ('100%', lambda: self.canvas.set_100()), ('Вписать', lambda: self.canvas.fit())]:
            b = Gtk.Button(label=label)
            b.connect('clicked', lambda _w, cb=cb: cb())
            row3.pack_start(b, False, False, 0)
        self.zoom_lbl = Gtk.Label(label='')
        row3.pack_start(self.zoom_lbl, False, False, 6)

        self.canvas = Canvas(self)
        fr = Gtk.Frame()
        fr.add(self.canvas)
        centre.pack_start(fr, True, True, 0)

        # --- справа: свойства + предпросмотр
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        right.set_size_request(300, -1)
        main.pack_start(right, False, False, 0)
        self.glyph_lbl = Gtk.Label(label='Глиф не выбран', xalign=0)
        right.pack_start(self.glyph_lbl, False, False, 0)
        grid = Gtk.Grid(column_spacing=8, row_spacing=4)
        self.spins = {}
        r = 0
        for name, lo, hi, label in META + RECT:
            if name == 'left':
                grid.attach(Gtk.Label(label='Рамка в атласе (px)', xalign=0, margin_top=6), 0, r, 2, 1)
                r += 1
            grid.attach(Gtk.Label(label=label, xalign=0), 0, r, 1, 1)
            s = Gtk.SpinButton.new_with_range(lo, hi, 1)
            s.set_sensitive(False)
            s.connect('value-changed', self.on_prop_changed, name)
            grid.attach(s, 1, r, 1, 1)
            self.spins[name] = s
            r += 1
        right.pack_start(grid, False, False, 0)
        ck3 = Gtk.CheckButton(label='w×h = размер рамки (при её правке)')
        ck3.set_active(True)
        ck3.connect('toggled', lambda s: setattr(self, 'keep_wh', s.get_active()))
        right.pack_start(ck3, False, False, 0)
        b = Gtk.Button(label='Подогнать рамку под пиксели')
        b.connect('clicked', lambda _w: self.cmd_autocrop())
        right.pack_start(b, False, False, 0)
        right.pack_start(Gtk.Label(label='Предпросмотр', xalign=0, margin_top=8), False, False, 0)
        self.prev_entry = Gtk.Entry()
        self.prev_entry.set_text('Новая игра. Настройки: АБВГД абвгд Ёё №')
        self.prev_entry.connect('changed', lambda _w: self.preview.queue_draw())
        right.pack_start(self.prev_entry, False, False, 0)
        ps = Gtk.Box(spacing=6)
        ps.pack_start(Gtk.Label(label='Масштаб:'), False, False, 0)
        self.prev_scale = Gtk.SpinButton.new_with_range(0.5, 4, 0.25)
        self.prev_scale.set_value(1.0)
        self.prev_scale.connect('value-changed', lambda _w: self.preview.queue_draw())
        ps.pack_start(self.prev_scale, False, False, 0)
        right.pack_start(ps, False, False, 0)
        self.preview = Gtk.DrawingArea()
        self.preview.set_size_request(-1, 170)
        self.preview.connect('draw', self.on_preview_draw)
        pf = Gtk.Frame()
        pf.add(self.preview)
        right.pack_start(pf, True, True, 0)

        self.status_lbl = Gtk.Label(label='Откройте code_post_gfx*.ff или .iwi', xalign=0, margin_start=8, margin_bottom=4)
        self.status_lbl.set_ellipsize(3)
        root.pack_start(self.status_lbl, False, False, 0)

    # ---------------------------------------------------------- вспомогательное
    def status(self, text):
        self.status_lbl.set_text(text)

    def cur_gl(self):
        return self.doc.fonts[self.cur_font]['gl'] if self.doc and self.cur_font else []

    def sel_glyph(self):
        gl = self.cur_gl()
        return gl[self.sel] if self.sel is not None and self.sel < len(gl) else None

    def sel_rect(self):
        g = self.sel_glyph()
        if g:
            l, t, r, b = g['rect']
            if r > l and b > t:
                return (l, t, r, b)
        return None

    def update_zoom_label(self):
        self.zoom_lbl.set_text('%d%%' % round(self.canvas.zoom * 100))

    def zoom_btn(self, f):
        a = self.canvas.get_allocation()
        self.canvas.zoom_at(a.width / 2, a.height / 2, f)

    def update_status(self, ax, ay):
        if not self.doc:
            return
        ix, iy = int(np.floor(ax)), int(np.floor(ay))
        if 0 <= ix < self.doc.W and 0 <= iy < self.doc.H:
            txt = 'x=%d y=%d  альфа=%d' % (ix, iy, self.doc.alpha[iy, ix])
            under = [g for g in self.cur_gl() if g['rect'][0] <= ix < g['rect'][2] and g['rect'][1] <= iy < g['rect'][3]]
            if under:
                g = min(under, key=lambda g: (g['rect'][2] - g['rect'][0]) * (g['rect'][3] - g['rect'][1]))
                txt += '  | глиф 0x%02X «%s»' % (g['code'], glyph_char(g['code']))
            self.status(txt)

    def set_modified(self, flag):
        self.modified = flag
        name = os.path.basename(self.doc.path) if self.doc else ''
        self.hb.props.title = 'codim glyph editor'
        self.hb.props.subtitle = (name + (' *' if flag else '')) if name else ''

    def can_edit(self):
        if self.doc and self.doc.editable:
            return True
        self.status('Этот файл открыт только для просмотра: формат не поддерживает правку. Можно экспортировать в PNG.')
        return False

    def update_buttons(self):
        self.btn_undo.set_sensitive(bool(self.undo))
        self.btn_redo.set_sensitive(bool(self.redo))

    def message(self, text, kind=Gtk.MessageType.INFO):
        d = Gtk.MessageDialog(transient_for=self, modal=True, message_type=kind,
                              buttons=Gtk.ButtonsType.OK, text=text)
        d.run()
        d.destroy()

    def file_dialog(self, title, save=False, suggest=None, patterns=None):
        d = Gtk.FileChooserDialog(title=title, transient_for=self,
                                  action=Gtk.FileChooserAction.SAVE if save else Gtk.FileChooserAction.OPEN)
        d.add_buttons('Отмена', Gtk.ResponseType.CANCEL, 'Сохранить' if save else 'Открыть', Gtk.ResponseType.OK)
        if save:
            d.set_do_overwrite_confirmation(True)
        if suggest:
            d.set_current_name(suggest)
        if patterns:
            f = Gtk.FileFilter()
            f.set_name(' / '.join(patterns))
            for p in patterns:
                f.add_pattern(p)
            d.add_filter(f)
            fa = Gtk.FileFilter()
            fa.set_name('Все файлы')
            fa.add_pattern('*')
            d.add_filter(fa)
        path = d.get_filename() if d.run() == Gtk.ResponseType.OK else None
        d.destroy()
        return path

    # ---------------------------------------------------------- undo/redo
    def snapshot(self):
        return (self.doc.alpha.copy(),
                {fn: [gfields(g) for g in f['gl']] for fn, f in self.doc.fonts.items()})

    def restore(self, snap):
        a, fields = snap
        self.doc.alpha[:] = a
        for fn, lst in fields.items():
            for g, f in zip(self.doc.fonts[fn]['gl'], lst):
                g['x0'], g['y0'], g['dx'], g['w'], g['h'] = f[:5]
                g['rect'] = list(f[5])
        self.canvas.set_atlas(self.doc.alpha)
        self.fill_list()
        self.refresh_panel()
        self.preview.queue_draw()
        self.set_modified(True)

    def push_undo(self, key=None):
        now = time.monotonic()
        if key is not None and key == self._last_key and now - self._last_t < 1.5:
            self._last_t = now
            return
        self.undo.append(self.snapshot())
        del self.undo[:-UNDO_LIMIT]
        self.redo.clear()
        self._last_key, self._last_t = key, now
        self.update_buttons()

    def cmd_undo(self):
        if not self.doc or not self.undo:
            return
        self.redo.append(self.snapshot())
        self.restore(self.undo.pop())
        self._last_key = None
        self.update_buttons()

    def cmd_redo(self):
        if not self.doc or not self.redo:
            return
        self.undo.append(self.snapshot())
        self.restore(self.redo.pop())
        self._last_key = None
        self.update_buttons()

    # ---------------------------------------------------------- файлы
    def confirm_discard(self):
        if not self.modified:
            return True
        d = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.QUESTION,
                              text='Есть несохранённые изменения.')
        d.add_buttons('Отмена', Gtk.ResponseType.CANCEL, 'Не сохранять', Gtk.ResponseType.REJECT,
                      'Сохранить', Gtk.ResponseType.ACCEPT)
        resp = d.run()
        d.destroy()
        if resp == Gtk.ResponseType.ACCEPT:
            return self.cmd_save()
        return resp == Gtk.ResponseType.REJECT

    def on_delete(self, _w, _ev):
        return not self.confirm_discard()

    def cmd_open(self):
        if not self.confirm_discard():
            return
        p = self.file_dialog('Открыть .ff или .iwi', patterns=['*.ff', '*.iwi', '*.FF', '*.IWI'])
        if p:
            self.load(p, force=True)

    def load(self, path, force=False):
        if not force and not self.confirm_discard():
            return
        try:
            with open(path, 'rb') as f:
                magic = f.read(3)
            doc = IwiDoc(path) if magic == b'IWi' else Doc(path)
        except Exception as e:
            self.message('Не удалось открыть файл:\n%s' % e, Gtk.MessageType.ERROR)
            return
        self.doc = doc
        self.out_path = None
        self.undo.clear()
        self.redo.clear()
        self.sel = None
        self.cur_font = None
        for name, s in self.spins.items():
            if name in ('left', 'right'):
                s.set_range(0, doc.W)
            elif name in ('top', 'bottom'):
                s.set_range(0, doc.H)
        self.canvas.set_atlas(doc.alpha)
        self._updating = True
        self.font_combo.remove_all()
        for n in doc.fonts:
            self.font_combo.append_text(n)
        self._updating = False
        self.font_combo.set_active(0)
        if not doc.fonts:
            self.store.clear()
        self.refresh_panel()
        self.preview.queue_draw()
        self.canvas.fit()
        self.set_modified(False)
        self.update_buttons()
        self.status(doc.info)

    def cmd_save(self):
        if not self.doc or not self.can_edit():
            return False
        if not self.out_path:
            return self.cmd_save_as()
        return self.do_save(self.out_path)

    def cmd_save_as(self):
        if not self.doc or not self.can_edit():
            return False
        base = os.path.basename(self.doc.path)
        ext = '.' + self.doc.kind
        stem = base[:-len(ext)] if base.lower().endswith(ext) else base
        p = self.file_dialog('Сохранить как ' + ext, save=True, suggest=stem + '_edited' + ext, patterns=['*' + ext])
        return self.do_save(p) if p else False

    def do_save(self, path):
        self.status('Сохранение…')
        while Gtk.events_pending():
            Gtk.main_iteration()
        try:
            nb, nr = self.doc.save(path)
        except Exception as e:
            self.message('Не удалось сохранить:\n%s' % e, Gtk.MessageType.ERROR)
            return False
        self.out_path = path
        self.canvas.set_atlas(self.doc.alpha)
        self.fill_list()
        self.set_modified(False)
        self.preview.queue_draw()
        self.status('Сохранено: %s  (перекодировано блоков: %d, записей глифов: %d)' % (path, nb, nr))
        return True

    def cmd_export_pack(self):
        if not self.doc or self.doc.kind != 'ff':
            self.message('Набор глифов собирается из .ff с таблицами шрифтов. В режиме .iwi таблиц нет, '
                         'используйте экспорт атласа в PNG.')
            return
        base = os.path.basename(self.doc.path)
        stem = base[:-3] if base.lower().endswith('.ff') else base
        p = self.file_dialog('Экспорт набора глифов', save=True, suggest=stem + '_glyphs.zip', patterns=['*.zip'])
        if not p:
            return
        try:
            n = self.doc.export_pack(p)
        except Exception as e:
            self.message('Не удалось сохранить набор:\n%s' % e, Gtk.MessageType.ERROR)
            return
        self.status('Набор глифов сохранён: %s  (шрифтов: %d, глифов: %d, с текущими несохранёнными правками)'
                    % (p, len(self.doc.fonts), n))

    def cmd_apply_pack(self):
        if not self.doc or not self.can_edit():
            return
        if self.doc.kind != 'ff':
            self.message('Набор глифов применяется к .ff с таблицами шрифтов.')
            return
        p = self.file_dialog('Применить набор глифов', patterns=['*.zip'])
        if not p:
            return
        try:
            with zipfile.ZipFile(p) as z:
                js = json.loads(z.read('glyphs.json').decode('utf-8'))
                atlas = self.image_to_alpha(Image.open(io.BytesIO(z.read('atlas.png'))))
                meta = json.loads(z.read('pack.json').decode('utf-8')) if 'pack.json' in z.namelist() else {}
        except Exception as e:
            self.message('Не удалось прочитать набор глифов:\n%s' % e, Gtk.MessageType.ERROR)
            return
        if atlas.shape != self.doc.alpha.shape:
            self.message('Атлас в наборе %dx%d, а в открытом файле %dx%d — применить нельзя.'
                         % (atlas.shape[1], atlas.shape[0], self.doc.W, self.doc.H), Gtk.MessageType.ERROR)
            return
        common = [fn for fn in self.doc.fonts if fn in js]
        if not common:
            self.message('В наборе нет ни одного шрифта из открытого файла.', Gtk.MessageType.ERROR)
            return
        d = Gtk.Dialog(title='Применить набор глифов', transient_for=self, modal=True)
        d.add_buttons('Отмена', Gtk.ResponseType.CANCEL, 'Применить', Gtk.ResponseType.OK)
        box = d.get_content_area()
        box.set_spacing(6)
        box.set_border_width(10)
        box.add(Gtk.Label(label='Источник: %s\nАтлас %dx%d, шрифтов в наборе: %d'
                          % (meta.get('source', os.path.basename(p)), atlas.shape[1], atlas.shape[0], len(js)), xalign=0))
        ck_atlas = Gtk.CheckButton(label='Заменить атлас целиком')
        ck_atlas.set_active(True)
        ck_tab = Gtk.CheckButton(label='Применить таблицы глифов (рамки, x0, y0, dx, w, h) — по коду символа')
        ck_tab.set_active(True)
        box.add(ck_atlas)
        box.add(ck_tab)
        box.add(Gtk.Label(label='Шрифты, у которых применять таблицы:', xalign=0, margin_top=4))
        cks = {}
        for fn in common:
            cks[fn] = Gtk.CheckButton(label=fn)
            cks[fn].set_active(True)
            cks[fn].set_margin_start(16)
            box.add(cks[fn])
        box.add(Gtk.Label(label='Таблицы без атласа (и наоборот) дают сбитые рамки: обычно нужно и то и другое.\n'
                                'После применения можно откатить кнопкой «↶ Отмена» (Ctrl+Z).', xalign=0, margin_top=4))
        d.show_all()
        ok = d.run() == Gtk.ResponseType.OK
        want_atlas, want_tab = ck_atlas.get_active(), ck_tab.get_active()
        fonts = [fn for fn in common if cks[fn].get_active()]
        d.destroy()
        if not ok or not (want_atlas or (want_tab and fonts)):
            return
        self.push_undo()
        report = []
        if want_atlas:
            self.doc.alpha[:] = atlas
            self.canvas.set_atlas(self.doc.alpha)
        if want_tab:
            for fn in fonts:
                src = {}
                for g in js[fn]['glyphs']:
                    src.setdefault(g['code'], g)
                done = 0
                for g in self.doc.fonts[fn]['gl']:
                    q = src.get(g['code'])
                    if q is None:
                        continue
                    g['x0'], g['y0'] = clamp(int(q['x0']), -128, 127), clamp(int(q['y0']), -128, 127)
                    g['dx'], g['w'], g['h'] = (clamp(int(q[k]), 0, 255) for k in ('dx', 'w', 'h'))
                    g['rect'] = [clamp(int(v), 0, lim) for v, lim in zip(q['rect'], (self.doc.W, self.doc.H) * 2)]
                    if q.get('uv'):
                        g['uv_set'] = tuple(float(v) for v in q['uv'])
                    done += 1
                miss_t = len(self.doc.fonts[fn]['gl']) - done
                miss_s = len(src) - done
                report.append('%s: %d' % (fn, done) + (' (не нашлось в наборе: %d)' % miss_t if miss_t else '') +
                              (' (лишних в наборе: %d)' % miss_s if miss_s else ''))
        self.fill_list()
        self.refresh_panel()
        self.preview.queue_draw()
        self.canvas.queue_draw()
        self.set_modified(True)
        self.status('Набор применён из %s. %s%s' % (os.path.basename(p), 'Атлас заменён. ' if want_atlas else '',
                                                    ('Глифов обновлено — ' + '; '.join(report)) if report else ''))

    def cmd_export_atlas(self):
        if not self.doc:
            return
        p = self.file_dialog('Экспорт атласа', save=True, suggest='gamefonts_atlas.png', patterns=['*.png'])
        if p:
            Image.fromarray(self.doc.alpha).save(p)
            self.status('Атлас сохранён: ' + p)

    def cmd_import_atlas(self):
        if not self.doc or not self.can_edit():
            return
        p = self.file_dialog('Импорт атласа', patterns=['*.png'])
        if not p:
            return
        im = Image.open(p)
        if im.size != (self.doc.W, self.doc.H):
            self.message('Нужен PNG %dx%d, а в файле %dx%d.' % (self.doc.W, self.doc.H, *im.size),
                         Gtk.MessageType.ERROR)
            return
        self.push_undo()
        self.doc.alpha[:] = self.image_to_alpha(im)
        self.canvas.set_atlas(self.doc.alpha)
        self.preview.queue_draw()
        self.set_modified(True)

    def cmd_import_iwi(self):
        if not self.doc or not self.can_edit():
            return
        p = self.file_dialog('Импорт атласа из .iwi', patterns=['*.iwi', '*.IWI'])
        if not p:
            return
        try:
            src = IwiDoc(p)
        except Exception as e:
            self.message('Не удалось прочитать .iwi:\n%s' % e, Gtk.MessageType.ERROR)
            return
        if src.alpha.shape != self.doc.alpha.shape:
            self.message('Размер %dx%d в .iwi не совпадает с %dx%d в открытом файле.'
                         % (src.W, src.H, self.doc.W, self.doc.H), Gtk.MessageType.ERROR)
            return
        self.push_undo()
        self.doc.alpha[:] = src.alpha
        self.canvas.set_atlas(self.doc.alpha)
        self.preview.queue_draw()
        self.set_modified(True)
        self.status('Атлас заменён изображением из %s (%s)' % (os.path.basename(p), src.iwi.describe()))

    @staticmethod
    def image_to_alpha(im):
        """Если у картинки есть настоящая прозрачность — берём альфу, иначе яркость."""
        if im.mode in ('RGBA', 'LA') or 'transparency' in im.info:
            a = np.asarray(im.convert('RGBA'))[..., 3]
            if a.min() < 255:
                return a.copy()
        return np.asarray(im.convert('L')).copy()

    def cmd_export_glyph(self):
        r = self.sel_rect()
        if not r:
            self.message('Выберите глиф с непустой рамкой.')
            return
        g = self.sel_glyph()
        p = self.file_dialog('Экспорт глифа', save=True, suggest='glyph_%02X.png' % g['code'], patterns=['*.png'])
        if p:
            Image.fromarray(self.doc.alpha[r[1]:r[3], r[0]:r[2]]).save(p)
            self.status('Глиф сохранён: ' + p)

    def cmd_import_glyph(self):
        if not self.can_edit():
            return
        r = self.sel_rect()
        if not r:
            self.message('Выберите глиф и задайте ему рамку (инструмент «Рамка глифа»).'
                         if self.doc.kind == 'ff' else 'В режиме .iwi нет глифов: используйте импорт атласа.')
            return
        p = self.file_dialog('Импорт PNG в глиф', patterns=['*.png'])
        if not p:
            return
        arr = self.image_to_alpha(Image.open(p))
        rw, rh = r[2] - r[0], r[3] - r[1]
        if arr.shape != (rh, rw):
            arr = np.asarray(Image.fromarray(arr).resize((rw, rh), Image.LANCZOS))
        self.push_undo()
        self.doc.alpha[r[1]:r[3], r[0]:r[2]] = arr
        self.canvas.refresh_region(*r)
        self.set_modified(True)
        self.status('Картинка вписана в рамку %dx%d' % (rw, rh))

    # ---------------------------------------------------------- шрифты и глифы
    def on_font_changed(self, combo):
        if self._updating:
            return
        n = combo.get_active_text()
        if not n or not self.doc:
            return
        self.cur_font = n
        self.sel = None
        self.fill_list()
        self.refresh_panel()
        self.canvas.queue_draw()
        self.preview.queue_draw()

    def row_for(self, g):
        return ['0x%02X' % g['code'], glyph_char(g['code']), '%d×%d' % (g['w'], g['h']),
                '●' if gfields(g) != g['orig'] else '']

    def fill_list(self):
        sel = self.sel
        self._updating = True
        self.store.clear()
        for g in self.cur_gl():
            self.store.append(self.row_for(g))
        self._updating = False
        if sel is not None and sel < len(self.store):
            self.tree.set_cursor(Gtk.TreePath(sel))

    def update_row(self, idx):
        g = self.cur_gl()[idx]
        for i, v in enumerate(self.row_for(g)):
            self.store[idx][i] = v

    def on_tree_sel(self, selection):
        if self._updating:
            return
        model, it = selection.get_selected()
        if it:
            self.select(model.get_path(it)[0])

    def select(self, idx):
        if idx == self.sel:
            return
        self.sel = idx
        self.refresh_panel()
        path = Gtk.TreePath(idx)
        cur = self.tree.get_cursor()[0]
        if cur is None or cur.get_indices()[0] != idx:
            self._updating = True
            self.tree.set_cursor(path)
            self._updating = False
        self.tree.scroll_to_cell(path, None, True, 0.5, 0)
        self.canvas.queue_draw()
        self.preview.queue_draw()

    def pick_at(self, ax, ay):
        ix, iy = int(np.floor(ax)), int(np.floor(ay))
        best, area = None, None
        for i, g in enumerate(self.cur_gl()):
            l, t, r, b = g['rect']
            if l <= ix < r and t <= iy < b:
                a = (r - l) * (b - t)
                if best is None or a < area:
                    best, area = i, a
        if best is not None:
            self.select(best)

    def pick_value(self, ax, ay):
        ix, iy = int(np.floor(ax)), int(np.floor(ay))
        if 0 <= ix < self.doc.W and 0 <= iy < self.doc.H:
            self.val_scale.set_value(int(self.doc.alpha[iy, ix]))

    def refresh_panel(self):
        g = self.sel_glyph()
        self._updating = True
        for name, s in self.spins.items():
            s.set_sensitive(g is not None)
            if g is not None:
                if name in ('left', 'top', 'right', 'bottom'):
                    s.set_value(g['rect'][('left', 'top', 'right', 'bottom').index(name)])
                else:
                    s.set_value(g[name])
        self._updating = False
        if g is None:
            self.glyph_lbl.set_text('Режим .iwi: изображение без таблицы глифов'
                                    if self.doc and self.doc.kind == 'iwi' else 'Глиф не выбран')
        else:
            self.glyph_lbl.set_text('Глиф 0x%02X «%s»   (%s)' % (g['code'], glyph_char(g['code']), self.cur_font))

    def on_prop_changed(self, spin, name):
        g = self.sel_glyph()
        if self._updating or g is None:
            return
        self.push_undo(key=(self.cur_font, self.sel, name))
        v = spin.get_value_as_int()
        if name in ('left', 'top', 'right', 'bottom'):
            g['rect'][('left', 'top', 'right', 'bottom').index(name)] = v
            if self.keep_wh:
                g['w'] = clamp(g['rect'][2] - g['rect'][0], 0, 255)
                g['h'] = clamp(g['rect'][3] - g['rect'][1], 0, 255)
                self.refresh_panel()
        else:
            g[name] = v
        self.after_glyph_change()

    def after_glyph_change(self):
        self.update_row(self.sel)
        self.set_modified(True)
        self.canvas.queue_draw()
        self.preview.queue_draw()

    def apply_rect(self, rect):
        g = self.sel_glyph()
        if g is None:
            return
        self.push_undo()
        g['rect'] = list(rect)
        if self.keep_wh:
            g['w'] = clamp(rect[2] - rect[0], 0, 255)
            g['h'] = clamp(rect[3] - rect[1], 0, 255)
        self.refresh_panel()
        self.after_glyph_change()

    def cmd_autocrop(self):
        r = self.sel_rect()
        if not r:
            return
        sub = self.doc.alpha[r[1]:r[3], r[0]:r[2]]
        ys, xs = np.nonzero(sub)
        if len(xs) == 0:
            self.status('В рамке нет пикселей')
            return
        self.apply_rect([r[0] + int(xs.min()), r[1] + int(ys.min()), r[0] + int(xs.max()) + 1, r[1] + int(ys.max()) + 1])

    # ---------------------------------------------------------- инструменты
    def on_tool_toggled(self, btn, name):
        if btn.get_active():
            self.tool = name
            self.canvas.queue_draw()

    def set_tool(self, name):
        self.tool_btns[name].set_active(True)

    def on_show_rects(self, btn):
        self.show_rects = btn.get_active()
        self.canvas.queue_draw()

    # ---------------------------------------------------------- предпросмотр
    def on_preview_draw(self, w, cr):
        cr.set_source_rgb(0.08, 0.08, 0.1)
        cr.paint()
        if not self.doc or not self.cur_font:
            return False
        f = self.doc.fonts[self.cur_font]
        table = {}
        for g in f['gl']:
            table.setdefault(g['code'], g)
        sc = self.prev_scale.get_value()
        width = w.get_allocated_width()
        line_h = f['pixh'] * sc * 1.25
        x, base = 6.0, 6 + f['pixh'] * sc * 0.85
        pat = cairo.SurfacePattern(self.canvas.surf)
        pat.set_filter(cairo.FILTER_GOOD)
        for code in text_to_codes(self.prev_entry.get_text()):
            g = table.get(code)
            if g is None:
                continue
            adv = g['dx'] * sc
            if x + adv > width - 6 and x > 6:
                x, base = 6.0, base + line_h
            l, t, r, b = g['rect']
            if r > l and b > t and g['w'] > 0 and g['h'] > 0:
                cr.save()
                cr.translate(x + g['x0'] * sc, base + g['y0'] * sc)
                cr.scale(g['w'] * sc / (r - l), g['h'] * sc / (b - t))
                cr.translate(-l, -t)
                cr.rectangle(l, t, r - l, b - t)
                cr.clip()
                cr.set_source_rgb(0.95, 0.95, 0.95)
                cr.mask(pat)
                cr.restore()
            x += adv
        return False


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    win = Editor(path)
    win.connect('destroy', Gtk.main_quit)
    win.show_all()
    Gtk.main()


if __name__ == '__main__':
    main()
