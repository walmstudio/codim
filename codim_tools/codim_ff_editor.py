#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoD4 .ff (PS3 и PC) — редактор локализационных строк (GTK3).

Формат определяется автоматически. Общее: 12 байт заголовка ("IWffu100" + версия), дальше сжатая "зона".
  PS3 (версия 00 00 00 01): блоки [2 байта длины BE][raw deflate + нулевой добор], в конце "00 01".
  PC  (версия 05 00 00 00): один zlib-поток (78 DA ...) до конца файла.
Строки лежат в зоне как:  FF*8 | текст (cp1251) | 00 | КЛЮЧ | 00

Правила, которые соблюдает тулза (структура не ломается; для PC пункты про блоки не применимы —
там один поток, размер файла может чуть измениться, размер зоны и заголовок XFile остаются прежними):
  * длина слота строки фиксирована — текст не может быть длиннее исходного слота (в байтах cp1251),
    короче — добивается пробелами (как в оригинале);
  * меняются только изменённые строки, остальные байты зоны не трогаются;
  * каждый затронутый блок пересжимается так, чтобы влезть в ИСХОДНЫЙ размер сжатого блока
    (остаток добивается нулями), нетронутые блоки копируются байт в байт;
  * после сборки результат заново разбирается и проверяется: размер файла, число блоков,
    размеры всех сжатых блоков, размер зоны, хвост 00 01, побайтовое соответствие зоны.

Запуск:   python3 cod4_ff_editor.py [файл.ff]
Зависимости: python3-gi и gir1.2-gtk-3.0  (Debian/Ubuntu: sudo apt install python3-gi gir1.2-gtk-3.0)
Проверка без GUI:  python3 cod4_ff_editor.py --selftest файл.ff
"""
import json
import os
import re
import sys
import zlib

ENC = "cp1251"
MAGIC = b"IWffu100"
HEADER_LEN = 12
KEY_RE = re.compile(rb"\x00([A-Z][A-Z0-9_]{3,})\x00")


class FFError(Exception):
    pass


# ---------------------------------------------------------------- escaping
def esc(raw: bytes) -> str:
    """bytes -> строка для показа (управляющие байты экранируются)."""
    out = []
    for b in raw:
        if b == 0x5C:
            out.append("\\\\")
        elif b == 10:
            out.append("\\n")
        elif b == 13:
            out.append("\\r")
        elif b < 32 or b in (0x7F, 0x98):
            out.append("\\x%02x" % b)
        else:
            out.append(bytes([b]).decode(ENC))
    return "".join(out)


def unesc(text: str) -> bytes:
    """строка из редактора -> bytes (cp1251). Бросает FFError при невозможных символах."""
    out = bytearray()
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            d = text[i + 1]
            if d == "\\":
                out.append(0x5C); i += 2; continue
            if d == "n":
                out.append(10); i += 2; continue
            if d == "r":
                out.append(13); i += 2; continue
            if d == "x" and re.fullmatch(r"[0-9a-fA-F]{2}", text[i + 2:i + 4]):
                out.append(int(text[i + 2:i + 4], 16)); i += 4; continue
        if c == "\n":
            out.append(10); i += 1; continue
        try:
            out += c.encode(ENC)
        except UnicodeEncodeError:
            raise FFError("символ %r отсутствует в cp1251" % c)
        i += 1
    return bytes(out)


# ---------------------------------------------------------------- model
class Entry:
    __slots__ = ("idx", "key", "off", "ln", "raw", "orig_text", "text")

    def __init__(self, idx, key, off, ln, raw):
        self.idx, self.key, self.off, self.ln, self.raw = idx, key, off, ln, raw
        self.orig_text = esc(raw.rstrip(b" "))
        self.text = self.orig_text

    @property
    def used(self):
        return len(unesc(self.text))

    @property
    def changed(self):
        return self.text != self.orig_text

    def new_raw(self):
        nb = unesc(self.text)
        if len(nb) > self.ln:
            raise FFError("%s: %d байт при лимите %d" % (self.key, len(nb), self.ln))
        return nb.ljust(self.ln, b" ")


class FFFile:
    def __init__(self):
        self.path = None
        self.data = b""
        self.blocks = []       # (pos, L, out_len, region_bytes)
        self.zone = b""
        self.starts = []       # начало каждого блока в зоне
        self.tail = b""
        self.kind = "ps3"
        self.entries = []

    # -- parsing
    @staticmethod
    def parse(data):
        """-> (kind, blocks, zone, starts, tail); kind: 'ps3' | 'pc'"""
        if data[:8] != MAGIC:
            raise FFError("Это не IWffu100 (.ff)")
        if len(data) > HEADER_LEN + 2 and data[HEADER_LEN] == 0x78 and ((data[HEADER_LEN] << 8) | data[HEADER_LEN + 1]) % 31 == 0:
            do = zlib.decompressobj(15)
            try:
                zone = do.decompress(data[HEADER_LEN:])
            except zlib.error as e:
                raise FFError("Поток zlib не распаковывается: %s" % e)
            if not do.eof:
                raise FFError("Поток zlib не завершён")
            L = len(data) - HEADER_LEN - len(do.unused_data)
            return "pc", [(HEADER_LEN, L, len(zone), data[HEADER_LEN:HEADER_LEN + L])], zone, [0], do.unused_data
        pos, blocks, zone, starts = HEADER_LEN, [], bytearray(), []
        while pos + 2 <= len(data):
            L = int.from_bytes(data[pos:pos + 2], "big")
            if L <= 1:
                break
            chunk = data[pos + 2:pos + 2 + L]
            if len(chunk) != L:
                raise FFError("Обрыв блока на смещении %d" % pos)
            do = zlib.decompressobj(-15)
            try:
                out = do.decompress(chunk)
            except zlib.error as e:
                raise FFError("Блок #%d не распаковывается: %s" % (len(blocks), e))
            if not do.eof:
                raise FFError("Блок #%d: поток deflate не завершён" % len(blocks))
            starts.append(len(zone))
            zone += out
            blocks.append((pos, L, len(out), chunk))
            pos += 2 + L
        return "ps3", blocks, bytes(zone), starts, data[pos:]

    def load(self, path):
        with open(path, "rb") as f:
            data = f.read()
        kind, blocks, zone, starts, tail = self.parse(data)
        self.path, self.data, self.blocks, self.zone, self.starts, self.tail = path, data, blocks, zone, starts, tail
        self.kind = kind
        self.entries = []
        prev_term = -1
        for m in KEY_RE.finditer(zone):
            end = m.start()
            q = zone.rfind(b"\x00", 0, end)
            term = m.end() - 1
            if q < 0:
                prev_term = term
                continue
            region = zone[q + 1:end]
            if len(region) > 8 and region[:8] == b"\xff" * 8:
                start = 8
                if q != prev_term:
                    # первая строка таблицы: перед маркером может стоять 0xFF от предыдущей структуры —
                    # лишние ведущие FF не считаем частью текста (не трогаем их при записи)
                    while region[start:start + 1] == b"\xff":
                        start += 1
                value = region[start:]
                if value:
                    self.entries.append(Entry(len(self.entries), m.group(1).decode(), q + 1 + start, len(value), value))
            prev_term = term
        return self

    def block_of(self, off):
        lo, hi = 0, len(self.starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def set_text(self, e, text):
        old = e.text
        e.text = text
        try:
            e.new_raw()
        except FFError:
            e.text = old
            raise

    # -- building
    @staticmethod
    def _fit(block_bytes, limit):
        """Подбирает параметры deflate, чтобы сжатый блок <= limit. Возвращает (bytes, min_len)."""
        best_len = None
        for lvl in (9, 8, 7, 6, 5, 4, 3, 2, 1):
            for mem in (9, 8, 7, 6, 5, 4, 3, 2, 1):
                for strat in (0, 1, 2, 3, 4):
                    c = zlib.compressobj(lvl, zlib.DEFLATED, -15, mem, strat)
                    z = c.compress(block_bytes) + c.flush()
                    if best_len is None or len(z) < best_len:
                        best_len = len(z)
                    if len(z) <= limit:
                        return z, best_len
        return None, best_len

    def build(self):
        zone = bytearray(self.zone)
        touched = {}
        for e in self.entries:
            nr = e.new_raw()
            if nr != e.raw:
                zone[e.off:e.off + e.ln] = nr
                for b in {self.block_of(e.off), self.block_of(e.off + e.ln - 1)}:
                    touched.setdefault(b, []).append(e.key)
        if self.kind == "pc":
            if not touched:
                out = self.data
            else:
                out = self.data[:HEADER_LEN] + zlib.compress(bytes(zone), 9) + self.tail
            return out, self.verify(out, bytes(zone), sorted(touched))
        out = bytearray(self.data[:HEADER_LEN])
        problems = []
        for i, (pos, L, olen, chunk) in enumerate(self.blocks):
            if i not in touched:
                out += self.data[pos:pos + 2 + L]
                continue
            s = self.starts[i]
            blk = bytes(zone[s:s + olen])
            z, best = self._fit(blk, L)
            if z is None:
                problems.append("Блок #%d не влезает в %d байт (лучшее сжатие: %d, лишних %d). Строки: %s"
                                % (i, L, best, best - L, ", ".join(touched[i][:12])))
                continue
            out += self.data[pos:pos + 2] + z.ljust(L, b"\x00")
        if problems:
            raise FFError("\n".join(problems) + "\n\nУкоротите/упростите эти строки (чаще повторяющиеся символы "
                          "и пробелы сжимаются лучше) и повторите.")
        out += self.tail
        out = bytes(out)
        report = self.verify(out, bytes(zone), sorted(touched))
        return out, report

    def verify(self, out, zone, touched):
        kind, blocks, z2, starts, tail = self.parse(out)
        if kind != self.kind:
            raise FFError("Проверка структуры провалена: изменился тип контейнера")
        if self.kind == "pc":
            checks = [
                ("заголовок (IWffu100 + версия)", out[:HEADER_LEN] == self.data[:HEADER_LEN],
                 self.data[:HEADER_LEN].hex(" ")),
                ("размер зоны", len(z2) == len(self.zone), "%d" % len(z2)),
                ("заголовок XFile (44 байта зоны)", z2[:44] == self.zone[:44], "не тронут"),
                ("зона = ожидаемая", z2 == zone, "изменено слотов: байтовое соответствие"),
                ("поток zlib одним куском", tail == self.tail, "хвост %d байт" % len(tail)),
                ("размер файла (для PC допустимо отличие)", True, "%d → %d" % (len(self.data), len(out))),
            ]
            bad = [c for c in checks if not c[1]]
            if bad:
                raise FFError("Проверка структуры провалена: " + ", ".join(c[0] for c in bad))
            return checks
        checks = [
            ("размер файла", len(out) == len(self.data), "%d → %d" % (len(self.data), len(out))),
            ("число блоков", len(blocks) == len(self.blocks), "%d" % len(blocks)),
            ("размеры всех сжатых блоков", [b[1] for b in blocks] == [b[1] for b in self.blocks], "без изменений"),
            ("размер зоны", len(z2) == len(self.zone), "%d" % len(z2)),
            ("хвост 00 01", tail == self.tail, tail.hex(" ")),
            ("зона = ожидаемая", z2 == zone, "%d блок(ов) пересжато: %s" % (len(touched), touched)),
            ("заголовок", out[:HEADER_LEN] == self.data[:HEADER_LEN], "без изменений"),
        ]
        bad = [c for c in checks if not c[1]]
        if bad:
            raise FFError("Проверка структуры провалена: " + ", ".join(c[0] for c in bad))
        return checks


# ---------------------------------------------------------------- selftest
def selftest(path):
    ff = FFFile().load(path)
    print("Формат:", ff.kind.upper(), "| строк:", len(ff.entries), "| блоков:", len(ff.blocks), "| зона:", len(ff.zone))
    out, rep = ff.build()
    assert out == ff.data, "пустая правка должна давать идентичный файл"
    print("Без правок: файл идентичен исходному")
    cyr = [e for e in ff.entries if re.search("[а-яА-Я]", e.text)]
    for e in cyr[:3] + cyr[-3:]:
        ff.set_text(e, (e.text.rstrip() + "X")[: e.ln])
        ff.set_text(e, ("Тест" + e.text)[: e.ln].rstrip() or "Т")
    out, rep = ff.build()
    for n, ok, info in rep:
        print(("OK  " if ok else "FAIL"), n, "—", info)


# ---------------------------------------------------------------- GUI
def run_gui(initial=None):
    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk, Pango

    C_IDX, C_KEY, C_INFO, C_TEXT, C_CHG, C_WEIGHT, C_COLOR = range(7)

    class App(Gtk.Window):
        def __init__(self):
            super().__init__(title="CoD4 PS3 .ff — редактор строк")
            self.set_default_size(1100, 720)
            self.ff = None
            self.cur = None
            self.connect("destroy", Gtk.main_quit)

            hb = Gtk.HeaderBar(show_close_button=True, title="CoD4 PS3 .ff — редактор строк")
            self.set_titlebar(hb)
            for label, cb in (("Открыть…", self.on_open), ("Сохранить как…", self.on_save),
                              ("Экспорт JSON", self.on_export), ("Импорт JSON", self.on_import)):
                b = Gtk.Button(label=label)
                b.connect("clicked", cb)
                hb.pack_start(b)

            root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=6)
            self.add(root)

            bar = Gtk.Box(spacing=8)
            self.search = Gtk.SearchEntry(placeholder_text="Поиск по ключу или тексту")
            self.search.connect("search-changed", lambda *_: self.filter.refilter())
            self.only_chg = Gtk.CheckButton(label="Только изменённые")
            self.only_cyr = Gtk.CheckButton(label="Только с кириллицей")
            for w in (self.only_chg, self.only_cyr):
                w.connect("toggled", lambda *_: self.filter.refilter())
            bar.pack_start(self.search, True, True, 0)
            bar.pack_start(self.only_chg, False, False, 0)
            bar.pack_start(self.only_cyr, False, False, 0)
            root.pack_start(bar, False, False, 0)

            self.store = Gtk.ListStore(int, str, str, str, bool, int, str)
            self.filter = self.store.filter_new()
            self.filter.set_visible_func(self.visible)
            self.tree = Gtk.TreeView(model=self.filter)
            self.tree.set_rules_hint(True)
            r = Gtk.CellRendererText()
            col = Gtk.TreeViewColumn("Ключ", r, text=C_KEY)
            col.set_resizable(True); col.set_min_width(240)
            self.tree.append_column(col)
            r = Gtk.CellRendererText(xalign=1.0)
            col = Gtk.TreeViewColumn("Байт/макс", r, text=C_INFO, foreground=C_COLOR)
            col.set_min_width(90)
            self.tree.append_column(col)
            self.rtext = Gtk.CellRendererText(editable=True, ellipsize=Pango.EllipsizeMode.END)
            self.rtext.connect("edited", self.on_cell_edited)
            col = Gtk.TreeViewColumn("Текст", self.rtext, text=C_TEXT, weight=C_WEIGHT)
            col.set_expand(True)
            self.tree.append_column(col)
            self.tree.get_selection().connect("changed", self.on_select)
            sw = Gtk.ScrolledWindow()
            sw.add(self.tree)

            detail = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            self.lbl_key = Gtk.Label(xalign=0)
            self.lbl_cnt = Gtk.Label(xalign=0)
            self.lbl_orig = Gtk.Label(xalign=0, selectable=True, wrap=True)
            self.view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, left_margin=4, right_margin=4)
            self.buf = self.view.get_buffer()
            self.buf.connect("changed", self.on_buf_changed)
            sw2 = Gtk.ScrolledWindow(min_content_height=90)
            sw2.add(self.view)
            btns = Gtk.Box(spacing=6)
            b_apply = Gtk.Button(label="Применить (Ctrl+Enter)")
            b_apply.connect("clicked", self.on_apply)
            b_rev = Gtk.Button(label="Вернуть оригинал")
            b_rev.connect("clicked", self.on_revert)
            btns.pack_start(b_apply, False, False, 0)
            btns.pack_start(b_rev, False, False, 0)
            btns.pack_start(self.lbl_cnt, False, False, 12)
            hint = Gtk.Label(xalign=0, label="\\n — перенос строки, \\xNN — управляющий байт (например \\x01 — кнопка PS3). "
                                             "Кодировка cp1251; лишнее место добивается пробелами.")
            hint.get_style_context().add_class("dim-label")
            for w in (self.lbl_key, self.lbl_orig, sw2, btns, hint):
                detail.pack_start(w, w is sw2, w is sw2, 0)
            self.view.connect("key-press-event", self.on_view_key)

            paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
            paned.pack1(sw, True, False)
            paned.pack2(detail, False, False)
            paned.set_position(440)
            root.pack_start(paned, True, True, 0)

            self.status = Gtk.Label(xalign=0, label="Откройте .ff файл")
            root.pack_start(self.status, False, False, 0)
            self.set_sensitive_detail(False)
            self.show_all()
            if initial:
                self.open_path(initial)

        # ---- helpers
        def set_sensitive_detail(self, on):
            self.view.set_sensitive(on)

        def msg(self, title, text, kind=Gtk.MessageType.INFO):
            d = Gtk.MessageDialog(transient_for=self, modal=True, message_type=kind,
                                  buttons=Gtk.ButtonsType.OK, text=title)
            d.format_secondary_text(text)
            d.run(); d.destroy()

        def visible(self, model, it, _):
            if self.ff is None:
                return True
            row = model[it]
            if self.only_chg.get_active() and not row[C_CHG]:
                return False
            if self.only_cyr.get_active() and not re.search("[а-яА-ЯёЁ]", row[C_TEXT]):
                return False
            q = self.search.get_text().lower()
            return not q or q in row[C_KEY].lower() or q in row[C_TEXT].lower()

        def fill_row(self, e):
            row = self.store[e.idx]
            row[C_TEXT] = e.text
            row[C_CHG] = e.changed
            row[C_WEIGHT] = 700 if e.changed else 400
            used = e.used
            row[C_INFO] = "%d/%d" % (used, e.ln)
            row[C_COLOR] = "#c66900" if used == e.ln else None
            self.update_status()

        def update_status(self):
            n = sum(1 for e in self.ff.entries if e.changed)
            fmt = "PC" if self.ff.kind == "pc" else "PS3, блоков: %d" % len(self.ff.blocks)
            self.status.set_text("%s [%s] — строк: %d, изменено: %d" % (os.path.basename(self.ff.path), fmt,
                                                                      len(self.ff.entries), n))

        # ---- file ops
        def open_path(self, path):
            try:
                self.ff = FFFile().load(path)
            except (FFError, OSError) as ex:
                self.msg("Не удалось открыть", str(ex), Gtk.MessageType.ERROR)
                return
            self.store.clear()
            for e in self.ff.entries:
                self.store.append([e.idx, e.key, "%d/%d" % (e.used, e.ln), e.text, False, 400, None])
                self.fill_row(e)
            self.cur = None
            self.filter.refilter()

        def on_open(self, *_):
            d = Gtk.FileChooserDialog(title="Открыть .ff", transient_for=self, action=Gtk.FileChooserAction.OPEN)
            d.add_buttons("Отмена", Gtk.ResponseType.CANCEL, "Открыть", Gtk.ResponseType.OK)
            if d.run() == Gtk.ResponseType.OK:
                p = d.get_filename(); d.destroy(); self.open_path(p)
            else:
                d.destroy()

        def on_save(self, *_):
            if not self.ff:
                return
            try:
                out, rep = self.ff.build()
            except FFError as ex:
                self.msg("Сохранение невозможно", str(ex), Gtk.MessageType.ERROR)
                return
            d = Gtk.FileChooserDialog(title="Сохранить как", transient_for=self, action=Gtk.FileChooserAction.SAVE)
            d.add_buttons("Отмена", Gtk.ResponseType.CANCEL, "Сохранить", Gtk.ResponseType.OK)
            d.set_do_overwrite_confirmation(True)
            d.set_current_name(os.path.basename(self.ff.path))
            if d.run() == Gtk.ResponseType.OK:
                p = d.get_filename(); d.destroy()
                with open(p, "wb") as f:
                    f.write(out)
                self.msg("Сохранено: " + os.path.basename(p),
                         "\n".join("✔ %s — %s" % (n, info) for n, ok, info in rep))
            else:
                d.destroy()

        def on_export(self, *_):
            if not self.ff:
                return
            d = Gtk.FileChooserDialog(title="Экспорт JSON", transient_for=self, action=Gtk.FileChooserAction.SAVE)
            d.add_buttons("Отмена", Gtk.ResponseType.CANCEL, "Сохранить", Gtk.ResponseType.OK)
            d.set_do_overwrite_confirmation(True)
            d.set_current_name("strings.json")
            if d.run() == Gtk.ResponseType.OK:
                data = {e.key: {"text": e.text, "max_bytes": e.ln} for e in self.ff.entries}
                with open(d.get_filename(), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=1)
            d.destroy()

        def on_import(self, *_):
            if not self.ff:
                return
            d = Gtk.FileChooserDialog(title="Импорт JSON", transient_for=self, action=Gtk.FileChooserAction.OPEN)
            d.add_buttons("Отмена", Gtk.ResponseType.CANCEL, "Открыть", Gtk.ResponseType.OK)
            if d.run() != Gtk.ResponseType.OK:
                d.destroy(); return
            p = d.get_filename(); d.destroy()
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as ex:
                self.msg("Ошибка JSON", str(ex), Gtk.MessageType.ERROR); return
            by_key = {}
            for e in self.ff.entries:
                by_key.setdefault(e.key, []).append(e)
            ok, bad = 0, []
            for k, v in data.items():
                text = v["text"] if isinstance(v, dict) else v
                for e in by_key.get(k, []):
                    if text == e.text:
                        continue
                    try:
                        self.ff.set_text(e, text); self.fill_row(e); ok += 1
                    except FFError as ex:
                        bad.append(str(ex))
            self.filter.refilter()
            self.msg("Импорт завершён", "Применено: %d\nПропущено: %d\n%s" % (ok, len(bad), "\n".join(bad[:15])))

        # ---- editing
        def entry_at(self, path):
            it = self.filter.get_iter(path)
            cit = self.filter.convert_iter_to_child_iter(it)
            return self.ff.entries[self.store[cit][C_IDX]]

        def on_cell_edited(self, _r, path, new):
            e = self.entry_at(path)
            try:
                self.ff.set_text(e, new)
            except FFError as ex:
                self.status.set_text("Отклонено: %s" % ex)
                return
            self.fill_row(e)
            if self.cur is e:
                self.load_detail(e)

        def on_select(self, sel):
            model, it = sel.get_selected()
            if not it:
                return
            self.cur = self.entry_at(model.get_path(it))
            self.load_detail(self.cur)

        def load_detail(self, e):
            self.set_sensitive_detail(True)
            self.lbl_key.set_markup("<b>%s</b>   (слот: %d байт)" % (e.key, e.ln))
            self.lbl_orig.set_text("Оригинал: " + e.orig_text)
            self.buf.handler_block_by_func(self.on_buf_changed)
            self.buf.set_text(e.text)
            self.buf.handler_unblock_by_func(self.on_buf_changed)
            self.on_buf_changed()

        def buf_text(self):
            return self.buf.get_text(self.buf.get_start_iter(), self.buf.get_end_iter(), True)

        def on_buf_changed(self, *_):
            if not self.cur:
                return
            try:
                n = len(unesc(self.buf_text()))
            except FFError as ex:
                self.lbl_cnt.set_markup('<span foreground="#c00000">%s</span>' % GLib_escape(str(ex)))
                return
            over = n > self.cur.ln
            color = "#c00000" if over else ("#c66900" if n == self.cur.ln else "#2a7a2a")
            self.lbl_cnt.set_markup('<span foreground="%s"><b>%d / %d байт</b>%s</span>'
                                    % (color, n, self.cur.ln, "  — слишком длинно" if over else ""))

        def on_apply(self, *_):
            if not self.cur:
                return
            try:
                self.ff.set_text(self.cur, self.buf_text())
            except FFError as ex:
                self.status.set_text("Отклонено: %s" % ex)
                return
            self.fill_row(self.cur)
            self.filter.refilter()

        def on_revert(self, *_):
            if self.cur:
                self.ff.set_text(self.cur, self.cur.orig_text)
                self.fill_row(self.cur)
                self.load_detail(self.cur)

        def on_view_key(self, _w, ev):
            from gi.repository import Gdk
            if ev.keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) and ev.state & Gdk.ModifierType.CONTROL_MASK:
                self.on_apply()
                return True
            return False

    def GLib_escape(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    App()
    Gtk.main()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        selftest(args[1])
    else:
        try:
            run_gui(args[0] if args else None)
        except ImportError:
            sys.exit("Нужен PyGObject/GTK3:  sudo apt install python3-gi gir1.2-gtk-3.0")
