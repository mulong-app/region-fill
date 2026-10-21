#!/usr/bin/env python3
"""
region_fill.py のGUI（ブラウザで操作）。 v1.1.0

  python region_fill_gui.py            # ブラウザが自動で開きます
  python region_fill_gui.py --port 9000 --no-browser
  python region_fill_gui.py --settings my_settings.json   # 設定の保存先を変える

Python 標準ライブラリだけで小さなローカルサーバーを立て、ブラウザで操作します。
プレビューは出力される SVG そのものなので、表示と保存結果が完全に一致します。
同じフォルダに region_fill.py を置いてください（必要なのは numpy だけ）。
"""
import argparse
import json
import os
import random
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import region_fill as rf

# 直近の生成結果（「色だけ変える」で使う）
_last = {}
_lock = threading.Lock()

# 実行中の生成（中断用）。新しい生成が始まると、前の生成は自動で中断する。
# 生成は常に1つだけ走らせる（前の生成が止まるのを待ってから始める）ので、
# 重い処理が裏に溜まって全体が遅くなることがない
_job = {"stop": None}
_job_lock = threading.Lock()
_run_lock = threading.Lock()
GUI_TIME_LIMIT = 30.0          # GUI では 30 秒で打ち切って理由を表示する


def _start_job():
    with _job_lock:
        if _job["stop"] is not None:
            _job["stop"].set()
        ev = threading.Event()
        _job["stop"] = ev
        return ev


def api_cancel(_p):
    with _job_lock:
        if _job["stop"] is not None:
            _job["stop"].set()
    return {"ok": True}


def _render(res, colors, randomness, color_seed):
    """保存してある線と区画に、色を塗り直して SVG にする。"""
    rng = random.Random(color_seed)
    cidx = rf.color_graph(res["adj"], len(colors), rng, randomness)
    fill_cols = [colors[c] for c in cidx]
    p = res["params"]
    uniform = _uniform_stroke(p, res["info"])
    return rf.svg_string(int(p["width"]), int(p["height"]), res["faces"], fill_cols, res["curves"],
                         res["widths"], p["line_color"], p["bg"], uniform)


def _uniform_stroke(p, info):
    """線の強弱なし、または線なしのときは均一な線幅（None なら強弱つき）。"""
    s = info["stroke_px"]
    return s if (float(p["pressure"]) <= 0 or s <= 0) else None


def api_generate(p):
    stop = _start_job()
    if not _run_lock.acquire(timeout=10):        # 前の生成が止まるのを待つ
        raise RuntimeError("前の生成が止まりませんでした。少し待ってからもう一度お試しください。")
    try:
        if stop.is_set():                        # 待っている間に、さらに新しい生成が来た
            raise rf.Cancelled()
        return _generate(p, stop)
    finally:
        _run_lock.release()


def _generate(p, stop):
    t0 = time.time()
    colors = [c for c in p["colors"] if c]
    curves, faces, fill_cols, adj, widths, info = rf.generate(
        int(p["width"]), int(p["height"]), int(p["curves"]), colors, int(p["seed"]),
        float(p["randomness"]), float(p["wiggle"]), f'{float(p["stroke"])}%',
        min_radius=f'{float(p["min_radius"])}%', length=float(p["length"]), max_bends=int(p["max_bends"]),
        attractors=f'{int(p["att_min"])}-{int(p["att_max"])}', pull=float(p["pull"]),
        avoid=float(p["avoid"]), smooth=float(p["smooth"]), pressure=float(p["pressure"]),
        should_stop=stop.is_set, time_limit=GUI_TIME_LIMIT)
    if STORE is not None:
        STORE.save_last({k: v for k, v in p.items() if k != "quiet"})   # 次回起動時に復元
    res = {"curves": curves, "faces": faces, "adj": adj, "widths": widths, "info": info, "params": p}
    with _lock:
        _last.clear()
        _last.update(res)
    svg = rf.svg_string(int(p["width"]), int(p["height"]), faces, fill_cols, curves, widths,
                        p["line_color"], p["bg"], _uniform_stroke(p, info))
    if STORE is not None and not p.get("quiet"):
        STORE.save_view(svg, info["attractors"])
    return {
        "svg": svg,
        "attractors": info["attractors"],
        "stats": {
            "faces": len(faces),
            "crossings": info["crossings"],
            "stroke_px": round(info["stroke_px"], 1),
            "min_radius_px": round(info["min_radius_px"], 1),
            "raised": info["min_radius_px"] > info["min_radius_requested_px"] + 1e-9,
            "rollbacks": sum(w.get("rollbacks", 0) for w in info["walk"]),
            "seconds": round(time.time() - t0, 2),
        },
    }


def api_harmony(p):
    rng = random.Random(p.get("seed"))
    cols, mode = rf.harmony_palette(rng, p.get("mode", "おまかせ"), int(p.get("n", 5)))
    return {"colors": cols, "mode": mode}


def api_palette_from_image(p):
    """ブラウザで縮小した画像の画素（RGBA）から配色を作る。"""
    import numpy as np
    w, h = int(p["w"]), int(p["h"])
    data = np.asarray(p["data"], dtype=np.uint8)
    if w <= 0 or h <= 0 or data.size != w * h * 4 or w * h > 400 * 400:
        raise ValueError("画像を読み込めませんでした。")
    n = max(4, min(10, int(p.get("n", 5))))
    cols = rf.palette_from_array(data.reshape(h, w, 4), n, vivid=bool(p.get("vivid", True)))
    return {"colors": cols}


def api_recolor(p):
    with _lock:
        if not _last:
            raise ValueError("まだ準備中です（前回の絵を再計算しています）。少し待ってからもう一度お試しください。")
        res = dict(_last)
        res["params"] = dict(res["params"])
    for k in ("line_color", "bg"):              # 配色プリセットで線・背景が変わる場合に反映
        if p.get(k):
            res["params"][k] = p[k]
    colors = [c for c in p["colors"] if c]
    if len(colors) < 4:
        raise ValueError("色は4色以上指定してください。")
    svg = _render(res, colors, float(p["randomness"]), int(p["color_seed"]))
    if STORE is not None:
        STORE.save_view(svg, res["info"]["attractors"])
    return {"svg": svg}


class SettingsStore:
    """設定をJSONファイルに保存する（最後に使った設定＋名前付きの設定）。"""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def load(self):
        with self.lock:
            return self._read()

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, ValueError):
            data = {}
        data.setdefault("last", None)
        data.setdefault("presets", {})
        return data

    def _write(self, data):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)              # 書きかけのファイルが残らないように置き換える

    @property
    def view_path(self):
        return os.path.splitext(self.path)[0] + "_last.svg"

    def save_view(self, svg, attractors):
        """最後に表示した絵を保存（次回起動時にすぐ表示するため）。"""
        with self.lock:
            tmp = self.view_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fp:
                fp.write(svg)
            os.replace(tmp, self.view_path)
            data = self._read()
            data["last_view"] = {"attractors": attractors}
            self._write(data)

    def load_view(self):
        with self.lock:
            try:
                with open(self.view_path, encoding="utf-8") as fp:
                    svg = fp.read()
            except OSError:
                return None
            meta = self._read().get("last_view") or {}
            return {"svg": svg, "attractors": meta.get("attractors", [])}

    def save_last(self, settings):
        with self.lock:
            data = self._read()
            data["last"] = settings
            self._write(data)

    def save_preset(self, name, settings):
        name = str(name).strip()
        if not name:
            raise ValueError("設定の名前を入力してください。")
        with self.lock:
            data = self._read()
            data["presets"][name] = settings
            self._write(data)
            return {"presets": data["presets"]}

    def delete_preset(self, name):
        with self.lock:
            data = self._read()
            data["presets"].pop(str(name), None)
            self._write(data)
            return {"presets": data["presets"]}


STORE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):            # アクセスログは出さない
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/palettes":
            styles = [{"name": st["name"], "desc": st["desc"], "settings": rf.style_settings(st["name"])}
                      for st in rf.STYLE_PRESETS]
            self._send(200, json.dumps({"palettes": rf.PALETTES, "modes": rf.HARMONY_MODES, "styles": styles,
                                        "version": rf.__version__}), "application/json")
        elif self.path == "/api/settings":
            data = STORE.load()
            data["view"] = STORE.load_view() if data.get("last") else None
            self._send(200, json.dumps(data), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            p = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/generate":
                out = api_generate(p)
            elif self.path == "/api/recolor":
                out = api_recolor(p)
            elif self.path == "/api/harmony":
                out = api_harmony(p)
            elif self.path == "/api/palette_from_image":
                out = api_palette_from_image(p)
            elif self.path == "/api/cancel":
                out = api_cancel(p)
            elif self.path == "/api/presets/save":
                out = STORE.save_preset(p.get("name", ""), p.get("settings", {}))
            elif self.path == "/api/presets/delete":
                out = STORE.delete_preset(p.get("name", ""))
            else:
                return self._send(404, "not found", "text/plain")
            self._send(200, json.dumps(out), "application/json")
        except rf.Cancelled:
            self._send(200, json.dumps({"cancelled": True}), "application/json")
        except (ValueError, RuntimeError) as e:     # 設定が厳しすぎる等：画面に理由を表示する
            self._send(400, json.dumps({"error": str(e)}), "application/json")
        except Exception as e:                      # 想定外のエラーは詳細も端末に出す
            traceback.print_exc()
            msg = f"{type(e).__name__}: {e}"
            self._send(400, json.dumps({"error": msg}), "application/json")


PAGE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Region Fill</title>
<style>
:root{
  --bg:#f4f3ef; --panel:#ffffff; --ink:#1d1d1b; --muted:#6d6a63; --line:#e3e0d8;
  --accent:#1d1d1b; --accent-ink:#ffffff; --track:#dcd8ce; --err:#b42318;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:13px/1.45 system-ui,-apple-system,"Hiragino Sans","Yu Gothic UI","Noto Sans JP",sans-serif;}
.app{display:grid;grid-template-columns:320px 1fr;height:100vh}
aside{background:var(--panel);border-right:1px solid var(--line);overflow-y:auto;padding:16px 18px 28px}
h1{font-size:15px;margin:0 0 2px;letter-spacing:.02em}
.sub{color:var(--muted);font-size:12px;margin-bottom:14px}
details{border-top:1px solid var(--line);padding:10px 0 6px}
summary{cursor:pointer;font-weight:600;list-style:none;display:flex;justify-content:space-between;align-items:center}
summary::-webkit-details-marker{display:none}
summary::after{content:"＋";color:var(--muted);font-weight:400}
details[open] summary::after{content:"－"}
.row{margin:10px 0}
.row label{display:flex;justify-content:space-between;gap:8px;margin-bottom:3px}
.row label .v{color:var(--muted);font-variant-numeric:tabular-nums}
.hint{color:var(--muted);font-size:11px;margin-top:2px}
.ver{font-weight:400;color:var(--muted);font-size:12px}
.load{display:inline-block;font-size:10px;line-height:1;padding:3px 5px;border-radius:4px;margin-left:6px;vertical-align:1px;font-weight:600;letter-spacing:.02em;white-space:nowrap}
.load.ln{background:#F0EFEA;color:#8A867D;font-weight:500}
.lt{display:inline-flex;align-items:center;flex-wrap:wrap;gap:2px}
.load.l0{background:#E8F3EA;color:#2E6B3A}.load.l1{background:#FFF3D6;color:#7A5800}.load.l2{background:#FDE2DE;color:#9B2C1F}
.estimate{font-variant-numeric:tabular-nums}
.imgrow{display:grid;grid-template-columns:auto 1fr auto;gap:6px;align-items:center}
.imgrow img{width:44px;height:44px;object-fit:cover;border-radius:6px;border:1px solid var(--line);display:none}
.drop{outline:2px dashed var(--accent);outline-offset:4px;border-radius:8px}
input[type=range]{width:100%;accent-color:var(--accent)}
input[type=number],select{width:100%;padding:5px 7px;border:1px solid var(--line);border-radius:6px;background:#fff;font:inherit}
.two{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.seed{display:grid;grid-template-columns:1fr auto;gap:6px}
button{font:inherit;border:1px solid var(--line);background:#fff;color:var(--ink);border-radius:7px;padding:6px 10px;cursor:pointer}
button:hover{border-color:#bdb8ad}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:600;padding:9px 12px;width:100%}
button:disabled{opacity:.5;cursor:default}
.check{display:flex;align-items:center;gap:7px;margin:8px 0}
.swatches{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.sw{position:relative;width:34px;height:34px}
.sw input{width:34px;height:34px;padding:0;border:1px solid var(--line);border-radius:7px;background:none;cursor:pointer}
.sw .x{position:absolute;top:-6px;right:-6px;width:16px;height:16px;border-radius:50%;padding:0;font-size:10px;line-height:14px}
.cats{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px}
.cat{padding:3px 9px;border-radius:999px;font-size:12px}
.cat.on{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.pals{position:relative;display:grid;grid-template-columns:1fr 1fr;gap:6px;max-height:300px;overflow-y:auto;padding:2px}
.pal{display:flex;flex-direction:column;gap:4px;padding:5px;text-align:left;border-radius:8px}
.pal .strip{display:flex;height:16px;border-radius:4px;overflow:hidden;box-shadow:inset 0 0 0 1px rgba(0,0,0,.08)}
.pal .strip span{flex:1}
.pal .nm{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pal.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.pal.sel .nm{color:var(--ink);font-weight:600}
.harmony{display:grid;grid-template-columns:1fr auto;gap:6px}
.colors2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.colors2 input[type=color]{width:100%;height:28px;border:1px solid var(--line);border-radius:6px;background:none;padding:0}
main{display:grid;grid-template-rows:auto 1fr auto;min-width:0}
.bar{display:flex;gap:8px;align-items:center;padding:12px 18px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
.bar .grow{flex:1}
.status{color:var(--muted);font-variant-numeric:tabular-nums}
.status.err{color:var(--err)}
.stage{position:relative;display:flex;align-items:center;justify-content:center;padding:24px;min-height:0;overflow:auto}
.frame{position:relative;box-shadow:0 1px 2px rgba(0,0,0,.06),0 8px 30px rgba(0,0,0,.08);background:#fff;line-height:0}
.frame svg{display:block;width:100%;height:auto}
.overlay{position:absolute;inset:0;pointer-events:none}
.busy{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:rgba(244,243,239,.45);font-weight:600;line-height:1.4}
.busy.on{display:flex}
.busy .card{display:flex;flex-direction:column;gap:10px;align-items:center;background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px 26px;box-shadow:0 6px 24px rgba(0,0,0,.12)}
.busy button{font-weight:600;padding:7px 22px;background:#fff}
.busy .sec{font-weight:400;color:var(--muted);font-variant-numeric:tabular-nums}
.presetrow{display:grid;grid-template-columns:1fr auto auto;gap:6px}
.saverow{display:grid;grid-template-columns:1fr auto;gap:6px}
input[type=text]{width:100%;padding:5px 7px;border:1px solid var(--line);border-radius:6px;background:#fff;font:inherit}
.btnrow{display:flex;gap:6px;flex-wrap:wrap}
.btnrow button{flex:1}
.empty{color:var(--muted)}
.foot{padding:8px 18px;color:var(--muted);font-size:11px;border-top:1px solid var(--line);background:var(--panel)}
@media (max-width:760px){.app{grid-template-columns:1fr;height:auto}aside{border-right:0;border-bottom:1px solid var(--line)}}
</style>
</head>
<body>
<div class="app">
<aside>
  <h1>Region Fill <span class="ver" id="ver"></span></h1>
  <div class="sub">一筆書きの曲線で区切り、隣り合う区画を別の色で塗り分けます</div>

  <details id="settings_box" open><summary>設定の保存</summary>
    <div class="row presetrow">
      <select id="preset_list"><option value="">保存した設定…</option></select>
      <button id="preset_load" title="選んだ設定を読み込む">読込</button>
      <button id="preset_del" title="選んだ設定を削除">削除</button>
    </div>
    <div class="row saverow">
      <input id="preset_name" type="text" placeholder="メモ（空欄可）。保存名は「版_年月日-時分秒」">
      <button id="preset_save">保存</button>
    </div>
    <div class="row btnrow">
      <button id="export_cfg" title="今の設定をJSONファイルに書き出す">書き出す</button>
      <button id="import_cfg" title="JSONファイルから設定を読み込む">読み込む</button>
      <button id="reset_cfg" title="最初の設定に戻す">既定に戻す</button>
    </div>
    <input id="import_file" type="file" accept=".json,application/json" hidden>
    <div class="hint">最後に使った設定は自動で保存され、次回起動時に復元されます。「書き出す」「読み込む」はJSONファイルでのやりとりです。書き出したファイルは <code>region_fill.py --config ファイル名</code> でも使えます</div>
  </details>

  <div class="row seed">
    <div><label>シード <span class="v">同じ値なら同じ絵</span></label><input id="seed" type="number" value="1"></div>
    <button id="dice" title="ランダムなシード" style="align-self:end">🎲</button>
  </div>
  <div class="row"><button id="gen" class="primary">生成</button></div>
  <label class="check"><input id="auto" type="checkbox" checked> 値を変えたら自動で生成</label>

  <details open><summary>線の形</summary>
    <div class="row"><label>長さ <span class="v" id="v_length"></span></label><input id="length" type="range" min="4" max="40" step="1" value="12"><div class="hint">長いほど交差と区画が増えます（画面の短辺の何倍か）</div></div>
    <div class="row"><label>クネクネ度 <span class="v" id="v_wiggle"></span></label><input id="wiggle" type="range" min="0.2" max="3" step="0.1" value="1"></div>
    <div class="row"><label>最小半径 <span class="v" id="v_min_radius"></span></label><input id="min_radius" type="range" min="0.5" max="8" step="0.5" value="3"><div class="hint" id="mr_hint">これより急なカーブ・小さなループを作りません</div></div>
    <div class="row"><label>なめらかさ <span class="v" id="v_smooth"></span></label><input id="smooth" type="range" min="0.3" max="3" step="0.1" value="1"><div class="hint">曲がりの入り・抜けの長さ</div></div>
    <div class="row two">
      <div><label>交差までの曲がり</label><select id="max_bends"><option value="0">制限なし</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>8</option></select></div>
      <div><label>線の本数</label><select id="curves"><option>1</option><option>2</option><option>3</option></select></div>
    </div>
    <div class="hint">曲がりの回数を制限すると線が絡み合います（2回はばらつきが大きく、まれに数十秒かかります。30秒で打ち切り）</div>
  </details>

  <details open><summary>重心と重なり回避</summary>
    <div class="row two">
      <div><label>重心の数（最小）</label><input id="att_min" type="number" min="0" max="5" value="1"></div>
      <div><label>（最大）</label><input id="att_max" type="number" min="0" max="5" value="3"></div>
    </div>
    <div class="row"><label>重心へ向かう強さ <span class="v" id="v_pull"></span></label><input id="pull" type="range" min="0" max="1" step="0.05" value="0.5"></div>
    <div class="row"><label>重なりを避ける強さ <span class="v" id="v_avoid"></span></label><input id="avoid" type="range" min="0" max="1" step="0.05" value="0.6"><div class="hint">同じ場所を線が何度も並んで通らないようにします</div></div>
    <label class="check"><input id="show_att" type="checkbox"> 重心の位置を表示（保存には含まれません）</label>
  </details>

  <details open><summary>線の見た目</summary>
    <div class="row"><label>線の太さ <span class="v" id="v_stroke"></span></label><input id="stroke" type="range" min="0" max="4" step="0.1" value="1.2"><div class="hint">0 で線なし（塗りだけ）</div></div>
    <div class="row"><label>線の強弱 <span class="v" id="v_pressure"></span></label><input id="pressure" type="range" min="0" max="1" step="0.05" value="0.5"></div>
    <div class="row colors2">
      <div><label>線の色</label><input id="line_color" type="color" value="#111111"></div>
      <div><label>背景色</label><input id="bg" type="color" value="#ffffff"></div>
    </div>
  </details>

  <details open><summary>塗りの色</summary>
    <div class="row"><div class="swatches" id="swatches"></div><div class="hint">4色以上。クリックで色を変更、×で削除</div></div>
    <div class="row"><div class="cats" id="pal_cats"></div><div class="pals" id="pal_list"></div></div>
    <div class="row harmony"><select id="harm_mode" title="配色のルール"></select><button id="harm_go" title="ルールに沿った配色をランダムに作る">🎨 おまかせ配色</button></div>
    <div class="row imgrow"><img id="img_thumb" alt="">
      <button id="img_pal" title="画像の代表色を取り出して配色にします（ここに画像をドラッグしても可）">🖼 画像から配色</button>
      <select id="img_n" title="取り出す色の数" style="width:auto"><option>4</option><option selected>5</option><option>6</option><option>7</option><option>8</option><option>10</option></select></div>
    <label class="check" title="オフにすると、面積の大きい色（背景の白など）を優先します"><input id="img_vivid" type="checkbox" checked> 鮮やかな色を優先</label>
    <input id="img_file" type="file" accept="image/*" hidden>
    <div class="hint" id="pal_note">プリセットを選ぶと、線の形はそのままで色だけ変わります</div>
    <div class="row"><label>色選びのランダムさ <span class="v" id="v_randomness"></span></label><input id="randomness" type="range" min="0" max="1" step="0.05" value="0.5"><div class="hint">0 = 各色を均等に使う / 1 = 完全にランダム</div></div>
    <div class="row"><button id="recolor" style="width:100%">線はそのままで、色だけ変える</button></div>
  </details>

  <details><summary>サイズ</summary>
    <div class="row two">
      <div><label>幅 (px)</label><input id="width" type="number" min="200" max="4000" value="800"></div>
      <div><label>高さ (px)</label><input id="height" type="number" min="200" max="4000" value="500"></div>
    </div>
  </details>
</aside>

<main>
  <div class="bar">
    <span id="status" class="status grow">「生成」を押してください</span>
    <span id="estimate" class="estimate status" title="今の設定で1枚作るのにかかる時間の目安（実際にかかった時間から、このパソコンの速さに合わせて補正）"></span>
    <label class="check" style="margin:0" title="曲線の外側（背景）を透明にして、図形だけを保存します"><input id="transparent" type="checkbox"> 背景を透過</label>
    <button id="save_svg" disabled>SVGを保存</button>
    <select id="png_scale" style="width:auto"><option value="1">PNG 1倍</option><option value="2" selected>PNG 2倍</option><option value="4">PNG 4倍</option></select>
    <button id="save_png" disabled>PNGを保存</button>
  </div>
  <div class="stage">
    <div class="frame" id="frame" style="width:800px">
      <div id="canvas"><div class="empty" style="padding:120px 0;text-align:center;line-height:1.4">ここにプレビューが表示されます</div></div>
      <svg class="overlay" id="overlay"></svg>
      <div class="busy" id="busy"><div class="card"><div>生成中… <span class="sec" id="busy_sec"></span></div><button id="cancel">中断</button><div class="sec">Esc キーでも中断できます</div></div></div>
    </div>
  </div>
  <div class="foot">プレビューは保存される SVG そのものです。PNG はブラウザ内で SVG から書き出します。</div>
</main>
</div>

<script>
let PALETTES = [{category:"カラフル", name:"ポップ", colors:["#FFE600","#FF8A1F","#FFB3C7","#1EA0E6","#9ADCF0"]}];
let curCat = "すべて", selPal = "ポップ", palOverride = false;
let colors = PALETTES[0].colors.slice();
let colorSeed = 1;
let lastSvg = null, lastAttractors = [], reqId = 0, timer = null;
const $ = id => document.getElementById(id);
const sliders = {length:v=>v+"倍", wiggle:v=>v, min_radius:v=>v+"%", smooth:v=>v, pull:v=>v, avoid:v=>v,
                 stroke:v=>v==0?"なし":v+"%", pressure:v=>v==0?"均一":v, randomness:v=>v};

function showVals(){
  for (const [k,f] of Object.entries(sliders)) $("v_"+k).textContent = f($(k).value);
  if(typeof showEstimate==="function" && $("estimate")) showEstimate();
}

function renderSwatches(){
  const box = $("swatches"); box.innerHTML = "";
  colors.forEach((c,i)=>{
    const w = document.createElement("div"); w.className="sw";
    const inp = document.createElement("input"); inp.type="color"; inp.value=c;
    inp.oninput = e=>{ colors[i]=e.target.value; };
    inp.onchange = ()=>{ selPal=null; renderPalettes(); recolor(); };
    w.appendChild(inp);
    if (colors.length>4){
      const x=document.createElement("button"); x.className="x"; x.textContent="×"; x.title="この色を削除";
      x.onclick=()=>{ colors.splice(i,1); selPal=null; renderPalettes(); renderSwatches(); recolor(); };
      w.appendChild(x);
    }
    box.appendChild(w);
  });
  if (colors.length<10){
    const add=document.createElement("button"); add.textContent="＋"; add.title="色を追加";
    add.onclick=()=>{ colors.push("#"+Math.floor(Math.random()*0xffffff).toString(16).padStart(6,"0")); selPal=null; renderPalettes(); renderSwatches(); recolor(); };
    box.appendChild(add);
  }
}
function renderPalettes(){
  const cats=["すべて", ...new Set(PALETTES.map(p=>p.category))];
  const cb=$("pal_cats"); cb.innerHTML="";
  for(const c of cats){
    const b=document.createElement("button"); b.className="cat"+(c===curCat?" on":""); b.textContent=c;
    b.onclick=()=>{ curCat=c; renderPalettes(); $("pal_list").scrollTop=0; };
    cb.appendChild(b);
  }
  const box=$("pal_list"); box.innerHTML="";
  for(const p of PALETTES){
    if(curCat!=="すべて" && p.category!==curCat) continue;
    const b=document.createElement("button"); b.className="pal"+(p.name===selPal?" sel":"");
    b.dataset.name=p.name;
    b.title=`${p.category}／${p.name}`+((p.line||p.bg)?"（線と背景の色も変わります）":"");
    const strip=document.createElement("div"); strip.className="strip";
    p.colors.forEach(c=>{const s=document.createElement("span"); s.style.background=c; strip.appendChild(s);});
    const nm=document.createElement("div"); nm.className="nm"; nm.textContent=p.name+((p.line||p.bg)?" ◐":"");
    b.appendChild(strip); b.appendChild(nm);
    b.onclick=()=>applyPalette(p);
    box.appendChild(b);
  }
}
function applyPalette(p, note){
  colors=p.colors.slice(); selPal=p.name||null;
  if(p.line||p.bg){
    if(p.line) $("line_color").value=p.line.toLowerCase();
    if(p.bg) $("bg").value=p.bg.toLowerCase();
    palOverride=true;
  }else if(palOverride){                     // 前のプリセットで変えた線・背景を元に戻す
    $("line_color").value="#111111"; $("bg").value="#ffffff"; palOverride=false;
  }
  renderSwatches(); renderPalettes();
  $("pal_note").textContent = note || (p.name ? `「${p.name}」を適用しました`+((p.line||p.bg)?"（線と背景の色も変更）":"") : "");
  recolor();
}
let VERSION="", STYLES=[];
function stamp(){
  const d=new Date(), z=n=>String(n).padStart(2,"0");
  return `v${VERSION}_${d.getFullYear()}${z(d.getMonth()+1)}${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}`;
}
async function loadPalettes(){
  try{
    const j=await getJSON("/api/palettes");
    PALETTES=j.palettes; VERSION=j.version||""; STYLES=j.styles||[];
    $("ver").textContent = VERSION ? "v"+VERSION : "";
    const sel=$("harm_mode"); sel.innerHTML="";
    for(const m of j.modes){ const o=document.createElement("option"); o.value=m; o.textContent=m; sel.appendChild(o); }
  }catch(e){}
  renderPalettes();
}

function params(){
  const g = id => $(id).value;
  let amin = parseInt(g("att_min"))||0, amax = parseInt(g("att_max"))||0;
  if (amax<amin) amax=amin;
  return {width:parseInt(g("width"))||800, height:parseInt(g("height"))||500, curves:g("curves"),
    seed:parseInt(g("seed"))||0, randomness:g("randomness"), wiggle:g("wiggle"), stroke:g("stroke"),
    min_radius:g("min_radius"), length:g("length"), max_bends:g("max_bends"), att_min:amin, att_max:amax,
    pull:g("pull"), avoid:g("avoid"), smooth:g("smooth"), pressure:g("pressure"),
    line_color:g("line_color"), bg:g("bg"), colors:colors.slice(),
    auto:$("auto").checked, show_att:$("show_att").checked, png_scale:g("png_scale"), palette:selPal};
}

// ---- 設定の反映・保存
const FIELDS=["width","height","curves","seed","randomness","wiggle","stroke","min_radius","length","max_bends",
  "att_min","att_max","pull","avoid","smooth","pressure","line_color","bg","png_scale"];
let DEFAULTS=null;
function applySettings(st, regenerate=true){
  if(!st) return;
  for(const k of FIELDS){
    if(st[k]===undefined || st[k]===null) continue;
    const el=$(k); if(!el) continue;
    el.value=String(st[k]);
  }
  if(Array.isArray(st.colors) && st.colors.length>=4) colors=st.colors.slice();
  selPal = st.palette || null;
  if(st.auto!==undefined) $("auto").checked=!!st.auto;
  if(st.show_att!==undefined) $("show_att").checked=!!st.show_att;
  showVals(); renderSwatches(); renderPalettes(); fitFrame();
  $("pal_note").textContent = selPal ? `配色「${selPal}」` : "";
  // 選んでいる配色が一覧の中で見えるように（サイドバー全体はスクロールさせない）
  const cur=document.querySelector("#pal_list .pal.sel"), box=$("pal_list");
  if(cur){
    const top=cur.offsetTop, bottom=top+cur.offsetHeight;
    if(top<box.scrollTop) box.scrollTop=top-4;
    else if(bottom>box.scrollTop+box.clientHeight) box.scrollTop=bottom-box.clientHeight+4;
  }
  if(regenerate) generate();
}
function fillPresets(presets, select){
  const sel=$("preset_list"); sel.innerHTML='<option value="">設定を選ぶ…</option>';
  if(STYLES.length){
    const g=document.createElement("optgroup"); g.label="おすすめ設定";
    for(const st of STYLES){
      const o=document.createElement("option"); o.value="style:"+st.name; o.textContent=`${st.name}（${st.desc}）`; g.appendChild(o);
    }
    sel.appendChild(g);
  }
  const names=Object.keys(presets).sort().reverse();        // 新しい保存が上に来る
  if(names.length){
    const g=document.createElement("optgroup"); g.label="保存した設定";
    for(const name of names){ const o=document.createElement("option"); o.value=name; o.textContent=name; g.appendChild(o); }
    sel.appendChild(g);
  }
  window._presets=presets;
  if(select) sel.value=select;
}
async function getJSON(url){ const r=await fetch(url); return r.json(); }

function setStatus(t, err){ const s=$("status"); s.textContent=t; s.className="status grow"+(err?" err":""); }

function fitFrame(){
  const p=params(), stage=document.querySelector(".stage");
  const maxW=stage.clientWidth-48, maxH=stage.clientHeight-48;
  const s=Math.min(maxW/p.width, maxH/p.height, 1.6);
  $("frame").style.width=Math.max(120, Math.floor(p.width*s))+"px";
}

function showSvg(svg){
  lastSvg=svg; $("canvas").innerHTML=svg; drawOverlay();
  $("save_svg").disabled=false; $("save_png").disabled=false;
}
function drawOverlay(){
  const o=$("overlay"), p=params();
  o.setAttribute("viewBox",`0 0 ${p.width} ${p.height}`); o.innerHTML="";
  if(!$("show_att").checked) return;
  const r=Math.min(p.width,p.height)*0.2;
  for(const [x,y] of lastAttractors){
    o.insertAdjacentHTML("beforeend",
      `<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="#d6292b" stroke-width="1.5" stroke-dasharray="6 5" opacity=".7"/>
       <circle cx="${x}" cy="${y}" r="7" fill="none" stroke="#d6292b" stroke-width="2.5"/>
       <path d="M${x-11},${y}H${x+11}M${x},${y-11}V${y+11}" stroke="#d6292b" stroke-width="2.5"/>`);
  }
}

async function post(url, body){
  let r;
  try{ r = await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}); }
  catch(e){ throw new Error("サーバーに接続できません。端末で region_fill_gui.py が動いているか確認し、このページを再読み込みしてください"); }
  const j = await r.json();
  if(!r.ok) throw new Error(j.error||"エラーが発生しました");
  return j;
}

let busyTimer=null, busyStart=0, pendingNote="";
function setBusy(on){
  $("busy").classList.toggle("on",on); $("gen").disabled=on;
  clearInterval(busyTimer); $("busy_sec").textContent="";
  if(on){ busyStart=Date.now(); busyTimer=setInterval(()=>{ $("busy_sec").textContent=((Date.now()-busyStart)/1000).toFixed(0)+"秒"; },500); }
}
async function cancelGenerate(){
  if(!$("busy").classList.contains("on")) return;
  ++reqId;                                          // 返ってくる結果は捨てる
  setBusy(false);
  setStatus("中断しました（プレビューは直前のままです）");
  try{ await post("/api/cancel",{}); }catch(e){}
}
async function generate(quiet){
  // quiet: 起動時、保存してあった前回の絵を表示したまま、裏で再計算だけする（絵は差し替えない）
  const id=++reqId, p=params();
  if (p.colors.length<4){ setStatus("色は4色以上にしてください",true); return; }
  fitFrame();
  if(!quiet){ setBusy(true); setStatus("生成中…"); }
  try{
    const j = await post("/api/generate", quiet ? {...p, quiet:true} : p);
    if(id!==reqId) return;                         // 新しい生成が始まっていたら捨てる
    if(j.cancelled){ if(!quiet) setStatus("中断しました（プレビューは直前のままです）"); return; }
    if(quiet){ lastAttractors=j.attractors; drawOverlay(); return; }
    lastAttractors=j.attractors; showSvg(j.svg);
    const s=j.stats;
    learnSpeed(p, s.seconds); showEstimate();
    let t=`区画 ${s.faces} ・ 交点 ${s.crossings} ・ 線 ${s.stroke_px}px ・ 最小半径 ${s.min_radius_px}px`;
    if(s.raised) t+="（線の太さに合わせて引き上げ）";
    if(p.max_bends>0) t+=` ・ やり直し ${s.rollbacks}回`;
    t+=` ・ ${s.seconds}秒`;
    if(pendingNote){ t=pendingNote+" ／ "+t; pendingNote=""; }
    setStatus(t);
  }catch(e){ if(id===reqId && !quiet) setStatus(e.message,true); }
  finally{ if(id===reqId && !quiet) setBusy(false); }
}

let recolorId=0;
async function recolor(newSeed){
  if(!lastSvg) return;
  const rid=++recolorId;
  if(colors.length<4){ setStatus("色は4色以上にしてください",true); return; }
  if(newSeed) colorSeed++;
  const p=params();
  try{
    const j=await post("/api/recolor",{colors:p.colors, randomness:p.randomness, color_seed:colorSeed,
                                        line_color:p.line_color, bg:p.bg});
    if(rid!==recolorId) return;                    // 後から押した操作の結果を優先する
    showSvg(j.svg);
  }catch(e){ setStatus(e.message,true); }
}

// ---- 負荷（生成時間）の目安：v1.1 で実測した値から作った式。既定の設定を 1 とした倍率
const LOAD_INFO = {
  length:     ["中", "長いほど重い（12→20で約2倍、40で約5倍）。区画の数もほぼ比例して増えます"],
  min_radius: ["大", "3%以下はほぼ影響なし。6%で約4倍、8%で約9倍。10%以上はほとんど生成できません"],
  smooth:     ["小", "ほぼ影響なし"],
  max_bends:  ["大", "制限なしが最も軽い。4〜6回で約1〜2倍、3回で約2倍、2回で約12倍（ばらつきが大きい）"],
  curves:     ["中", "2本で約2倍、3本で約3倍"],
  wiggle:     ["小", "ほぼ影響なし"],
  att_min:    ["小", "ほぼ影響なし"],
  pull:       ["小", "ほぼ影響なし"],
  avoid:      ["小", "ほぼ影響なし"],
  stroke:     ["なし", "生成時間には影響しません"],
  pressure:   ["なし", "生成時間には影響しません"],
  width:      ["なし", "サイズは比率で扱うので、生成時間には影響しません"],
};
function loadFactor(p){
  const L=+p.length, r=+p.min_radius, mb=+p.max_bends, nc=+p.curves;
  let f = L<=12 ? Math.max(0.4, L/12) : Math.pow(L/12, 1.3);
  f *= r<=3 ? 1 : Math.exp((r-3)*0.45);
  f *= ({0:1, 8:1.1, 6:1.2, 5:1.6, 4:2, 3:2.5, 2:12})[mb] || 1;
  f *= [1, 1, 2.1, 3.1][nc] || 1;
  return f;
}
let speedScale = 0.07;                       // 既定の設定での1枚の秒数（実測で更新）
try{ const v=parseFloat(localStorage.getItem("rf_speed")); if(v>0) speedScale=v; }catch(e){}
function learnSpeed(p, sec){
  if(!(sec>0)) return;
  const est = sec / loadFactor(p);
  speedScale = Math.exp(0.7*Math.log(speedScale) + 0.3*Math.log(est));   // ゆっくり追従
  try{ localStorage.setItem("rf_speed", String(speedScale)); }catch(e){}
}
function showEstimate(){
  const p=params(), t=speedScale*loadFactor(p);
  const lv = t<0.3 ? "軽い" : t<1.5 ? "普通" : t<5 ? "重い" : "とても重い";
  const sec = t<1 ? t.toFixed(1) : t<10 ? t.toFixed(0) : String(Math.round(t/5)*5);
  $("estimate").textContent = `予想 約${sec}秒（${lv}）`;
  $("estimate").style.color = t<1.5 ? "" : t<5 ? "#7A5800" : "#9B2C1F";
  const mr=+p.min_radius;
  $("mr_hint").textContent = mr>6 ? "大きすぎると生成に失敗しやすくなります（画面の狭い側に対して大きい）"
                                  : "これより急なカーブ・小さなループを作りません";
  $("mr_hint").style.color = mr>6 ? "#9B2C1F" : "";
}
function addLoadBadges(){
  for(const [id,[lv,tip]] of Object.entries(LOAD_INFO)){
    const el=$(id); if(!el) continue;
    const lab=el.previousElementSibling && el.previousElementSibling.tagName==="LABEL" ? el.previousElementSibling : null;
    if(!lab || lab.querySelector(".load")) continue;
    const b=document.createElement("span");
    b.className="load "+(lv==="大"?"l2":lv==="中"?"l1":lv==="なし"?"ln":"l0"); b.textContent="負荷 "+lv; b.title=tip;
    // 項目名と負荷を1つにまとめて左寄せ（値の表示は右端のまま）
    const t=lab.firstChild, grp=document.createElement("span"); grp.className="lt";
    lab.insertBefore(grp, t); grp.appendChild(t); grp.appendChild(b);
  }
}

function scheduleAuto(){
  showVals();
  if(!$("auto").checked){ setStatus("設定を変えました。「生成」を押すと反映されます（自動で生成はオフ）"); return; }
  clearTimeout(timer); timer=setTimeout(()=>generate(),300);
}

function download(name, blob){
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=name;
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(a.href),2000);
}
function baseName(){ const p=params(); return `region_${stamp()}_s${p.seed}`; }
function exportSvg(){
  // 「背景を透過」なら背景の四角形を取り除く（曲線の外側が透明になり、図形だけが残る）
  if(!$("transparent").checked) return lastSvg;
  return lastSvg.replace(/\s*<rect id="background"[^>]*\/>/, "");
}
$("save_svg").onclick=()=>{
  const t=$("transparent").checked;
  download(baseName()+(t?"_透過":"")+".svg", new Blob([exportSvg()],{type:"image/svg+xml"}));
};
$("save_png").onclick=()=>{
  const p=params(), k=parseInt($("png_scale").value), t=$("transparent").checked;
  const img=new Image();
  img.onload=()=>{
    const c=document.createElement("canvas"); c.width=p.width*k; c.height=p.height*k;
    c.getContext("2d").drawImage(img,0,0,c.width,c.height);
    c.toBlob(b=>download(baseName()+(k>1?`@${k}x`:"")+(t?"_透過":"")+".png", b),"image/png");
  };
  img.src="data:image/svg+xml;charset=utf-8,"+encodeURIComponent(exportSvg());
};

// ---- 画像から配色
let lastImg=null;                                   // {w,h,data,name}
function readImage(file){
  return new Promise((ok,ng)=>{
    const url=URL.createObjectURL(file), im=new Image();
    im.onload=()=>{
      const s=Math.min(1,160/Math.max(im.naturalWidth,im.naturalHeight));
      const w=Math.max(1,Math.round(im.naturalWidth*s)), h=Math.max(1,Math.round(im.naturalHeight*s));
      const c=document.createElement("canvas"); c.width=w; c.height=h;
      const g=c.getContext("2d"); g.drawImage(im,0,0,w,h);
      const d=g.getImageData(0,0,w,h).data;
      $("img_thumb").src=url; $("img_thumb").style.display="block";
      ok({w,h,data:Array.from(d),name:file.name});
    };
    im.onerror=()=>{ URL.revokeObjectURL(url); ng(new Error("画像として読み込めませんでした")); };
    im.src=url;
  });
}
async function paletteFromImage(){
  if(!lastImg) return;
  const n=parseInt($("img_n").value);
  try{
    const j=await post("/api/palette_from_image",{w:lastImg.w,h:lastImg.h,data:lastImg.data,n,vivid:$("img_vivid").checked});
    applyPalette({colors:j.colors}, `「${lastImg.name}」から${j.colors.length}色を取り出しました。気に入ったら上の欄で保存できます`);
  }catch(e){ setStatus(e.message,true); }
}
async function useImageFile(file){
  if(!file) return;
  if(!/^image\//.test(file.type)){ setStatus("画像ファイルを選んでください",true); return; }
  try{ lastImg=await readImage(file); await paletteFromImage(); }
  catch(e){ setStatus(e.message,true); }
}
$("img_pal").onclick=()=>$("img_file").click();
$("img_file").onchange=e=>{ const f=e.target.files[0]; e.target.value=""; useImageFile(f); };
$("img_n").onchange=paletteFromImage;
$("img_vivid").onchange=paletteFromImage;
{
  const box=$("img_pal").closest("details");
  box.addEventListener("dragover",e=>{ e.preventDefault(); box.classList.add("drop"); });
  box.addEventListener("dragleave",()=>box.classList.remove("drop"));
  box.addEventListener("drop",e=>{ e.preventDefault(); box.classList.remove("drop"); useImageFile(e.dataTransfer.files[0]); });
}

// イベント
for (const id of Object.keys(sliders)) $(id).addEventListener("input", id==="randomness"?showVals:scheduleAuto);
$("randomness").addEventListener("change", ()=>recolor(false));
for (const id of ["max_bends","curves","att_min","att_max","width","height","seed","line_color","bg"])
  $(id).addEventListener("change", scheduleAuto);
for (const id of ["max_bends","curves"]) $(id).addEventListener("input", showEstimate);
$("gen").onclick=()=>generate();
$("dice").onclick=()=>{ $("seed").value=Math.floor(Math.random()*100000); generate(); };
$("recolor").onclick=()=>recolor(true);
$("show_att").onchange=drawOverlay;
window.addEventListener("resize", fitFrame);

$("harm_go").onclick=async()=>{
  try{
    const j=await post("/api/harmony",{mode:$("harm_mode").value, seed:Math.floor(Math.random()*1e9), n:Math.max(5,colors.length)});
    applyPalette({colors:j.colors}, `おまかせ配色（${j.mode}）を作りました。気に入ったら上の欄で名前を付けて保存できます`);
  }catch(e){ setStatus(e.message,true); }
};
$("cancel").onclick=cancelGenerate;
document.addEventListener("keydown", e=>{ if(e.key==="Escape") cancelGenerate(); });
$("auto").onchange=()=>{ if($("auto").checked) generate(); }; $("png_scale").onchange=()=>{};

$("preset_save").onclick=async()=>{
  const memo=$("preset_name").value.trim().replace(/[\\/:*?"<>|]/g,"_");
  const name=stamp()+(memo?"_"+memo:"");                // 保存名：バージョン名＋年月日時分秒（＋メモ）
  if(window._presets && window._presets[name] && !confirm(`「${name}」を上書きしますか？`)) return;
  try{ const j=await post("/api/presets/save",{name, settings:params()}); fillPresets(j.presets,name);
       $("preset_name").value=""; setStatus(`設定「${name}」を保存しました`); }
  catch(e){ setStatus(e.message,true); }
};
$("preset_load").onclick=()=>{
  const name=$("preset_list").value;
  if(!name){ setStatus("読み込む設定を選んでください",true); return; }
  if(name.startsWith("style:")){
    const st=STYLES.find(x=>"style:"+x.name===name);
    pendingNote=`おすすめ設定「${st.name}」を読み込みました`;
    applySettings({...DEFAULTS, ...st.settings, seed:$("seed").value});   // シードは今のまま
    return;
  }
  pendingNote=`設定「${name}」を読み込みました`; applySettings(window._presets[name]);
};
$("preset_list").ondblclick=()=>$("preset_load").click();
$("preset_del").onclick=async()=>{
  const name=$("preset_list").value;
  if(!name){ setStatus("削除する設定を選んでください",true); return; }
  if(name.startsWith("style:")){ setStatus("おすすめ設定は削除できません",true); return; }
  if(!confirm(`設定「${name}」を削除しますか？`)) return;
  try{ const j=await post("/api/presets/delete",{name}); fillPresets(j.presets); setStatus(`設定「${name}」を削除しました`); }
  catch(e){ setStatus(e.message,true); }
};
$("export_cfg").onclick=()=>{
  const name=stamp();
  const data={app:"region_fill", version:1, app_version:VERSION, name, saved:new Date().toISOString(), settings:params()};
  download(`region_fill_${name}.json`, new Blob([JSON.stringify(data,null,2)],{type:"application/json"}));
};
$("import_cfg").onclick=()=>$("import_file").click();
$("import_file").onchange=async e=>{
  const f=e.target.files[0]; e.target.value="";
  if(!f) return;
  try{
    const data=JSON.parse(await f.text());
    const st=data.settings||data;
    if(typeof st!=="object" || !("seed" in st || "length" in st || "colors" in st)) throw new Error("設定ファイルの形式ではありません");
    pendingNote=`「${f.name}」から読み込みました`; applySettings(st);
  }catch(err){ setStatus("読み込めませんでした："+err.message,true); }
};
$("reset_cfg").onclick=()=>{ pendingNote="既定の設定に戻しました"; applySettings(DEFAULTS); };

(async()=>{
  addLoadBadges(); showVals(); renderSwatches(); fitFrame();
  await loadPalettes();
  DEFAULTS=params();
  try{
    const data=await getJSON("/api/settings");
    fillPresets(data.presets||{});
    if(data.last){ applySettings(data.last,false);
      pendingNote = "前回の設定を復元しました" + ($("auto").checked ? "" : "（自動で生成はオフです）"); }
    if(data.last && data.view && data.view.svg){
      // 前回の絵をすぐに表示し、「色だけ変える」等に必要な計算だけ裏で行う
      lastAttractors=data.view.attractors||[]; showSvg(data.view.svg);
      setStatus(pendingNote.replace("復元しました","復元し、前回の絵を表示しています")); pendingNote="";
      generate(true);
      return;
    }
  }catch(e){}
  await generate();
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="region_fill のGUI（ブラウザで操作）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    ap.add_argument("--settings", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "region_fill_settings.json"),
                    help="設定の保存先（既定: このスクリプトと同じフォルダの region_fill_settings.json）")
    args = ap.parse_args()
    global STORE
    STORE = SettingsStore(args.settings)
    port = args.port
    for _ in range(20):                       # 使用中なら次のポートを試す
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    else:
        raise SystemExit("空いているポートが見つかりませんでした。--port で指定してください。")
    url = f"http://127.0.0.1:{port}/"
    print(f"Region Fill GUI: {url}  （終了するには Ctrl+C）")
    print(f"設定の保存先: {args.settings}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n終了しました。")


if __name__ == "__main__":
    main()
