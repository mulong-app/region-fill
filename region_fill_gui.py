#!/usr/bin/env python3
"""
region_fill.py のGUI（ブラウザで操作）。 v1.2.0

  python region_fill_gui.py            # ブラウザが自動で開きます
  python region_fill_gui.py --port 9000 --no-browser
  python region_fill_gui.py --settings my_settings.json   # 設定の保存先を変える

Python 標準ライブラリだけで小さなローカルサーバーを立て、ブラウザで操作します。
プレビューは出力される SVG そのものなので、表示と保存結果が完全に一致します。
同じフォルダに region_fill.py・region_fill_api.py・index.html を置いてください（必要なのは numpy だけ）。
インストール不要のブラウザ版（GitHub Pages）も、同じ index.html と region_fill_api.py を使います。
"""
import argparse
import json
import os
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import region_fill as rf
import region_fill_api as api

# 実行中の生成（中断用）。新しい生成が始まると、前の生成は自動で中断する。
# 生成は常に1つだけ走らせる（前の生成が止まるのを待ってから始める）ので、
# 重い処理が裏に溜まって全体が遅くなることがない
_job = {"stop": None}
_job_lock = threading.Lock()
_run_lock = threading.Lock()


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


def api_generate(p):
    stop = _start_job()
    if not _run_lock.acquire(timeout=10):        # 前の生成が止まるのを待つ
        raise RuntimeError("前の生成が止まりませんでした。少し待ってからもう一度お試しください。")
    try:
        if stop.is_set():                        # 待っている間に、さらに新しい生成が来た
            raise rf.Cancelled()
        out = api.generate(p, should_stop=stop.is_set)
    finally:
        _run_lock.release()
    if STORE is not None and not p.get("quiet"):
        STORE.save_last({k: v for k, v in p.items() if k != "quiet"})   # 次回起動時に復元
        STORE.save_view(out["svg"], out["attractors"])
    return out


def api_recolor(p):
    out = api.recolor(p)
    if STORE is not None:
        STORE.save_view(out["svg"], out["attractors"])
    return out


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
            self._send(200, load_page(), "text/html; charset=utf-8")
        elif self.path == "/api/palettes":
            self._send(200, json.dumps(api.palettes()), "application/json")
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
                out = api.harmony(p)
            elif self.path == "/api/palette_from_image":
                out = api.palette_from_image(p)
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


def load_page():
    """画面（index.html）を読み込む。パソコン版とブラウザ版（GitHub Pages）で同じファイルを使う。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    try:
        with open(path, encoding="utf-8") as fp:
            return fp.read()
    except OSError:
        raise SystemExit(f"画面のファイル {path} が見つかりません。region_fill_gui.py と同じフォルダに "
                         "index.html を置いてください（ZIP を展開したフォルダから起動してください）。")




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
    load_page()                               # 画面のファイルがあるか、起動時に確かめる
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
