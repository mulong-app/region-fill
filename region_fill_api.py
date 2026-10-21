"""
GUI から呼ばれる処理（生成・色だけ変える・配色）。

パソコン版（region_fill_gui.py のローカルサーバー）と、ブラウザ版（Pyodide 上の Web Worker）の
両方がこのファイルを使う。サーバーやスレッドには依存しない。

ブラウザ版からは call(名前, JSON文字列) → JSON文字列 で呼ぶ。
"""
import json
import random
import time

import numpy as np

import region_fill as rf

GUI_TIME_LIMIT = 30.0          # GUI では 30 秒で打ち切って理由を表示する

# 直近の生成結果（「色だけ変える」で使う）。差し替えは代入1回で行う
_state = {"last": None}


def uniform_stroke(p, info):
    """線の強弱なし、または線なしのときは均一な線幅（None なら強弱つき）。"""
    s = info["stroke_px"]
    return s if (float(p["pressure"]) <= 0 or s <= 0) else None


def generate(p, should_stop=None):
    """設定 p で1枚作る。戻り値は画面に返す内容（svg・重心・統計）。"""
    t0 = time.time()
    colors = [c for c in p["colors"] if c]
    curves, faces, fill_cols, adj, widths, info = rf.generate(
        int(p["width"]), int(p["height"]), int(p["curves"]), colors, int(p["seed"]),
        float(p["randomness"]), float(p["wiggle"]), f'{float(p["stroke"])}%',
        min_radius=f'{float(p["min_radius"])}%', length=float(p["length"]), max_bends=int(p["max_bends"]),
        attractors=f'{int(p["att_min"])}-{int(p["att_max"])}', pull=float(p["pull"]),
        avoid=float(p["avoid"]), smooth=float(p["smooth"]), pressure=float(p["pressure"]),
        should_stop=should_stop, time_limit=GUI_TIME_LIMIT)
    _state["last"] = {"curves": curves, "faces": faces, "adj": adj, "widths": widths, "info": info,
                      "params": dict(p)}
    svg = rf.svg_string(int(p["width"]), int(p["height"]), faces, fill_cols, curves, widths,
                        p["line_color"], p["bg"], uniform_stroke(p, info))
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


# 線・区画（と線の強弱）を決める設定。色数も乱数の進み方に影響するので含める
GEN_KEYS = ("width", "height", "curves", "seed", "randomness", "wiggle", "stroke", "min_radius", "length",
            "max_bends", "att_min", "att_max", "pull", "avoid", "smooth", "pressure")


def _gen_key(p):
    def norm(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)
    return tuple(norm(p.get(k)) for k in GEN_KEYS) + (len([c for c in p.get("colors", []) if c]),)


def needs_regen(gen):
    """gen（表示中の絵を作った設定）と、手元にある直近の結果が違うか。"""
    if not gen:
        return False
    last = _state["last"]
    return last is None or _gen_key(last["params"]) != _gen_key(gen)


def recolor(p, should_stop=None):
    """直近の線と区画はそのままで、色だけ塗り直す。

    p["gen"]（表示中の絵を作った設定）が直近の結果と違うとき（元に戻す・履歴から選んだあと等）は、
    同じ設定で線を作り直してから塗る（同じ設定なら同じ線になる）。
    """
    gen = p.get("gen")
    if needs_regen(gen):
        generate(gen, should_stop=should_stop)
    last = _state["last"]
    if last is None:
        raise ValueError("まだ準備中です（前回の絵を再計算しています）。少し待ってからもう一度お試しください。")
    params = dict(last["params"])
    for k in ("line_color", "bg"):              # 配色プリセットで線・背景が変わる場合に反映
        if p.get(k):
            params[k] = p[k]
    colors = [c for c in p["colors"] if c]
    if len(colors) < 4:
        raise ValueError("色は4色以上指定してください。")
    rng = random.Random(int(p["color_seed"]))
    cidx = rf.color_graph(last["adj"], len(colors), rng, float(p["randomness"]))
    svg = rf.svg_string(int(params["width"]), int(params["height"]), last["faces"], [colors[c] for c in cidx],
                        last["curves"], last["widths"], params["line_color"], params["bg"],
                        uniform_stroke(params, last["info"]))
    return {"svg": svg, "attractors": last["info"]["attractors"]}


def harmony(p):
    rng = random.Random(p.get("seed"))
    cols, mode = rf.harmony_palette(rng, p.get("mode", "おまかせ"), int(p.get("n", 5)))
    return {"colors": cols, "mode": mode}


def palette_from_image(p):
    """ブラウザで縮小した画像の画素（RGBA）から配色を作る。"""
    w, h = int(p["w"]), int(p["h"])
    data = np.asarray(p["data"], dtype=np.uint8)
    if w <= 0 or h <= 0 or data.size != w * h * 4 or w * h > 400 * 400:
        raise ValueError("画像を読み込めませんでした。")
    n = max(4, min(10, int(p.get("n", 5))))
    cols = rf.palette_from_array(data.reshape(h, w, 4), n, vivid=bool(p.get("vivid", True)))
    return {"colors": cols}


def palettes(_p=None):
    """配色プリセット・おまかせ配色のルール・おすすめ設定・バージョン。"""
    styles = [{"name": st["name"], "desc": st["desc"], "settings": rf.style_settings(st["name"])}
              for st in rf.STYLE_PRESETS]
    return {"palettes": rf.PALETTES, "modes": rf.HARMONY_MODES, "styles": styles, "version": rf.__version__}


_CALLS = {"generate": generate, "recolor": recolor, "harmony": harmony,
          "palette_from_image": palette_from_image, "palettes": palettes}


def call(name, body_json):
    """ブラウザ版の入口：JSON文字列で受け取り、JSON文字列で返す（エラーも {"error": ...} で返す）。"""
    try:
        p = json.loads(body_json) if body_json else {}
        return json.dumps(_CALLS[name](p), ensure_ascii=False)
    except rf.Cancelled:
        return json.dumps({"cancelled": True})
    except (ValueError, RuntimeError, KeyError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
