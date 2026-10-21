#!/usr/bin/env python3
"""
自由曲線で画面を区切り、区画を自動で塗り分けて SVG に出力するツール。

  - 一筆書きの閉曲線（何度も自分と交差する）を「曲がりながら進むペン先」で生成
      * 曲がる強さの上限で最小曲率半径を保証、曲がりの入り抜けはなめらか（クロソイド状）
      * 1〜3個の重心へ緩く向かい、線の密度マップを見て同じ場所を何度も通らない
      * 交差するまでに曲がってよい回数を制限できる（超えたら直前の交差地点からやり直し）
  - 曲線どうしの交点を厳密に計算し、平面グラフとして区画（面）を列挙
      * 塗りの輪郭は線と同じベジェ曲線を分割したもの（線と塗りがぴったり一致）
      * 「辺」を共有する区画どうしを隣接とみなし（点で接するだけなら別扱い）
        指定色数でグラフ彩色（DSatur + バックトラック）
  - 線は曲率に応じて太さが変わる（筆圧のような強弱）
  - 配色プリセット81種、おすすめ設定10種、画像から配色を作る機能、背景を透過した保存

依存: numpy（PNG を出すときは opencv-python、--palette-from を使うときは Pillow）
例:
  python region_fill.py -o out.svg --png out.png --seed 1
  python region_fill.py --attractors 2 --pull 0.6 --avoid 0.8 --pressure 0.7
  python region_fill.py --style 繊細 --palette 夕焼け --transparent --png
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np

__version__ = "1.5.1"

DEFAULT_COLORS = ["#FFE600", "#FF8A1F", "#FFB3C7", "#1EA0E6", "#9ADCF0"]


def version_stamp(t=None):
    """保存名の既定：バージョン名＋年月日時分秒（例: v1.3.0_20261021-111530）。"""
    t = time.localtime() if t is None else t
    return f"v{__version__}_{time.strftime('%Y%m%d-%H%M%S', t)}"


# ================================================================ 配色プリセット
def _srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def hex_to_oklab(h):
    """#RRGGBB → OKLab（人の目に近い明るさ・色差の色空間）。"""
    h = h.lstrip("#")
    rgb = _srgb_to_lin(np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]))
    lms = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]]) @ rgb
    lms = np.cbrt(lms)
    return np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                     [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]]) @ lms


def _oklab_to_rgb(lab):
    lms = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]]) @ lab
    lms = lms ** 3
    rgb = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]]) @ lms
    return _lin_to_srgb(rgb)


def oklab_to_hex(lab):
    rgb = np.clip(_oklab_to_rgb(np.asarray(lab, float)), 0, 1)
    return "#" + "".join(f"{int(round(v * 255)):02X}" for v in rgb)


def oklch_to_hex(L, C, h_deg):
    """明るさ L・鮮やかさ C・色相 h で色を作る。表示できない色は鮮やかさを下げて収める。"""
    h = math.radians(h_deg)
    for _ in range(40):
        lab = np.array([L, C * math.cos(h), C * math.sin(h)])
        rgb = _oklab_to_rgb(lab)
        if np.all(rgb >= -1e-4) and np.all(rgb <= 1 + 1e-4):
            break
        C *= 0.92
    return oklab_to_hex(lab)


def ramp(h_deg, chroma=0.13, steps=5, light=(0.93, 0.36)):
    """単色グラデーション：同じ色相で、明るい→暗いを均等な段差で。"""
    out = []
    for i in range(steps):
        t = i / (steps - 1)
        L = light[0] + (light[1] - light[0]) * t
        C = chroma * (0.45 + 0.55 * math.sin(math.pi * (0.15 + 0.7 * t)))   # 中間ほど鮮やか
        out.append(oklch_to_hex(L, C, h_deg))
    return out


def blend(c1, c2, steps=5):
    """2色グラデーション：OKLab で直線的に補間（途中が濁りにくい）。"""
    a, b = hex_to_oklab(c1), hex_to_oklab(c2)
    return [oklab_to_hex(a + (b - a) * i / (steps - 1)) for i in range(steps)]


def color_distance(c1, c2):
    """OKLab での色の差（0.05 未満だと隣り合っても見分けにくい）。"""
    return float(np.linalg.norm(hex_to_oklab(c1) - hex_to_oklab(c2)))


def _p(cat, name, colors, line=None, bg=None):
    d = {"category": cat, "name": name, "colors": [c.upper() for c in colors]}
    if line:
        d["line"] = line
    if bg:
        d["bg"] = bg
    return d


def _build_palettes():
    P = []
    # --- カラフル
    for name, cols in [
        ("ポップ", DEFAULT_COLORS),
        ("レインボー", ["#E63946", "#F4A261", "#F9D923", "#52B788", "#3A86FF", "#8338EC"]),
        ("サーカス", ["#E63946", "#F1FAEE", "#FFD166", "#118AB2", "#073B4C"]),
        ("おもちゃ箱", ["#FF595E", "#FFCA3A", "#8AC926", "#1982C4", "#6A4C93"]),
        ("トロピカル", ["#F72585", "#FFBE0B", "#06D6A0", "#118AB2", "#FB5607"]),
        ("キャンディ", ["#FF70A6", "#FF9770", "#FFD670", "#E9FF70", "#70D6FF"]),
        ("マルシェ", ["#D62828", "#F77F00", "#FCBF49", "#EAE2B7", "#003049"]),
        ("バウハウス", ["#D7263D", "#F4D35E", "#1B998B", "#2E294E", "#F2E9E4"]),
        ("モンドリアン", ["#D40920", "#1356A2", "#F7D842", "#F2F2F2", "#222222"]),
        ("フィエスタ", ["#E4572E", "#F3A712", "#29335C", "#A8C686", "#DB2B39"]),
        ("ジェラート", ["#F94144", "#F8961E", "#F9C74F", "#90BE6D", "#43AA8B", "#577590"]),
    ]:
        P.append(_p("カラフル", name, cols))
    # --- パステル
    for name, cols in [
        ("パステル虹", ["#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF", "#A0C4FF", "#BDB2FF"]),
        ("マカロン", ["#F7C5CC", "#FBE3B9", "#C9E4CA", "#B5D8EB", "#D9C8F0"]),
        ("綿あめ", ["#FFC8DD", "#FFAFCC", "#BDE0FE", "#A2D2FF", "#CDB4DB"]),
        ("ミント＆ピーチ", ["#B8F2E6", "#FFA69E", "#FAF3DD", "#AED9E0", "#5E6472"]),
        ("ラベンダー畑", ["#EBD8FD", "#C6A3E8", "#A6B1FF", "#F5ECF6", "#7E6BB0"]),
        ("ベビー", ["#FDE2E4", "#F6BBD0", "#D5EFE6", "#A9D6E5", "#D2D9FA"]),
        ("朝もや", ["#FCF3EE", "#F8CDBF", "#EC9E8B", "#BFD8CB", "#B9A88F"]),
    ]:
        P.append(_p("パステル", name, cols))
    # --- ビビッド・ネオン（暗い背景と組み合わせるものは線・背景も指定）
    P += [
        _p("ビビッド・ネオン", "ネオン", ["#FF00E6", "#00F5FF", "#B4FF00", "#FFE600", "#7A00FF"],
           line="#0B0B12", bg="#0B0B12"),
        _p("ビビッド・ネオン", "サイバー", ["#F72585", "#7209B7", "#3A0CA3", "#4361EE", "#4CC9F0"]),
        _p("ビビッド・ネオン", "ポップアート", ["#FF1E56", "#FFAC41", "#FFF200", "#00A6ED", "#111111"]),
        _p("ビビッド・ネオン", "蛍光ペン", ["#FFFF3F", "#FF6EC7", "#7DF9FF", "#9DFF57", "#FFA62B"]),
        _p("ビビッド・ネオン", "夜の街", ["#FF2A6D", "#05D9E8", "#D1F7FF", "#7A04EB", "#FE7A15"],
           line="#01012B", bg="#01012B"),
    ]
    # --- 和の色
    for name, cols in [
        ("江戸", ["#6C5B9E", "#B7282E", "#F8B500", "#316745", "#165E83"]),     # 江戸紫・茜・山吹・千歳緑・藍
        ("和モダン", ["#C53D43", "#E6B422", "#2B4B65", "#EDE4CD", "#5B7E91"]),
        ("藍染", ["#EAF4FC", "#A0D8EF", "#5B8FB9", "#1E50A2", "#0F2350"]),
        ("桜", ["#FDF2F4", "#F6C1C6", "#E98FA8", "#B7A3C9", "#8C5E7E"]),
        ("抹茶", ["#F3EFD9", "#C5C56A", "#88964B", "#5B6F3A", "#3A4A2B"]),
        ("紅葉", ["#B7282E", "#E2041B", "#EE7800", "#F8B862", "#5B4B32"]),
        ("祭", ["#D9333F", "#FFF4E0", "#F39800", "#1E2A5A", "#008899"]),
    ]:
        P.append(_p("和の色", name, cols))
    # --- 季節・自然
    for name, cols in [
        ("春", ["#F7D6E0", "#F2B5D4", "#B9E3C6", "#FFF4B8", "#C3B1E1"]),
        ("夏", ["#00A6FB", "#FFD60A", "#FF5D8F", "#0582CA", "#F8F9FA"]),
        ("秋", ["#9C2C13", "#D35400", "#E9A23B", "#6B4226", "#B5651D"]),
        ("冬", ["#F1F5F9", "#BCD4E6", "#7FA7C9", "#3E6A8A", "#1B2F44"]),
        ("森", ["#DAD7CD", "#A3B18A", "#588157", "#3A5A40", "#1F2E26"]),
        ("海", ["#CAF0F8", "#90E0EF", "#00B4D8", "#0077B6", "#03045E"]),
        ("砂漠", ["#F4E1C1", "#E2B884", "#C68B59", "#9A5B3C", "#5E3A2E"]),
        ("夕焼け", ["#F9C784", "#FC7A1E", "#E24E1B", "#8C2F39", "#3D1E4F"]),
    ]:
        P.append(_p("季節・自然", name, cols))
    # --- アース・北欧
    for name, cols in [
        ("テラコッタ", ["#E07A5F", "#F2CC8F", "#81B29A", "#3D405B", "#F4F1DE"]),
        ("北欧", ["#F2E8CF", "#A7C957", "#6A994E", "#386641", "#BC4749"]),
        ("ボタニカル", ["#E9EDC9", "#B5C49A", "#FEFAE0", "#F2D5B0", "#C08552"]),
        ("カフェ", ["#EDE0D4", "#E6CCB2", "#DDB892", "#B08968", "#7F5539"]),
        ("リネン", ["#F5EBE0", "#E3D5CA", "#D5BDAF", "#A68A64", "#656D4A"]),
        ("ノルディック", ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]),
    ]:
        P.append(_p("アース・北欧", name, cols))
    # --- レトロ
    for name, cols in [
        ("70年代", ["#F4A259", "#F4E285", "#BC4B51", "#5B8E7D", "#8CB369"]),
        ("ミッドセンチュリー", ["#E8A33D", "#D9572B", "#2E8B8B", "#F2E3BC", "#3B3B3B"]),
        ("レトロポップ", ["#EF476F", "#FFD166", "#06D6A0", "#118AB2", "#073B4C"]),
        ("ヴィンテージポスター", ["#E2C290", "#C1502E", "#3E6259", "#F2E9D8", "#2C2A29"]),
        ("ディープ", ["#0B132B", "#1C2541", "#3A506B", "#5BC0BE", "#E0FBFC"]),
    ]:
        P.append(_p("レトロ", name, cols))
    # --- 単色グラデーション（同じ色相で明るい→暗い）
    for name, hue, ch in [("赤", 25, 0.17), ("朱・橙", 50, 0.15), ("黄", 95, 0.15), ("黄緑", 130, 0.15),
                          ("緑", 150, 0.13), ("青緑", 185, 0.11), ("水色", 220, 0.11), ("青", 255, 0.15),
                          ("藍", 270, 0.13), ("紫", 305, 0.15), ("ピンク", 350, 0.15), ("茶", 55, 0.08)]:
        P.append(_p("単色グラデーション", name, ramp(hue, ch)))
    # --- 2色グラデーション
    for name, a, b in [("夕暮れ", "#2E1F5E", "#FFB347"), ("海の底", "#0B1D51", "#7FE7DC"),
                       ("若葉", "#1B4332", "#D8F3DC"), ("炎", "#6A040F", "#FFD60A"),
                       ("氷", "#F0F8FF", "#1D3557"), ("ベリー", "#3C1642", "#FF8FAB"),
                       ("柑橘", "#2D6A4F", "#FFE066"), ("夜明け", "#22223B", "#F2CC8F"),
                       ("ぶどう", "#240046", "#E0AAFF"), ("ピーチ", "#FFF1E6", "#C9184A")]:
        P.append(_p("2色グラデーション", name, blend(a, b)))
    # --- 白黒・グレー
    P += [
        _p("白黒・グレー", "グレースケール", blend("#FFFFFF", "#111111")),
        _p("白黒・グレー", "明るいグレー", blend("#FFFFFF", "#8A8A8A")),
        _p("白黒・グレー", "チャコール", blend("#9A9A9A", "#1A1A1A")),
        _p("白黒・グレー", "墨絵", ["#F4F1EA", "#CFC8BB", "#9C9488", "#5E574F", "#2B2724"]),
        _p("白黒・グレー", "セピア", ["#F3E9D2", "#D9C3A0", "#B08F6A", "#7A5C42", "#45342A"]),
        _p("白黒・グレー", "白黒＋赤", ["#FFFFFF", "#C8C8C8", "#6E6E6E", "#1E1E1E", "#D62828"]),
        _p("白黒・グレー", "白黒＋黄", ["#FFFFFF", "#C8C8C8", "#6E6E6E", "#1E1E1E", "#FFD60A"]),
        _p("白黒・グレー", "白黒＋青", ["#FFFFFF", "#C8C8C8", "#6E6E6E", "#1E1E1E", "#1E6FD9"]),
        _p("白黒・グレー", "反転（黒地に白線）", ["#2B2B2B", "#444444", "#5E5E5E", "#7A7A7A", "#1A1A1A"],
           line="#F5F5F5", bg="#000000"),
        _p("白黒・グレー", "ブルーグレー", ["#F1F4F8", "#C9D3DF", "#8D9CB0", "#55657A", "#2B3645"]),
    ]
    return P


PALETTES = _build_palettes()


HARMONY_MODES = ["おまかせ", "類似色", "補色", "三色", "単色", "パステル", "ビビッド", "くすみ"]


def harmony_palette(rng, mode="おまかせ", n=5):
    """色相環のルールに沿った配色をランダムに作る（OKLCH で明るさの差も確保）。"""
    if mode == "おまかせ":
        mode = rng.choice(HARMONY_MODES[1:])
    best = None
    for _ in range(200):
        h0 = rng.uniform(0, 360)
        if mode == "単色":
            cols = ramp(h0, rng.uniform(0.08, 0.16), n)
        else:
            if mode == "類似色":
                hues = [h0 + rng.uniform(-40, 40) for _ in range(n)]
            elif mode == "補色":
                hues = [h0 + (180 if i % 2 else 0) + rng.uniform(-15, 15) for i in range(n)]
            elif mode == "三色":
                hues = [h0 + 120 * (i % 3) + rng.uniform(-12, 12) for i in range(n)]
            else:
                hues = [rng.uniform(0, 360) for _ in range(n)]
            if mode == "パステル":
                Ls, C = [rng.uniform(0.84, 0.95) for _ in range(n)], 0.07
            elif mode == "ビビッド":
                Ls, C = [rng.uniform(0.55, 0.85) for _ in range(n)], 0.2
            elif mode == "くすみ":
                Ls, C = [rng.uniform(0.45, 0.85) for _ in range(n)], 0.05
            else:
                Ls = sorted([rng.uniform(0.35, 0.95) for _ in range(n)], reverse=True)
                C = rng.uniform(0.09, 0.17)
            cols = [oklch_to_hex(L, C * rng.uniform(0.7, 1.2), h % 360) for L, h in zip(Ls, hues)]
        d = min(color_distance(a, b) for i, a in enumerate(cols) for b in cols[i + 1:])
        if d >= 0.07:
            return cols, mode
        if best is None or d > best[0]:
            best = (d, cols)
    return best[1], mode                    # 条件を満たすものがなければ、最も見分けやすいもの


def palette_from_pixels(rgb, n=5, seed=0, vivid=True):
    """画素の色（N×3、0〜255）から代表色を n 色取り出す。

    OKLab（人の見た目に近い色空間）で k-means を行い、
      - よく使われている色ほど（vivid=True なら鮮やかな色ほど）優先し、
      - すでに選んだ色から離れている色ほど優先する（似た色ばかりにならない）
    という順に選ぶ。結果は明るい順。
    """
    rgb = np.asarray(rgb, float).reshape(-1, 3) / 255.0
    if len(rgb) == 0:
        raise ValueError("画像に色が見つかりませんでした。")
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lab = np.cbrt(_srgb_to_lin(rgb) @ M1.T) @ M2.T
    rs = np.random.default_rng(seed)
    if len(lab) > 20000:
        lab = lab[rs.choice(len(lab), 20000, replace=False)]
    k = min(len(lab), n * 3)                    # 多めに分けて、少ないが鮮やかな色も残す
    C = [lab[rs.integers(len(lab))]]            # k-means++ で初期値
    for _ in range(1, k):
        d = np.min(((lab[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        if d.sum() <= 0:
            break
        C.append(lab[rs.choice(len(lab), p=d / d.sum())])
    C = np.array(C)
    for _ in range(25):
        lbl = np.argmin(((lab[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        C2 = np.array([lab[lbl == i].mean(0) if np.any(lbl == i) else C[i] for i in range(len(C))])
        if np.allclose(C2, C, atol=1e-5):
            break
        C = C2
    counts = np.bincount(lbl, minlength=len(C)).astype(float)
    score = counts / counts.sum()
    if vivid:
        # 背景の白や線の黒は面積が大きくなりがちなので、面積の差を縮めたうえで鮮やかな色ほど優先する
        # （白黒の画像なら全部の色が同じだけ割り引かれるので、面積の順になる）
        score = np.sqrt(score) * (0.04 + np.minimum(1.0, np.hypot(C[:, 1], C[:, 2]) / 0.08))
    chosen = [int(np.argmax(score))]
    min_d = 0.05                                # これより近い色は、隣り合うと見分けにくいので避ける
    while len(chosen) < min(n, len(C)):
        d = np.min(np.linalg.norm(C[:, None, :] - C[chosen][None], axis=-1), axis=1)
        gain = np.where(d >= min_d, score * np.minimum(d, 0.3), -1.0)
        gain[chosen] = -1
        if gain.max() <= 0:
            if min_d < 0.01:
                break
            min_d *= 0.6                          # 色の少ない画像では条件をゆるめる
            continue
        chosen.append(int(np.argmax(gain)))
    cols = sorted((C[i] for i in chosen), key=lambda c: -c[0])
    return [oklab_to_hex(c) for c in cols]


def flat_pixels(img):
    """画像（H×W×3 または ×4）から、色の境目（にじんだ画素）と透明な部分を除いた画素を返す。"""
    a = np.asarray(img)
    rgb = a[..., :3].astype(float)
    ok = np.ones(rgb.shape[:2], bool)
    if a.shape[-1] == 4:
        ok &= a[..., 3] > 127
    if min(rgb.shape[:2]) >= 3:
        d = np.zeros(rgb.shape[:2])
        d[1:, :] = np.maximum(d[1:, :], np.abs(rgb[1:] - rgb[:-1]).max(-1))
        d[:-1, :] = np.maximum(d[:-1, :], np.abs(rgb[1:] - rgb[:-1]).max(-1))
        d[:, 1:] = np.maximum(d[:, 1:], np.abs(rgb[:, 1:] - rgb[:, :-1]).max(-1))
        d[:, :-1] = np.maximum(d[:, :-1], np.abs(rgb[:, 1:] - rgb[:, :-1]).max(-1))
        flat = ok & (d < 24)
        if flat.sum() >= 0.2 * ok.sum():         # 写真などで平らな部分が少なければ全体を使う
            ok = flat
    return rgb[ok]


def palette_from_array(img, n=5, seed=0, vivid=True):
    """画像の配列（H×W×3 または ×4、0〜255）から配色を作る。"""
    return palette_from_pixels(flat_pixels(img), n, seed, vivid=vivid)


def palette_from_image(path, n=5, seed=0, vivid=True):
    """画像ファイルから配色を作る（Pillow が必要）。"""
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    im.thumbnail((160, 160))
    return palette_from_array(np.asarray(im), n, seed, vivid=vivid)


# おすすめ設定（GUI の設定と同じ形式。seed は含めない）
STYLE_PRESETS = [
    {"name": "標準", "desc": "迷ったらこれ", "settings": {}},
    {"name": "繊細", "desc": "細い線で長く、細かな区画",
     "settings": {"length": 26, "stroke": 0.2, "min_radius": 2, "smooth": 2, "wiggle": 1.4, "pressure": 0.8}},
    {"name": "ゆったり", "desc": "大きな区画とゆるやかな曲線",
     "settings": {"length": 7, "wiggle": 0.25, "min_radius": 3, "smooth": 2, "att_min": 0, "att_max": 0,
                  "avoid": 0.4, "stroke": 1.6, "palette": "テラコッタ"}},
    {"name": "ステンドグラス", "desc": "太い黒線と鮮やかな色",
     "settings": {"length": 16, "stroke": 2.6, "pressure": 0.2, "min_radius": 4, "max_bends": 4,
                  "palette": "レインボー"}},
    {"name": "絡み合い", "desc": "交差までの曲がりを3回に制限",
     "settings": {"length": 14, "max_bends": 3, "pull": 0.7, "pressure": 0.6, "palette": "ノルディック"}},
    {"name": "中心に集まる", "desc": "重心1つへ強く集める",
     "settings": {"length": 16, "att_min": 1, "att_max": 1, "pull": 0.9, "avoid": 0.4, "palette": "夕焼け"}},
    {"name": "散らばり", "desc": "重心3つ・重なりを強く避ける",
     "settings": {"length": 16, "att_min": 3, "att_max": 3, "pull": 0.3, "avoid": 0.9, "wiggle": 1.8,
                  "palette": "おもちゃ箱"}},
    {"name": "塗りだけ", "desc": "線なし。色面だけの構成",
     "settings": {"length": 20, "stroke": 0, "min_radius": 2, "smooth": 2, "palette": "パステル虹"}},
    {"name": "版画", "desc": "白黒と強い線の強弱",
     "settings": {"length": 18, "stroke": 1.8, "pressure": 0.9, "smooth": 1.6, "palette": "墨絵"}},
    {"name": "ネオン", "desc": "黒地に蛍光色",
     "settings": {"length": 16, "stroke": 1.0, "pressure": 0.4, "palette": "ネオン"}},
]


def style_settings(name):
    """おすすめ設定を、配色の色・線・背景まで展開した設定にして返す。"""
    for st in STYLE_PRESETS:
        if st["name"] == name:
            d = dict(st["settings"])
            pal = find_palette(d.get("palette", "ポップ"))
            d["palette"] = pal["name"]
            d["colors"] = list(pal["colors"])
            d["line_color"] = pal.get("line", "#111111")
            d["bg"] = pal.get("bg", "#FFFFFF")
            return d
    return None


def find_palette(name):
    for p in PALETTES:
        if p["name"] == name or f'{p["category"]}/{p["name"]}' == name:
            return p
    return None


# ================================================================ ベジェ曲線の基本
def closed_catmull_rom(P):
    """閉じた Catmull-Rom スプラインを 3次ベジェ区間の配列 (n, 4, 2) に変換。"""
    P = np.asarray(P, float)
    pm1, p0, p1, p2 = np.roll(P, 1, 0), P, np.roll(P, -1, 0), np.roll(P, -2, 0)
    return np.stack([p0, p0 + (p1 - pm1) / 6.0, p1 - (p2 - p0) / 6.0, p1], 1)


def sample_beziers(segs, per_seg=8):
    """各ベジェ区間を per_seg 点ずつ（終点を含まず）サンプリングして折れ線にする。"""
    S = np.asarray(segs, float)
    t = np.linspace(0, 1, per_seg, endpoint=False)[None, :, None]
    a, b, c, d = (S[:, i][:, None, :] for i in range(4))
    pts = ((1 - t) ** 3) * a + 3 * ((1 - t) ** 2) * t * b + 3 * (1 - t) * t * t * c + t ** 3 * d
    return pts.reshape(-1, 2)


def min_curvature_radius(segs, per_seg=24):
    """ベジェ曲線全体での最小曲率半径（px）。"""
    S = np.asarray(segs, dtype=float)
    a, b, c, d = (S[:, i][:, None, :] for i in range(4))
    t = np.linspace(0, 1, per_seg)[None, :, None]
    d1 = 3 * ((1 - t) ** 2 * (b - a) + 2 * (1 - t) * t * (c - b) + t ** 2 * (d - c))
    d2 = 6 * ((1 - t) * (c - 2 * b + a) + t * (d - 2 * c + b))
    cross = np.abs(d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0])
    speed = np.hypot(d1[..., 0], d1[..., 1])
    rad = np.where(cross > 1e-12, speed ** 3 / np.maximum(cross, 1e-12), np.inf)
    return float(rad.min())


def segment_curvature(segs):
    """各ベジェ区間の中央での曲率（符号なし）。"""
    S = np.asarray(segs, float)
    a, b, c, d = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    t = 0.5
    d1 = 3 * ((1 - t) ** 2 * (b - a) + 2 * (1 - t) * t * (c - b) + t ** 2 * (d - c))
    d2 = 6 * ((1 - t) * (c - 2 * b + a) + t * (d - 2 * c + b))
    cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    speed = np.maximum(np.hypot(d1[:, 0], d1[:, 1]), 1e-9)
    return cross / speed ** 3


def bez_split(seg, t):
    """de Casteljau 法で t の位置で2つに分割。"""
    p0, p1, p2, p3 = seg
    a = p0 + (p1 - p0) * t
    b = p1 + (p2 - p1) * t
    c = p2 + (p3 - p2) * t
    d = a + (b - a) * t
    e = b + (c - b) * t
    f = d + (e - d) * t
    return np.array([p0, a, d, f]), np.array([f, e, c, p3])


def bez_sub(seg, t0, t1):
    """ベジェ区間の [t0, t1] 部分を取り出す。"""
    if t1 < 1.0 - 1e-12:
        seg, _ = bez_split(seg, t1)
        t0 = t0 / t1
    if t0 > 1e-12:
        _, seg = bez_split(seg, t0)
    return seg


def _fmt(p):
    return f"{p[0]:.2f},{p[1]:.2f}"


def pieces_to_path(pieces):
    """連続するベジェ片のリストを、閉じた SVG サブパスにする。"""
    d = [f"M{_fmt(pieces[0][0])}"]
    for _, b, c, e in pieces:
        d.append(f"C{_fmt(b)} {_fmt(c)} {_fmt(e)}")
    return " ".join(d) + " Z"


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ================================================================ 交差判定（生成中に使う）
def _seg_cross(a, b, c, d):
    """線分 ab と cd が交差するか（端点の接触は含まない）。"""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    d1, d2 = orient(c, d, a), orient(c, d, b)
    d3, d4 = orient(a, b, c), orient(a, b, d)
    return d1 * d2 < 0 and d3 * d4 < 0


class _SegmentGrid:
    """自分の線との交差を高速に調べるための格子。末尾からの削除（やり直し）に対応。"""

    MIN_SIN = math.sin(math.radians(20))      # これより浅い（かすめる）交差は数えない

    def __init__(self, cell, skip_recent=3):
        self.cell = cell
        self.skip_recent = skip_recent
        self.grid = {}
        self.cells_of = []
        self.segs = []

    def _cells(self, a, b):
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        c = self.cell
        return [(gx, gy) for gx in range(int(x0 // c), int(x1 // c) + 1)
                for gy in range(int(y0 // c), int(y1 // c) + 1)]

    def add(self, a, b):
        """線分を追加し、それより前の線分と交差したら True を返す。"""
        i = len(self.segs)
        u = (b[0] - a[0], b[1] - a[1])
        lu = math.hypot(*u) or 1.0
        cells = self._cells(a, b)
        hit = False
        self.last_hit = -1
        for key in cells:
            for j in self.grid.get(key, ()):
                if j < i - self.skip_recent and _seg_cross(a, b, *self.segs[j]):
                    c, d = self.segs[j]
                    v = (d[0] - c[0], d[1] - c[1])
                    sin = abs(u[0] * v[1] - u[1] * v[0]) / (lu * (math.hypot(*v) or 1.0))
                    if sin >= self.MIN_SIN:
                        hit = True
                        self.last_hit = j
                        break
            if hit:
                break
        for key in cells:
            self.grid.setdefault(key, []).append(i)
        self.cells_of.append(cells)
        self.segs.append((a, b))
        return hit

    def truncate(self, n):
        """線分を先頭から n 本だけ残して削除する。"""
        for i in range(len(self.segs) - 1, n - 1, -1):
            for key in self.cells_of[i]:
                self.grid[key].pop()
        del self.cells_of[n:]
        del self.segs[n:]


def recent_window(R, L):
    """何歩前までの線との交差を「自分の小さなループ」として数えないか。"""
    return max(3, int(2 * math.pi * 2.5 * R / L))


def _bends_ok(P, bend_starts, n_steps, max_bends, skip_recent):
    """完成した線で、交差と交差の間の曲がり回数が上限以内かを確かめる。"""
    L = float(np.linalg.norm(P[1] - P[0]))
    grid = _SegmentGrid(max(4 * L, 1.0), skip_recent=skip_recent)
    n_steps = min(n_steps, len(P) - 1)
    crosses = [i for i in range(n_steps) if grid.add(P[i], P[i + 1])]
    starts = np.array(bend_starts)
    bounds = [-1] + crosses + [n_steps - 1]
    for j in range(len(bounds) - 1):
        cnt = int(((starts >= bounds[j] + 1) & (starts <= bounds[j + 1])).sum())
        if cnt > (max(max_bends, 3) if j == 0 else max_bends):
            return False
    return True


# ================================================================ 重心と線の密度
class _Field:
    """線がどこをどれだけ通ったか（密度）と、重心の「満たされ具合」を管理する。

    - 描いた線の長さを粗い格子に積算する（直前に描いた部分は少し遅らせて積算し、
      いま曲がっている自分のカーブを「混んでいる」と誤認しないようにする）
    - 重心ごとに、半径内に引かれた線の長さ（負荷）を数え、容量に近づくほど引力を弱める
    - やり直しのときは、取り消した歩で積算した分を差し引く
    - 問い合わせは1歩に何度もあるので、「周囲3×3マスの合計」を積算のたびに更新しておき、
      問い合わせは1回の参照で済ませる（Python のリストで持つと小さな読み書きが速い）
    """

    def __init__(self, W, H, R, L, short, centers, spacing):
        self.cs = max(R, short / 40.0)
        self.gw, self.gh = int(W // self.cs) + 2, int(H // self.cs) + 2
        n = self.gw * self.gh
        # 周囲3×3マスの合計：線の長さ、線の向き（180°で同一視するため 2θ で記録）
        self.D3 = [0.0] * n
        self.C3 = [0.0] * n
        self.S3 = [0.0] * n
        self.L = L
        self.lag = max(4, int(2 * math.pi * 1.2 * R / L))
        self.log = []                        # (積算した歩, セル番号, cos成分, sin成分)
        self.rho = 0.2 * short               # 重心の影響範囲
        self.centers = [np.array(c, float) for c in centers]
        self.cap = math.pi * self.rho ** 2 / spacing   # 重心まわりに引ける線の長さの目安
        self.load = [0.0] * len(self.centers)
        # セルごとに、その周囲3×3のセル番号と、含まれる重心の番号を前もって求めておく
        self.nb = []
        self.att_of = []
        for cx in range(self.gw):
            for cy in range(self.gh):
                self.nb.append([x * self.gh + y for x in range(max(cx - 1, 0), min(cx + 2, self.gw))
                                for y in range(max(cy - 1, 0), min(cy + 2, self.gh))])
                px, py = (cx + 0.5) * self.cs, (cy + 0.5) * self.cs
                self.att_of.append([i for i, c in enumerate(self.centers)
                                    if math.hypot(px - c[0], py - c[1]) <= self.rho])

    def cell(self, p):
        cx = int(p[0] // self.cs)
        cy = int(p[1] // self.cs)
        cx = 0 if cx < 0 else (self.gw - 1 if cx >= self.gw else cx)
        cy = 0 if cy < 0 else (self.gh - 1 if cy >= self.gh else cy)
        return cx * self.gh + cy

    def _add(self, c, ca, sa, sign):
        L = self.L * sign
        D3, C3, S3 = self.D3, self.C3, self.S3
        ca *= sign
        sa *= sign
        for k in self.nb[c]:
            D3[k] += L
            C3[k] += ca
            S3[k] += sa
        for i in self.att_of[c]:
            self.load[i] += L

    def deposit(self, step, pts):
        """step 歩目に、lag 歩前の点（とその向き）を積算する。"""
        j = step - self.lag
        if j < 1:
            return
        p, q = pts[j], pts[j - 1]
        c = self.cell(p)
        a = 2 * math.atan2(p[1] - q[1], p[0] - q[0])
        ca, sa = self.L * math.cos(a), self.L * math.sin(a)
        self._add(c, ca, sa, 1.0)
        self.log.append((step, c, ca, sa))

    def rollback(self, n_keep):
        while self.log and self.log[-1][0] >= n_keep:
            _, c, ca, sa = self.log.pop()
            self._add(c, ca, sa, -1.0)

    def at(self, p, W, H, margin):
        """点 p まわり（3×3セル）の線の密度。画面の外は混んでいる扱い。"""
        if p[0] < margin or p[1] < margin or p[0] > W - margin or p[1] > H - margin:
            return 3.0 * self.cs
        return self.D3[self.cell(p)]

    def parallel_at(self, p, c2t, s2t, W, H, margin):
        """点 p まわりで、向き θ（cos2θ, sin2θ で渡す）と「平行に走っている」線の量。
        直交する線はほぼ数えない。"""
        if p[0] < margin or p[1] < margin or p[0] > W - margin or p[1] > H - margin:
            return 3.0 * self.cs
        c = self.cell(p)
        d = self.D3[c]
        par = 0.5 * (d + self.C3[c] * c2t + self.S3[c] * s2t)
        return 0.1 * d + 0.9 * par

    def freshness(self, i):
        """重心 i の空き具合 0〜1（1=まだ線が少ない）。"""
        return max(0.0, 1.0 - self.load[i] / self.cap)


# ================================================================ 線の生成
def _solve3(A, b):
    """3×3 の連立方程式をクラメルの公式で解く（小さい行列は numpy より速い）。"""
    (a, b_, c), (d, e, f), (g, h, k) = A
    det = a * (e * k - f * h) - b_ * (d * k - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-18:
        return None
    x0 = (b[0] * (e * k - f * h) - b_ * (b[1] * k - f * b[2]) + c * (b[1] * h - e * b[2])) / det
    x1 = (a * (b[1] * k - f * b[2]) - b[0] * (d * k - f * g) + c * (d * b[2] - b[1] * g)) / det
    x2 = (a * (e * b[2] - b[1] * h) - b_ * (d * b[2] - b[1] * g) + b[0] * (d * h - e * g)) / det
    return x0, x1, x2


def _connector(p0, th0, turn0, p1, th1, turn1, L, n_steps, iters=30):
    """(p0, 向き th0, 直前の曲がり turn0) から (p1, 到着向き th1, 直後の曲がり turn1) へ、
    n_steps 歩でつなぐ曲がりの列を求める。

    曲がりの強さは turn0 → turn1 へ直線的に変化させ、そこに両端で0になる
    sin(πs), sin(2πs), sin(3πs) を足して、終点の位置（x, y）と向きを合わせる。
    両端の曲がりの強さが前後とつながるので、つなぎ目で曲率が飛ばない。
    解き方はレーベンバーグ・マーカート法（微分は解析的に求め、1歩の大きさを自動調整して発散を防ぐ）。
    """
    s = np.arange(1, n_steps + 1) / (n_steps + 1)
    base = turn0 * (1 - s) + turn1 * s
    B = np.stack([np.sin(np.pi * s), np.sin(2 * np.pi * s), np.sin(3 * np.pi * s)])
    CB = np.cumsum(B, axis=1)                  # 各補正が、各歩の向きをどれだけ変えるか
    h_base = th0 + np.cumsum(base)
    target = th1 + 2 * math.pi * round((h_base[-1] - th1) / (2 * math.pi))
    scale = L * n_steps / (2 * math.pi)
    cb_end = CB[:, -1]

    # 初期値：向きの誤差だけを sin(πs) の項で打ち消しておく（反復回数が減る）
    c = np.array([(target - h_base[-1]) / cb_end[0] if abs(cb_end[0]) > 1e-12 else 0.0, 0.0, 0.0])

    def resid(c):
        h = h_base + c @ CB
        ch, sh = np.cos(h), np.sin(h)
        return (p0[0] + L * ch.sum() - p1[0], p0[1] + L * sh.sum() - p1[1],
                (h[-1] - target) * scale), ch, sh

    e, ch, sh = resid(c)
    cost = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
    lam = 1e-3
    for _ in range(iters):
        if max(abs(e[0]), abs(e[1]), abs(e[2])) < 0.15:
            break
        J = np.stack([-L * (CB @ sh), L * (CB @ ch), cb_end * scale])   # 3×3
        A = J.T @ J
        g = J.T @ np.asarray(e)
        dA = np.diag(A) + 1e-12
        improved = False
        for _ in range(10):
            M = A + lam * np.diag(dA)
            sol = _solve3(M.tolist(), (-g).tolist())
            if sol is None:
                return None
            c2 = c + np.asarray(sol)
            e2, ch2, sh2 = resid(c2)
            cost2 = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]
            if cost2 < cost:
                c, e, ch, sh, cost = c2, e2, ch2, sh2, cost2
                lam = max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 4
        if not improved:
            break
    return base + c @ B if max(abs(e[0]), abs(e[1]), abs(e[2])) < 0.5 else None


def _try_close(p, th, turn_prev, early_P, early_H, early_T, dj, L, R, min_radius, W, H, margin):
    """描き始めの区間の近い点へ、つなぎの曲線で合流できるか試す。

    候補の合流点（近い順に数点）× つなぎの長さ（数通り）を試し、
    曲がりがいちばんゆるやかで、画面からはみ出さない解を選ぶ。"""
    p = np.asarray(p, float)
    lim = L / (min_radius / 0.8)            # スプライン化での曲率の強まりを見込んだ上限
    order = np.argsort(dj)[:4] + 1
    best, best_cost = None, None
    for j in order:
        p1 = early_P[j]
        d = float(np.linalg.norm(p1 - p))
        dth = abs(_wrap(early_H[j - 1] - th))
        for f in (1.6, 1.2, 2.3):
            length = f * d + R * dth
            ns = max(6, int(round(length / L)))
            if ns * L < 1.02 * d:                # 長さが足りず、そもそも届かない
                continue
            # 曲がれる量が足りないものは計算せずに除く（両端で曲がりを弱めるぶん、上限の6割で見積もる）
            alpha = abs(_wrap(math.atan2(p1[1] - p[1], p1[0] - p[0]) - th))
            if max(2 * alpha, dth) > 0.6 * lim * ns + 0.3:
                continue
            u = _connector(p, th, turn_prev, p1, early_H[j - 1], early_T[j], L, ns)
            if u is None:
                continue
            mx = float(np.abs(u).max())
            if mx > lim:
                continue
            # つなぎ目の両端で曲がりの強さが前後とつながっていること
            if abs(u[0] - turn_prev) > 0.25 * lim or abs(u[-1] - early_T[j]) > 0.25 * lim:
                continue
            h = th + np.cumsum(u)
            q = p + L * np.cumsum(np.stack([np.cos(h), np.sin(h)], 1), 0)
            if q[:, 0].min() < margin * 0.5 or q[:, 1].min() < margin * 0.5 or \
                    q[:, 0].max() > W - margin * 0.5 or q[:, 1].max() > H - margin * 0.5:
                continue
            cost = mx / lim + 0.1 * ns * L / max(d, L)
            if best is None or cost < best_cost:
                best, best_cost = (int(j), u), cost
            if mx <= 0.6 * lim:                  # 十分ゆるやか：これ以上探さず採用
                return best
    return best


# 重心のまわりに引ける線の量の目安：「線どうしの間隔が旋回半径の何倍になるまで」詰めてよいか
ATTRACTOR_SPACING = 1.0
PULL_GAIN = 1.0          # 重心へ向かう曲がりの強さ（最大曲率に対する比）


class Cancelled(Exception):
    """生成が中断された（GUI の「中断」ボタンなど）。"""


def _check_stop(should_stop):
    if should_stop is not None and should_stop():
        raise Cancelled("中断しました")


MAX_ROLLBACKS = 4000     # 1本の線で、これ以上やり直したら見切って最初から作り直す
KNOT_TURNS = 1.3          # 最小半径の輪 1.6 周ぶんの長さで、これ以上回ったら「結び目」とみなす


def walk_stroke(rng, W, H, margin, min_radius, length, wiggle, max_bends=0, stats=None,
                attractors=(), pull=0.5, avoid=0.6, smooth=1.0, should_stop=None, radius_margin=0.72):
    """一筆書きの閉曲線を「曲がりながら進むペン先」の軌跡として作る。

    - 線は「曲がり」（同じ向き・同じ強さで曲がり続ける弧）をつないで描く
    - 1歩ごとの曲がりの強さを制限して、曲率半径が min_radius を下回らないようにする
    - 曲がりの強さは少しずつ変わる（smooth が大きいほど入り抜けが長くなめらか）
    - 重心（attractors）があれば、そこへ緩く向かう。重心のまわりが線で埋まってくると
      引力が弱まり、別の重心や、重心近くのまだ空いている場所へ目標が移る
    - 進む先の左右の線の密度を見て、空いている方へ曲がる（同じ場所を何度も通らない）
    - max_bends > 0 のとき: 自分の線と交差するたびにそこを起点として記録し、
      交差しないまま max_bends 回を超えて曲がろうとしたら、起点まで戻ってやり直す
      （出発直後の最初の交差だけは、最低3回まで曲がってよい）
    - 画面の端に近づいたら中央へ曲がる（これも1回の曲がりとして数える）
    - 最後は出発点へ戻り、向きもそろえて閉じる
    """
    short = min(W, H)
    # スプライン化で曲率が少し強まるので、余裕を持たせた旋回半径で歩く
    R = min_radius / radius_margin
    L = max(1.0, min(0.3 * R, 0.02 * short))   # 1歩の長さ
    kmax = 1.0 / R                             # 最大曲率
    slew = kmax / (6.0 * max(smooth, 0.2))     # 1歩で変えられる曲がりの強さ
    n_target = int(length * short / L)
    n_main = int(n_target * 0.75)              # ここまでが自由に歩く区間、残りは帰り道
    rmax = max(R * 1.5, 0.45 * short / max(wiggle, 1e-3))
    lo = np.array([margin, margin], float)
    hi = np.array([W - margin, H - margin], float)
    center = (lo + hi) / 2
    # 端を検知する距離：旋回に必要な距離＋逆向きの最大の曲がりから反転し終えるまでに進む距離
    look = 2.3 * R + L * (3 + 2 * kmax / slew)

    field = _Field(W, H, R, L, short, attractors, spacing=ATTRACTOR_SPACING * R)
    n_att = len(field.centers)

    def pick_start():
        return np.array([rng.uniform(lo[0] + look, hi[0] - look) if hi[0] - lo[0] > 2 * look else center[0],
                         rng.uniform(lo[1] + look, hi[1] - look) if hi[1] - lo[1] > 2 * look else center[1]])

    def new_bend():
        """次の「曲がり」の強さと長さ（歩数）を決める。"""
        r = R * math.exp(rng.uniform(0, math.log(rmax / R)))
        sweep = rng.uniform(0.5, 4.0)          # 曲がる角度（約30°〜230°）
        return rng.choice((-1, 1)) / r, max(3, int(sweep * r / L))

    def pick_focus(i):
        """重心 i の影響範囲の中で、いちばん線の少ない地点を目標にする。
        今いる場所のすぐ近くは選ばない（着いた直後にまた着いて、選び直しを繰り返さないように）。"""
        c = field.centers[i]
        best, best_d = c, None
        for _ in range(10):
            a = rng.uniform(0, 2 * math.pi)
            r = field.rho * math.sqrt(rng.random()) * 0.8
            q = c + r * np.array([math.cos(a), math.sin(a)])
            near_me = max(0.0, 0.5 * field.rho - math.hypot(q[0] - p[0], q[1] - p[1]))
            dq = field.at(q, W, H, margin) + 0.02 * np.linalg.norm(q - c) + 4.0 * near_me
            if best_d is None or dq < best_d:
                best, best_d = q, dq
        return best

    def pick_attractor():
        """空いている重心ほど選ばれやすくする。"""
        w = [field.freshness(i) ** 2 + 1e-3 for i in range(n_att)]
        i = rng.choices(range(n_att), weights=w)[0]
        return i, pick_focus(i)

    p = tuple(float(v) for v in pick_start())   # 歩くあいだの位置は（速さのため）ただの数値の組で持つ
    th = th0 = rng.uniform(-math.pi, math.pi)
    start = np.array(p)
    pts = [p]
    lo_x, lo_y, hi_x, hi_y = float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])
    cx_, cy_ = float(center[0]), float(center[1])
    turns = []
    # 直前に描いた自分の小さなループとの交差は数えない（渦巻きばかりになるのを防ぐ）
    grid = _SegmentGrid(max(4 * L, 1.0), skip_recent=recent_window(R, L))
    k = 0.0
    target, remain = new_bend()
    edge_mode = False
    edge_k = kmax
    bends = 0                                   # 直前の起点から曲がった回数
    limit = max(max_bends, 3) if max_bends else 0
    ckpts = [(0, th, k)]                        # 起点（交差地点）の履歴
    retries = rollbacks = crossings = switches = 0
    per_ckpt = 40                               # 同じ起点からのやり直し上限
    bend_starts = []                            # 各「曲がり」が始まった歩の番号（検証用）
    # 帰り道が長引いたら見切って作り直す（そのほうが速い）。曲がり回数の制限があるときは
    # やり直しで歩数が行き来するので少し長めに待つ
    n_max = n_main + int(n_target * (1.05 if max_bends else 0.45))
    # 帰り道では、描き始めの区間（最初の約20%）のどこかに向きをそろえて合流して閉じる
    J0 = max(20, int(0.2 * n_main))
    early_P = early_H = None
    home_pt, home_th = start, th0
    join_j, conn = 0, None
    retry_close_at = 0
    knot_w = max(5, int(2 * math.pi * 1.6 * R / L))   # 最小半径の輪 1.6 周ぶんの歩数
    wind = None
    hit_log = []                                # (歩, 交差された相手の線分番号)
    cur, focus = pick_attractor() if n_att else (None, None)
    refocus = max(10, int(field.rho / L))       # 目標地点を見直す間隔（歩）
    since_focus = 0
    probe = 3.0 * R

    loops = 0
    while True:
        n = len(turns)
        loops += 1
        if loops % 64 == 0:
            _check_stop(should_stop)           # 中断の確認（64歩ごと）
        if n > n_max:
            return None                         # 戻りきれなかった
        homing = n >= n_main                    # 後半は出発点へ戻る帰り道
        if homing and wind is None:
            wind = float(sum(turns[-knot_w:]))
        if homing and early_P is None:
            # （曲がり回数の制限があるときは、合流点より前を切り落とすとそこと交差していた分の
            #   交差が消えて条件が崩れるので、交差された線より前でだけ合流する：下の jlim）
            early_P = np.array(pts[:J0 + 1])
            early_H = th0 + np.cumsum(turns[:J0 + 1])     # early_H[j] = j歩目→j+1歩目の向き
            early_T = np.array(turns[:J0 + 1])             # early_T[j] = j歩目での曲がり
            jt = J0 // 2
            home_pt, home_th = early_P[jt], early_H[jt - 1]
        hx, hy = math.cos(th), math.sin(th)
        ax, ay = p[0] + hx * look, p[1] + hy * look
        near_edge = ax < lo_x or ay < lo_y or ax > hi_x or ay > hi_y
        to_center_left = hx * (cy_ - p[1]) - hy * (cx_ - p[0]) >= 0   # 画面の中央は左側か

        starting = False
        if near_edge and not edge_mode:          # 端の回避も1回の曲がり
            # 曲がりの強さに毎回わずかな揺らぎ（端で同じ円を何度もなぞらないように）
            edge_k = kmax * rng.uniform(0.8, 1.0)
            target = edge_k if to_center_left else -edge_k
            remain, edge_mode, starting = 10 ** 9, True, True
        elif edge_mode and not near_edge:
            edge_mode = False
            target, remain = new_bend()
            starting = True
        elif not edge_mode and remain <= 0:
            target, remain = new_bend()
            starting = True

        if starting:
            bend_starts.append(n)
            bends += 1
            if limit and bends > limit:
                # 交差しないまま曲がりすぎた → 直前の起点まで戻ってやり直す
                retries += 1
                rollbacks += 1
                if rollbacks > MAX_ROLLBACKS:
                    return None
                if retries > per_ckpt and len(ckpts) > 1:
                    # この起点からはうまくいかない → ひとつ前の起点まで戻る
                    ckpts.pop()
                    retries = 0
                    crossings = max(0, crossings - 1)
                n_keep, th, k = ckpts[-1]
                while bend_starts and bend_starts[-1] >= n_keep:
                    bend_starts.pop()
                bend_starts.append(n_keep)
                del pts[n_keep + 1:]
                del turns[n_keep:]
                grid.truncate(n_keep)
                field.rollback(n_keep)
                while hit_log and hit_log[-1][0] > n_keep:
                    hit_log.pop()
                p = pts[-1]
                target, remain = new_bend()
                edge_mode = False
                bends = 1
                continue

        tgt = target
        if edge_mode:
            # 端の回避中は、毎歩「画面の中央へ向かう側」に曲がる（角で向きを見失わないように）
            tgt = edge_k if to_center_left else -edge_k
        else:
            fade = 1.0
            if homing:
                fade = 1.0 - min(1.0, (n - n_main) / (0.25 * n_target))

            # --- 重心へ緩く向かう
            if n_att and pull > 0 and fade > 0:
                since_focus += 1
                fr = field.freshness(cur)
                dist_f = math.hypot(focus[0] - p[0], focus[1] - p[1])
                arrived = dist_f < 0.35 * field.rho
                # 選び直しは最低12歩あける（重心がすべて埋まったときに毎歩選び直さないように）
                if since_focus >= 12 and (fr < 0.25 or arrived or since_focus > refocus):
                    old = cur
                    if fr < 0.25 or arrived:
                        cur, focus = pick_attractor()     # 着いた／埋まった → 次の目標へ
                    else:
                        focus = pick_focus(cur)           # 定期的に、空いている地点へ目標を更新
                    switches += cur != old
                    since_focus = 0
                    fr = field.freshness(cur)
                    dist_f = math.hypot(focus[0] - p[0], focus[1] - p[1])
                want = math.atan2(focus[1] - p[1], focus[0] - p[0])
                turn = max(-1.0, min(1.0, _wrap(want - th) / (math.pi / 2)))
                near = min(1.0, dist_f / (0.6 * field.rho))      # 近くでは弱めて周回を防ぐ
                k_pull = PULL_GAIN * kmax * turn * near * pull * (0.25 + 0.75 * fr) * fade
                tgt = target * (1 - 0.7 * min(1.0, pull) * fade) + k_pull

            # --- 線の混んでいる方を避ける
            av = avoid
            if homing and math.hypot(home_pt[0] - p[0], home_pt[1] - p[1]) < 6 * R:
                av = 0.0                        # 合流点の手前では回避しない（戻れなくなるため）
            if limit:
                av *= max(0.0, 1.0 - bends / limit)   # 曲がり回数の上限が近いほど、交差しに行く
            if av > 0:
                c2t, s2t = math.cos(2 * th), math.sin(2 * th)
                px_, py_ = p

                def dens(a, dist):
                    return field.parallel_at((px_ + dist * math.cos(th + a), py_ + dist * math.sin(th + a)),
                                             c2t, s2t, W, H, margin)
                dl, df, dr = dens(0.7, probe), dens(0.0, probe), dens(-0.7, probe)
                sl, sr = dens(1.57, 1.3 * R), dens(-1.57, 1.3 * R)     # 真横（並走の検出）
                tot = dl + df + dr + sl + sr
                if tot > 0:
                    urgency = min(1.0, (df + 0.5 * (sl + sr)) / (1.5 * field.cs))
                    bias = ((dr - dl) + 2.0 * (sr - sl)) / (tot + 2 * L)
                    k_avoid = av * kmax * max(-1.0, min(1.0, 1.6 * bias)) * (0.5 + urgency)
                    tgt += k_avoid

            # --- 帰り道：描き始めの区間へ、同じ向きで合流するように向かう（少し手前を経由）
            if homing:
                w = 1.0 - fade
                jlim = J0
                if max_bends and hit_log:
                    # 帰り道で描き始めの線と交差した場合も含め、交差された線より前でだけ合流する
                    jlim = min(J0, min(j for _, j in hit_log))
                jt = max(1, min(J0 // 2, (jlim - 1) // 2))
                home_pt, home_th = early_P[jt], early_H[jt - 1]
                dist = math.hypot(home_pt[0] - p[0], home_pt[1] - p[1])
                back = min(3.0 * R, 0.5 * dist)
                want = math.atan2(home_pt[1] - back * math.sin(home_th) - p[1],
                                  home_pt[0] - back * math.cos(home_th) - p[0])
                k_home = 2.5 * _wrap(want - th) / max(dist, L)
                # ランダムな揺らぎも少し残して、帰り道が直線的にならないようにする
                tgt = w * k_home + (1 - 0.7 * w) * tgt
                # 描き始めの区間に十分近づいたら、つなぎの曲線で閉じられるか試す
                if n % 3 == 0 and jlim >= 3 and n >= retry_close_at:
                    dj = np.hypot(early_P[1:jlim, 0] - p[0], early_P[1:jlim, 1] - p[1])
                    if dj.min() < 6 * R:
                        closing = _try_close(p, th, turns[-1] if turns else 0.0, early_P, early_H, early_T,
                                             dj, L, R, min_radius, W, H, margin)
                        if closing is not None:
                            join_j, conn = closing
                            break
                        retry_close_at = n + max(3, int(0.8 * R / L))   # 少し進んでから再挑戦
        tgt = max(-kmax, min(kmax, tgt))
        k += max(-slew, min(slew, tgt - k))
        turns.append(k * L)
        if homing:
            # 帰り道で同じ場所をぐるぐる回り始めたら（結び目になる）、この線はやめて作り直す
            wind += k * L
            if len(turns) > knot_w:
                wind -= turns[-knot_w - 1]
            if abs(wind) > KNOT_TURNS * 2 * math.pi:
                return None
        th += k * L
        q = (p[0] + L * math.cos(th), p[1] + L * math.sin(th))
        hit = grid.add(p, q)
        p = q
        pts.append(p)
        field.deposit(len(turns), pts)
        remain -= 1
        if hit:
            crossings += 1
            hit_log.append((len(turns), grid.last_hit))
            if max_bends:
                # 交差した地点を新しい起点にする
                limit = max_bends
                bends = 0
                retries = 0
                ckpts.append((len(turns), th, k))

    # ---- 閉じる：合流点より前を切り落とし、つなぎの曲線を足して一周させる
    turns = np.concatenate([np.array(turns[join_j:]), conn])
    start = np.asarray(pts[join_j], float)
    th0 = float(early_H[join_j - 1])
    bend_starts = [b - join_j for b in bend_starts if b >= join_j]
    n = len(turns)
    heads = th0 + np.cumsum(turns)
    steps = L * np.stack([np.cos(heads), np.sin(heads)], 1)
    P = start + np.vstack([[0, 0], np.cumsum(steps, 0)])
    P = P + np.linspace(0, 1, len(P))[:, None] * (start - P[-1])   # 残る誤差は 0.5px 未満
    edge = margin * 0.5
    if P[:, 0].min() < edge or P[:, 1].min() < edge or \
            P[:, 0].max() > W - edge or P[:, 1].max() > H - edge:
        return None  # 画面からはみ出した
    if max_bends and not _bends_ok(P, bend_starts, n, max_bends, recent_window(R, L)):
        return None  # 閉じる補正で交差が変わった → 作り直す
    # 完成した線にも、同じ場所を何周も回る結び目がないか確かめる（つなぎ目もまたいで）
    tt = np.concatenate([turns, turns[:knot_w]])
    if np.abs(np.convolve(tt, np.ones(knot_w), "valid")).max() > KNOT_TURNS * 2 * math.pi:
        return None
    if stats is not None:
        stats.update(rollbacks=rollbacks, crossings=crossings, steps=n,
                     skip_recent=recent_window(R, L), bend_starts=bend_starts,
                     switches=switches, step_len=L, turn_radius=R)
    return P[:-1]


def make_stroke(rng, W, H, margin, min_radius, length, wiggle, max_bends=0, tries=60,
                stats=None, time_limit=60.0, **walk_kw):
    """条件（最小半径・画面内・曲がり回数）を満たす一筆書きの閉曲線を作る。

    - 線を滑らかなスプラインにすると曲率が少し強まる。強まり方は設定（なめらかさ等）で変わるので、
      最小半径を下回ったら、その分だけ余裕を広げて作り直す（自動調整）
    - time_limit 秒を超えたら打ち切る（それまでの最良の線を使うか、理由を添えてエラーにする）
    """
    t0 = time.time()
    best, best_r = None, -1.0
    # 歩くときの旋回半径 = 最小半径 / radius_margin。なめらかさが小さいほど曲がりが急に変わり
    # スプラインで曲率が強まりやすいので、最初から余裕を広めに取る（実測に合わせた式）
    smooth = float(walk_kw.get("smooth", 1.0))
    radius_margin = 0.72 * min(1.0, 0.62 + 0.5 * smooth)
    for _ in range(tries):
        _check_stop(walk_kw.get("should_stop"))
        if time.time() - t0 > time_limit:
            break
        st = {}
        P = walk_stroke(rng, W, H, margin, min_radius, length, wiggle, max_bends, st,
                        radius_margin=radius_margin, **walk_kw)
        if P is None:
            continue
        segs = closed_catmull_rom(P)
        r = min_curvature_radius(segs)
        if r >= min_radius:
            if stats is not None:
                stats.update(st, points=P, radius_margin=radius_margin)
            return segs
        if r > best_r:
            best, best_r = segs, r
            if stats is not None:
                stats.update(st, points=P, radius_margin=radius_margin)
        # 足りなかった分だけ余裕を広げる（次の線は少しゆるやかに曲がる）
        radius_margin = max(0.3, radius_margin * (r / min_radius) * 0.97)
    if best is None:
        if time.time() - t0 > time_limit:
            raise RuntimeError(f"{time_limit:.0f}秒以内に線を作れなかったため中止しました。"
                               "線を短くする、曲がってよい回数を増やす（または制限なし）、"
                               "最小半径を小さくする、のいずれかを試してください。")
        raise RuntimeError("曲線を作れませんでした。最小半径を小さくする、曲がってよい回数を増やす、"
                           "画面を大きくする、のいずれかを試してください。")
    print(f"警告: 最小半径 {min_radius:.1f}px を満たせませんでした（最良 {best_r:.1f}px）。",
          file=sys.stderr)
    return best


# ================================================================ 区画（面）の厳密な計算
def _find_intersections(Q, curve_of, local_of, nseg_of):
    """全折れ線分どうしの交点を求める（同じ曲線の隣り合う線分は除く）。

    戻り値: (i, j, t, u)  線分 i の t、線分 j の u の位置で交差
    """
    A = Q[:, 0]
    B = Q[:, 1]
    mid = (A + B) / 2
    cell = max(float(np.hypot(*(B - A).T).max()) * 1.01, 1e-6)
    cx = np.floor(mid[:, 0] / cell).astype(np.int64)
    cy = np.floor(mid[:, 1] / cell).astype(np.int64)
    key = cx * 1_000_003 + cy
    order = np.argsort(key, kind="stable")
    ks = key[order]
    uniq, start, counts = np.unique(ks, return_index=True, return_counts=True)

    out = []
    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1)):
        q = uniq + ox * 1_000_003 + oy
        pos = np.searchsorted(uniq, q)
        ok = (pos < len(uniq)) & (uniq[np.minimum(pos, len(uniq) - 1)] == q)
        c1 = np.nonzero(ok)[0]
        c2 = pos[ok]
        n1, n2 = counts[c1], counts[c2]
        tot = n1 * n2
        if tot.sum() == 0:
            continue
        rep_cell = np.repeat(np.arange(len(c1)), tot)
        offs = np.arange(tot.sum()) - np.repeat(np.cumsum(tot) - tot, tot)
        a_idx = order[start[c1][rep_cell] + offs // n2[rep_cell]]
        b_idx = order[start[c2][rep_cell] + offs % n2[rep_cell]]
        if ox == 0 and oy == 0:
            keep = a_idx < b_idx
            a_idx, b_idx = a_idx[keep], b_idx[keep]
        same = curve_of[a_idx] == curve_of[b_idx]
        dg = np.abs(local_of[a_idx] - local_of[b_idx])
        adjacent = same & ((dg <= 1) | (dg == nseg_of[a_idx] - 1))
        a_idx, b_idx = a_idx[~adjacent], b_idx[~adjacent]
        p, r = A[a_idx], B[a_idx] - A[a_idx]
        qq, s = A[b_idx], B[b_idx] - A[b_idx]
        den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
        w = qq - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[:, 0] * s[:, 1] - w[:, 1] * s[:, 0]) / den
            u = (w[:, 0] * r[:, 1] - w[:, 1] * r[:, 0]) / den
        hit = (den != 0) & (t >= 0) & (t < 1) & (u >= 0) & (u < 1)
        for i, j, tt, uu in zip(a_idx[hit], b_idx[hit], t[hit], u[hit]):
            out.append((int(i), int(j), float(tt), float(uu)))
    return out


def _point_in_poly(pt, poly):
    x, y = pt
    xs, ys = poly[:, 0], poly[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    cond = (ys > y) != (ys2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
    return bool(np.count_nonzero(cond & (x < xint)) % 2)


def build_faces(curves, m=8):
    """曲線群が作る平面グラフの面（区画）を求める。

    戻り値:
      faces: 各区画の dict {"path": SVGパス, "polys": 折れ線のリスト, "area": 面積}
      adj:   隣接リスト（辺を共有する区画どうし）
    """
    curves = [np.asarray(c, float) for c in curves]
    # --- 曲線ごとの折れ線（ベジェ1区間を m 本の線分に）
    polys, seg_rows = [], []
    for ci, segs in enumerate(curves):
        Qc = sample_beziers(segs, m)
        polys.append(Qc)
        N = len(Qc)
        seg_rows.append(np.stack([Qc, np.roll(Qc, -1, 0)], 1))
    Q = np.concatenate(seg_rows)
    curve_of = np.concatenate([np.full(len(p), ci) for ci, p in enumerate(polys)])
    local_of = np.concatenate([np.arange(len(p)) for p in polys])
    nseg_of = np.concatenate([np.full(len(p), len(p)) for p in polys])
    base = np.concatenate([[0], np.cumsum([len(p) for p in polys])])

    # --- 交点（頂点）と、各曲線上での出現位置
    hits = _find_intersections(Q, curve_of, local_of, nseg_of)
    node_pt = []
    occ = [[] for _ in curves]           # 曲線ごとの (位置, 頂点, 進行方向ベクトル)
    for i, j, t, u in hits:
        v = len(node_pt)
        r = Q[i, 1] - Q[i, 0]
        s = Q[j, 1] - Q[j, 0]
        node_pt.append(Q[i, 0] + t * r)
        occ[curve_of[i]].append((local_of[i] + t, v, r, r))
        occ[curve_of[j]].append((local_of[j] + u, v, s, s))
    for ci, o in enumerate(occ):
        if not o:                        # 交点のない曲線には仮の頂点を置く
            Pc = polys[ci]
            v = len(node_pt)
            node_pt.append(Pc[0])
            # (位置, 頂点, そこから出ていく向き, そこへ入ってくる向き)
            o.append((0.0, v, Pc[1] - Pc[0], Pc[0] - Pc[-1]))
    node_pt = np.array(node_pt)

    # --- 辺とハーフエッジ
    he_origin, he_dir, he_geom = [], [], []      # geom: (曲線, 開始位置, 終了位置, 順方向か)
    edge_comp = []
    for ci, o in enumerate(occ):
        o.sort(key=lambda x: x[0])
        N = len(polys[ci])
        for k in range(len(o)):
            pa, va, da, _ = o[k]
            pb, vb, _, db_in = o[(k + 1) % len(o)]
            if k + 1 == len(o):
                pb += N
            # 順方向（va→vb）と逆方向（vb→va）
            he_origin += [va, vb]
            he_dir += [da, -db_in]
            he_geom += [(ci, pa, pb, True), (ci, pa, pb, False)]
    nh = len(he_origin)
    twin = np.arange(nh) ^ 1
    ang = np.array([math.atan2(d[1], d[0]) for d in he_dir])
    out_of = [[] for _ in range(len(node_pt))]
    for h in range(nh):
        out_of[he_origin[h]].append(h)
    idx_in_node = np.zeros(nh, int)
    for v, lst in enumerate(out_of):
        lst.sort(key=lambda h: ang[h])
        for k, h in enumerate(lst):
            idx_in_node[h] = k

    def nxt(h):
        t = twin[h]
        v = he_origin[t]
        lst = out_of[v]
        return lst[(idx_in_node[t] - 1) % len(lst)]

    # --- ハーフエッジの形（折れ線・ベジェ片）
    def he_polyline(h):
        ci, pa, pb, fwd = he_geom[h]
        Pc = polys[ci]
        N = len(Pc)
        idx = np.arange(int(math.floor(pa)) + 1, int(math.ceil(pb)))
        pts_ = [Pc[int(math.floor(pa)) % N] + (pa - math.floor(pa)) * (Pc[(int(math.floor(pa)) + 1) % N] - Pc[int(math.floor(pa)) % N])]
        pts_ += list(Pc[idx % N])
        fb = int(math.floor(pb))
        pts_.append(Pc[fb % N] + (pb - fb) * (Pc[(fb + 1) % N] - Pc[fb % N]))
        arr = np.array(pts_)
        return arr if fwd else arr[::-1]

    def he_pieces(h):
        ci, pa, pb, fwd = he_geom[h]
        segs = curves[ci]
        ns = len(segs)
        a, b = pa / m, pb / m
        out = []
        s = a
        while s < b - 1e-9:
            i = int(math.floor(s + 1e-12))
            e = min(b, i + 1)
            if e - s > 1e-9:
                out.append(bez_sub(segs[i % ns], s - i, e - i))
            s = e
        if not fwd:
            out = [pc[::-1] for pc in out[::-1]]
        return out

    # --- 面をたどる
    face_of = np.full(nh, -1)
    cycles = []
    for h0 in range(nh):
        if face_of[h0] >= 0:
            continue
        cyc = []
        h = h0
        while face_of[h] < 0:
            face_of[h] = len(cycles)
            cyc.append(h)
            h = nxt(h)
        cycles.append(cyc)
    cyc_poly = []
    cyc_area = []
    for cyc in cycles:
        pl = np.vstack([he_polyline(h)[:-1] for h in cyc])
        cyc_poly.append(pl)
        x, y = pl[:, 0], pl[:, 1]
        cyc_area.append(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

    # --- 連結成分（交差でつながった曲線のまとまり）ごとの外周を見つける
    parent = list(range(len(curves)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j, _, _ in hits:
        a, b = find(curve_of[i]), find(curve_of[j])
        parent[a] = b
    comp_of_cycle = [find(he_geom[c[0]][0]) for c in cycles]
    outer = {}
    for f, comp in enumerate(comp_of_cycle):
        if comp not in outer or cyc_area[f] < cyc_area[outer[comp]]:
            outer[comp] = f
    outer_set = set(outer.values())
    interior = [f for f in range(len(cycles)) if f not in outer_set]

    # 外周が別の成分の区画の中にあるなら、その区画の「穴」になる
    merged = {}
    holes = {f: [] for f in interior}
    for comp, fo in outer.items():
        probe_pt = polys[[c for c in range(len(curves)) if find(c) == comp][0]][0]
        best, best_a = None, None
        for f in interior:
            if comp_of_cycle[f] == comp:
                continue
            if _point_in_poly(probe_pt, cyc_poly[f]) and (best is None or cyc_area[f] < best_a):
                best, best_a = f, cyc_area[f]
        if best is not None:
            merged[fo] = best
            holes[best].append(fo)

    fid = {f: i for i, f in enumerate(interior)}

    def region_of(f):
        f = merged.get(f, f)
        return fid.get(f, -1)

    faces = []
    for f in interior:
        subpaths = [pieces_to_path([pc for h in cycles[f] for pc in he_pieces(h)])]
        polys_f = [cyc_poly[f]]
        for hf in holes[f]:
            subpaths.append(pieces_to_path([pc for h in cycles[hf] for pc in he_pieces(h)]))
            polys_f.append(cyc_poly[hf])
        area = cyc_area[f] - sum(abs(cyc_area[hf]) for hf in holes[f])
        faces.append({"path": " ".join(subpaths), "polys": polys_f, "area": area})

    adj = [set() for _ in faces]
    for h in range(0, nh, 2):
        a, b = region_of(face_of[h]), region_of(face_of[h + 1])
        if a >= 0 and b >= 0 and a != b:
            adj[a].add(b)
            adj[b].add(a)
    return faces, adj, len(node_pt)


# ================================================================ 彩色
class _Budget(Exception):
    pass


def color_graph(adj, k, rng, randomness, should_stop=None, budget=20000):
    """DSatur + バックトラックで k 色彩色。

    randomness = 0 : 各色の使用回数がなるべく均等になる決定的な塗り方
    randomness = 1 : 使える色の中から完全ランダム
    その間は確率的に混ざる。
    バックトラックが長引いたら（まれ）、順番を変えて最初からやり直す。

    次に塗る区画（周りに使われている色の種類が最も多いもの）は、色の種類の数ごとの
    ヒープで管理して選ぶ。区画が数千あっても速い（選ばれる順番は全探索と同じ）。
    """
    import heapq
    n = len(adj)
    adj = [list(a) for a in adj]
    sys.setrecursionlimit(max(10000, 4 * n + 100))

    for attempt in range(12):
        colors = [-1] * n
        used = [0] * k
        tiebreak = [rng.random() for _ in range(n)]
        # 同じ「色の種類の数」の中では、この値が大きい区画から塗る（固定の値）
        static = [len(adj[v]) * (1 - randomness) + tiebreak[v] * (1 + randomness * len(adj[v]))
                  for v in range(n)]
        cnt = [[0] * k for _ in range(n)]      # 区画ごとの「周りで各色が使われている数」
        sat = [0] * n                          # 周りで使われている色の種類の数
        heaps = [[] for _ in range(k + 1)]
        for v in range(n):
            heaps[0].append((-static[v], v))
        heapq.heapify(heaps[0])
        steps = [0]

        def pick():
            for s_ in range(k, -1, -1):
                hp = heaps[s_]
                while hp:
                    _, v = hp[0]
                    if colors[v] < 0 and sat[v] == s_:
                        return v
                    heapq.heappop(hp)           # 古くなった記録は捨てる
            return None

        def paint(v, c):
            colors[v] = c
            used[c] += 1
            for u in adj[v]:
                cu = cnt[u]
                cu[c] += 1
                if cu[c] == 1:
                    sat[u] += 1
                    if colors[u] < 0:
                        heapq.heappush(heaps[sat[u]], (-static[u], u))

        def unpaint(v, c):
            colors[v] = -1
            used[c] -= 1
            for u in adj[v]:
                cu = cnt[u]
                cu[c] -= 1
                if cu[c] == 0:
                    sat[u] -= 1
                    if colors[u] < 0:
                        heapq.heappush(heaps[sat[u]], (-static[u], u))
            heapq.heappush(heaps[sat[v]], (-static[v], v))

        def solve(left):
            if left == 0:
                return True
            steps[0] += 1
            if steps[0] % 500 == 0:
                _check_stop(should_stop)
            if steps[0] > budget + n:
                raise _Budget()
            v = pick()
            cv = cnt[v]
            cand = [c for c in range(k) if cv[c] == 0]
            if rng.random() < randomness:
                rng.shuffle(cand)
            else:
                cand.sort(key=lambda c: (used[c], rng.random()))
            for c in cand:
                paint(v, c)
                if solve(left - 1):
                    return True
                unpaint(v, c)
            return False

        try:
            if solve(n):
                return colors
            break                                  # 本当に塗り分けられない
        except _Budget:
            continue                               # 順番を変えてやり直す
    raise RuntimeError(f"{k} 色では塗り分けられませんでした。色数を増やしてください。")


# ================================================================ 線の強弱
def stroke_widths(segs, stroke, pressure, min_radius, rng, short):
    """ベジェ区間ごとの線幅。カーブがきついほど細く、ゆるやかに波打つ揺らぎを加える。"""
    n = len(segs)
    if pressure <= 0 or stroke <= 0:
        return np.full(n, stroke)
    kap = np.minimum(1.0, segment_curvature(segs) * min_radius)
    lens = np.linalg.norm(np.asarray(segs)[:, 3] - np.asarray(segs)[:, 0], axis=1)
    s = np.cumsum(lens)
    total = s[-1]
    wave = np.zeros(n)
    for _ in range(3):                       # 閉曲線上で周期的になるゆるやかな揺らぎ
        cyc = rng.randint(2, max(3, int(total / (0.6 * short))))
        wave += np.sin(2 * math.pi * cyc * s / total + rng.uniform(0, 2 * math.pi))
    wave /= 3.0
    f = 1.0 - 0.55 * pressure * kap + 0.3 * pressure * wave
    # 隣どうしで急に変わらないようにならす
    ker = np.ones(5) / 5
    f = np.convolve(np.concatenate([f[-2:], f, f[:2]]), ker, "valid")
    return stroke * np.clip(f, 0.3, 1.5)


def stroke_runs(segs, widths, group=3):
    """線をいくつかの区間ごとに分け、区間ごとの線幅で描くための (ベジェ片リスト, 幅) を返す。"""
    n = len(segs)
    runs = []
    for i in range(0, n, group):
        idx = [(i + j) % n for j in range(group + 1)]   # 1区間重ねて継ぎ目を隠す
        runs.append(([segs[j] for j in idx[:-1]] + [segs[idx[-1]]], float(widths[idx[:group]].mean())))
    return runs


# ================================================================ 出力
def write_svg(path, W, H, faces, fill_cols, curves, widths, line_color, bg, uniform_stroke):
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(svg_string(W, H, faces, fill_cols, curves, widths, line_color, bg, uniform_stroke))


def svg_string(W, H, faces, fill_cols, curves, widths, line_color, bg, uniform_stroke):
    """SVG の中身を文字列で返す（GUI からも使う）。"""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    if bg:
        out.append(f'  <rect id="background" width="{W}" height="{H}" fill="{bg}"/>')
    # 塗りの継ぎ目のわずかな隙間を消すため、塗りと同色の極細の線を添える
    out.append('  <g id="fills" fill-rule="evenodd" stroke-width="0.6" stroke-linejoin="round">')
    for f, col in zip(faces, fill_cols):
        out.append(f'    <path fill="{col}" stroke="{col}" d="{f["path"]}"/>')
    out.append('  </g>')
    if uniform_stroke is not None and uniform_stroke > 0:
        out.append(f'  <g id="lines" fill="none" stroke="{line_color}" stroke-width="{uniform_stroke:.2f}" '
                   'stroke-linejoin="round" stroke-linecap="round">')
        for segs in curves:
            out.append(f'    <path d="{pieces_to_path(list(segs))}"/>')
        out.append('  </g>')
    elif uniform_stroke is None:
        out.append(f'  <g id="lines" fill="none" stroke="{line_color}" '
                   'stroke-linejoin="round" stroke-linecap="round">')
        for segs, w in zip(curves, widths):
            for run, wd in stroke_runs(segs, w):
                d = [f"M{_fmt(run[0][0])}"] + [f"C{_fmt(b)} {_fmt(c)} {_fmt(e)}" for _, b, c, e in run]
                out.append(f'    <path stroke-width="{wd:.2f}" d="{" ".join(d)}"/>')
        out.append('  </g>')
    out.append('</svg>')
    return "\n".join(out) + "\n"


def hex_to_bgr(h):
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def write_png(path, W, H, faces, fill_cols, curves, widths, line_color, bg, uniform_stroke,
              markers=(), ss=4, transparent=False):
    """PNG で書き出す（OpenCV で描画）。transparent=True なら背景を透明に（図形だけ）。"""
    import cv2
    if transparent:
        img = np.zeros((H * ss, W * ss, 4), np.uint8)
        col = lambda h: hex_to_bgr(h) + (255,)        # noqa: E731
    else:
        img = np.full((H * ss, W * ss, 3), hex_to_bgr(bg or "#FFFFFF"), np.uint8)
        col = hex_to_bgr
    sh = 4
    f = ss * (1 << sh)
    for face, fc in zip(faces, fill_cols):
        pts = [(p * f).round().astype(np.int32) for p in face["polys"]]
        cv2.fillPoly(img, pts, col(fc), lineType=cv2.LINE_AA, shift=sh)
        cv2.polylines(img, pts, True, col(fc), max(1, int(0.6 * ss)), cv2.LINE_AA, shift=sh)
    lc = col(line_color)
    for segs, w in zip(curves, widths):
        if uniform_stroke is not None:
            if uniform_stroke > 0:
                p = (sample_beziers(segs, 8) * f).round().astype(np.int32)
                cv2.polylines(img, [p], True, lc, max(1, int(round(uniform_stroke * ss))), cv2.LINE_AA, shift=sh)
            continue
        for run, wd in stroke_runs(segs, w):
            p = (sample_beziers(run, 8) * f).round().astype(np.int32)
            cv2.polylines(img, [p], False, lc, max(1, int(round(wd * ss))), cv2.LINE_AA, shift=sh)
    for (x, y) in markers:
        c = (int(x * ss), int(y * ss))
        cv2.circle(img, c, 7 * ss, col('#DC2828'), 2 * ss, cv2.LINE_AA)
        cv2.drawMarker(img, c, col('#DC2828'), cv2.MARKER_CROSS, 10 * ss, 2 * ss, cv2.LINE_AA)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if transparent:
        # 透明な背景となじませたふちは「色×不透明度」になっているので、色を戻す（黒ずみ防止）
        a = img[..., 3:4].astype(np.float32)
        rgb = img[..., :3].astype(np.float32)
        img[..., :3] = np.where(a > 0, np.clip(rgb * 255.0 / np.maximum(a, 1), 0, 255), 0).astype(np.uint8)
    cv2.imwrite(path, img)


# ================================================================ メイン
def parse_length(value, ref):
    """長さ指定を px に変換する。 "5%" → 短辺 ref の 5% / "40px"・"40"・40 → 40px"""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0 * ref
    if s.endswith("px"):
        s = s[:-2]
    return float(s)


def parse_count(value):
    """"2" → (2, 2) / "1-3" → (1, 3)"""
    s = str(value).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    return int(s), int(s)


# 線がカーブの内側で潰れないための、最小半径の下限（線幅に対する倍率）
RADIUS_FLOOR_PER_STROKE = 0.75


def generate(W=800, H=500, n_curves=1, colors=None, seed=None, randomness=0.5,
             wiggle=1.0, stroke="0.3%", min_radius="3%", length=12.0, max_bends=0,
             attractors="1-3", pull=0.5, avoid=0.6, smooth=1.0, pressure=0.5, should_stop=None,
             time_limit=60.0):
    """stroke と min_radius は "0.3%"（短辺に対する比率）か px の数値/文字列で指定。

    should_stop: 呼ぶと True を返したら生成を中断する関数（中断時は Cancelled 例外）。
    """
    colors = colors or DEFAULT_COLORS
    if len(colors) < 4:
        raise ValueError("色は4色以上指定してください。")
    ref = min(W, H)
    stroke = parse_length(stroke, ref)
    min_radius_req = parse_length(min_radius, ref)
    # 線幅に対して半径が小さすぎると、カーブの内側で線が潰れて黒い塊になる
    min_radius = max(min_radius_req, stroke * RADIUS_FLOOR_PER_STROKE, 1.0)
    rng = random.Random(seed)
    margin = max(30, 0.08 * ref, stroke)

    # 重心：画面の内側（端から15%以上内側）にランダムに置く
    a_min, a_max = parse_count(attractors)
    n_att = rng.randint(a_min, a_max) if a_max > 0 else 0
    att = [(rng.uniform(0.15 * W, 0.85 * W), rng.uniform(0.15 * H, 0.85 * H)) for _ in range(n_att)]

    walk_stats, curves = [], []
    for _ in range(n_curves):
        st = {}
        curves.append(make_stroke(rng, W, H, margin, min_radius, length, wiggle, max_bends, stats=st,
                                  attractors=att, pull=pull, avoid=avoid, smooth=smooth,
                                  should_stop=should_stop, time_limit=time_limit))
        walk_stats.append(st)

    _check_stop(should_stop)
    faces, adj, n_nodes = build_faces(curves)
    _check_stop(should_stop)
    cidx = color_graph(adj, len(colors), rng, randomness, should_stop=should_stop)
    for v, nb in enumerate(adj):            # 念のため検証
        assert all(cidx[v] != cidx[u] for u in nb)

    widths = [stroke_widths(c, stroke, pressure, min_radius, rng, ref) for c in curves]
    info = {"stroke_px": stroke, "min_radius_px": min_radius, "min_radius_requested_px": min_radius_req,
            "walk": walk_stats, "attractors": att, "crossings": n_nodes}
    return curves, faces, [colors[c] for c in cidx], adj, widths, info


def config_to_defaults(cfg):
    """GUI で保存した設定（JSON）を、コマンドラインの既定値の形に変換する。

    GUI では線の太さ・最小半径は「短辺に対する %」の数値、重心の数は最小・最大で保存している。
    """
    d = {}
    for k in ("width", "height", "curves", "seed", "max_bends"):
        if k in cfg:
            d[k] = int(cfg[k])
    for k in ("randomness", "wiggle", "length", "pull", "avoid", "smooth", "pressure"):
        if k in cfg:
            d[k] = float(cfg[k])
    for k in ("stroke", "min_radius"):
        if k in cfg:
            v = str(cfg[k]).strip()
            d[k] = v if v.endswith(("%", "px")) else f"{float(v)}%"
    if "att_min" in cfg or "att_max" in cfg:
        lo = int(cfg.get("att_min", 0))
        d["attractors"] = f"{lo}-{max(lo, int(cfg.get('att_max', lo)))}"
    elif "attractors" in cfg:
        d["attractors"] = str(cfg["attractors"])
    if "colors" in cfg:
        c = cfg["colors"]
        d["colors"] = ",".join(c) if isinstance(c, list) else str(c)
    for k in ("line_color", "bg"):
        if k in cfg:
            d[k] = str(cfg[k])
    if cfg.get("show_att"):
        d["show_attractors"] = True
    return d


def main():
    ap = argparse.ArgumentParser(description="自由曲線で区切った区画を自動で塗り分けてSVG出力")
    ap.add_argument("--config", help="GUI で保存・書き出した設定ファイル（JSON）。他のオプションで上書きできる")
    ap.add_argument("-o", "--out", help="出力SVG（省略時: region_v版_年月日-時分秒_s種.svg）")
    ap.add_argument("--png", nargs="?", const="auto",
                    help="PNGも出力（opencv-python が必要）。パス省略時はSVGと同じ名前")
    ap.add_argument("--transparent", action="store_true",
                    help="背景を透明にして図形だけを保存（SVG・PNGとも）")
    ap.add_argument("--style", help="おすすめ設定の名前（--list-styles で一覧）。他のオプションで上書きできる")
    ap.add_argument("--list-styles", action="store_true", help="おすすめ設定の一覧を表示して終了")
    ap.add_argument("--version", action="version", version=f"region_fill {__version__}")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=500)
    ap.add_argument("--curves", type=int, default=1, help="一筆書きの線の本数（1で完全な一筆書き）")
    ap.add_argument("--colors", default=",".join(DEFAULT_COLORS), help="カンマ区切りの色（4色以上）")
    ap.add_argument("--palette", help="配色プリセットの名前（例: '夕焼け', '赤', 'グレースケール'）。"
                                      "--list-palettes で一覧。--colors を指定するとそちらが優先")
    ap.add_argument("--list-palettes", action="store_true", help="配色プリセットの一覧を表示して終了")
    ap.add_argument("--palette-from", metavar="画像", help="画像ファイルの代表色を配色にする（Pillow が必要）")
    ap.add_argument("--palette-count", type=int, default=5, help="--palette-from で取り出す色の数（4〜10）")
    ap.add_argument("--seed", type=int, help="乱数シード（同じ値なら同じ絵）")
    ap.add_argument("--randomness", type=float, default=0.5,
                    help="色選びのランダムさ 0〜1（0=色の使用回数を均等に, 1=完全ランダム）")
    g = ap.add_argument_group("線の形")
    g.add_argument("--length", type=float, default=12.0,
                   help="線の長さ（画面の短辺の何倍か）。長いほど交差と区画が増える")
    g.add_argument("--wiggle", type=float, default=1.0,
                   help="線のクネクネ度合い。大きいほど小さなカーブやループが増える")
    g.add_argument("--min-radius", default="3%",
                   help="曲線の最小曲率半径。'5%%'（短辺に対する比率）または '40' / '40px'。"
                        "線が潰れないよう、線幅の0.75倍は自動で下限になる")
    g.add_argument("--smooth", type=float, default=1.0,
                   help="曲がりの入り・抜けの長さ（大きいほどなめらか）。0.5〜3程度")
    g.add_argument("--max-bends", type=int, default=0,
                   help="交差するまでに曲がってよい回数（0で制限なし、2以上）。"
                        "交差しないまま超えたら直前の交差地点からやり直す")
    g = ap.add_argument_group("重心と重なり回避")
    g.add_argument("--attractors", default="1-3",
                   help="重心の数。'2' や '1-3'（範囲からランダム）。'0' で重心なし")
    g.add_argument("--pull", type=float, default=0.5, help="重心へ向かう強さ 0〜1（緩く=0.3〜0.6）")
    g.add_argument("--avoid", type=float, default=0.6,
                   help="線の混んでいる場所を避ける強さ 0〜1（0で回避なし）")
    g.add_argument("--show-attractors", action="store_true", help="PNGプレビューに重心の位置を表示")
    g = ap.add_argument_group("線の見た目")
    g.add_argument("--stroke", default="0.3%",
                   help="線の太さ（既定 0.3%%）。'0.3%%'（短辺に対する比率）または '2' / '2px'。0 で線なし（塗りだけ）")
    g.add_argument("--pressure", type=float, default=0.5,
                   help="線の強弱 0〜1（0=均一。大きいほどカーブで細く、ゆるやかに太さが揺らぐ）")
    g.add_argument("--line-color", default="#111111")
    g.add_argument("--bg", default="#FFFFFF", help="背景色（曲線の外側）")
    pre, _ = ap.parse_known_args()
    if pre.list_styles:
        for st in STYLE_PRESETS:
            print(f"  {st['name']:<10s} {st['desc']}")
        return
    if pre.style:
        st = style_settings(pre.style)
        if st is None:
            ap.error(f"おすすめ設定「{pre.style}」が見つかりません。--list-styles で一覧を確認してください。")
        ap.set_defaults(**config_to_defaults(st))
    if pre.config:
        import json
        with open(pre.config, encoding="utf-8") as fp:
            cfg = json.load(fp)
        cfg = cfg.get("settings", cfg)     # GUI の書き出し形式（{"settings": {...}}）にも対応
        ap.set_defaults(**config_to_defaults(cfg))
    if pre.list_palettes:
        cat = None
        for p in PALETTES:
            if p["category"] != cat:
                cat = p["category"]
                print(f"\n[{cat}]")
            extra = "（線・背景も変更）" if ("line" in p or "bg" in p) else ""
            print(f"  {p['name']:<14s} {' '.join(p['colors'])}{extra}")
        return
    if pre.palette:
        pal = find_palette(pre.palette)
        if pal is None:
            ap.error(f"配色プリセット「{pre.palette}」が見つかりません。--list-palettes で一覧を確認してください。")
        d = {"colors": ",".join(pal["colors"])}
        if "line" in pal:
            d["line_color"] = pal["line"]
        if "bg" in pal:
            d["bg"] = pal["bg"]
        ap.set_defaults(**d)
    if pre.palette_from:
        cols = palette_from_image(pre.palette_from, max(4, min(10, pre.palette_count)))
        print("画像から作った配色:", " ".join(cols))
        ap.set_defaults(colors=",".join(cols))
    args = ap.parse_args()
    if args.max_bends == 1 or args.max_bends < 0:
        ap.error("--max-bends は 0（制限なし）か 2 以上を指定してください。"
                 "1 は条件が厳しすぎて線を作れないことが多いためです。")

    if args.seed is None:
        # シードを省略したらランダムに決めて表示する（同じ絵をあとで再現できるように）
        args.seed = random.randrange(100000)
        print(f"シード: {args.seed}（同じ絵は --seed {args.seed} で再現できます）")
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    curves, faces, fill_cols, adj, widths, info = generate(
        args.width, args.height, args.curves, colors, args.seed,
        max(0.0, min(1.0, args.randomness)), args.wiggle, args.stroke,
        min_radius=args.min_radius, length=args.length, max_bends=args.max_bends,
        attractors=args.attractors, pull=args.pull, avoid=args.avoid, smooth=args.smooth,
        pressure=args.pressure)
    stroke = info["stroke_px"]
    uniform = stroke if (args.pressure <= 0 or stroke <= 0) else None
    if not args.out:
        base = f"region_{version_stamp()}_s{args.seed}"
        args.out, i = base + ".svg", 2
        while os.path.exists(args.out):          # 同じ秒に続けて実行しても上書きしない
            args.out, i = f"{base}-{i}.svg", i + 1
    bg = None if args.transparent else args.bg

    write_svg(args.out, args.width, args.height, faces, fill_cols, curves, widths,
              args.line_color, bg, uniform)
    if args.png:
        png = args.png if args.png != "auto" else (args.out[:-4] if args.out.lower().endswith(".svg")
                                                   else args.out) + ".png"
        write_png(png, args.width, args.height, faces, fill_cols, curves, widths,
                  args.line_color, args.bg, uniform,
                  markers=info["attractors"] if args.show_attractors else (), transparent=args.transparent)
    n_edges = sum(len(s) for s in adj) // 2
    note = ""
    if info["min_radius_px"] > info["min_radius_requested_px"] + 1e-9:
        note = f"（線幅に合わせて {info['min_radius_requested_px']:.1f}px から引き上げ）"
    print(f"区画 {len(faces)} 個 / 交点 {info['crossings']} / 隣接 {n_edges} 組 / {len(colors)} 色 / "
          f"線 {stroke:.1f}px / 最小半径 {info['min_radius_px']:.1f}px{note} / "
          f"重心 {len(info['attractors'])} 個 → {args.out}")
    if args.max_bends:
        rb = sum(w.get("rollbacks", 0) for w in info["walk"])
        print(f"  曲がり {args.max_bends} 回以内で交差（やり直し {rb} 回）")


if __name__ == "__main__":
    main()
