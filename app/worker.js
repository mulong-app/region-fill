// Region Fill ブラウザ版の計算担当（Web Worker）
// ブラウザの中で Python（Pyodide）を動かし、パソコン版と同じ region_fill.py / region_fill_api.py で計算する。
// 画面（index.html）とは、{type:"call", id, name, json} → {type:"result", id, json} でやりとりする。
let api = null;

function progress(text) { postMessage({ type: "progress", text }); }

async function init(indexURL) {
  progress("ブラウザ版を準備しています…（初回は10〜20秒ほどかかります）");
  importScripts(indexURL + "pyodide.js");
  const py = await loadPyodide({ indexURL });
  progress("計算用のライブラリ（numpy）を読み込んでいます…");
  await py.loadPackage("numpy");
  progress("プログラムを読み込んでいます…");
  for (const f of ["region_fill.py", "region_fill_api.py"]) {
    // このファイル（app/worker.js）から見て1つ上のフォルダにある、パソコン版と同じファイルを使う
    const r = await fetch(new URL("../" + f, self.location).href, { cache: "no-cache" });
    if (!r.ok) throw new Error(f + " を読み込めませんでした（" + r.status + "）");
    py.FS.writeFile("/home/pyodide/" + f, await r.text());
  }
  py.runPython("import sys\nif '/home/pyodide' not in sys.path:\n    sys.path.insert(0, '/home/pyodide')");
  api = py.pyimport("region_fill_api");
  postMessage({ type: "ready" });
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    try { await init(m.pyodide); }
    catch (err) { postMessage({ type: "fatal", error: String((err && err.message) || err) }); }
    return;
  }
  if (m.type === "call") {
    let out;
    try { out = api.call(m.name, m.json); }
    catch (err) {
      const msg = String((err && err.message) || err).trim().split("\n").slice(-1)[0];
      out = JSON.stringify({ error: "計算中にエラーが起きました：" + msg });
    }
    postMessage({ type: "result", id: m.id, json: out });
  }
};
