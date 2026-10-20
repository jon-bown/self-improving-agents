"""Build the architecture diagrams from docs/architecture/src/*.mmd.

For each diagram this writes:
    docs/architecture/mmd/<name>.mmd   standalone Mermaid (styles inlined) -- paste into mermaid.live,
                                       Notion, Confluence, or a slide tool that speaks Mermaid
    docs/architecture/svg/<name>.svg   vector, for slides and the web
    docs/architecture/png/<name>.png   2x raster, for tools that want images
    docs/architecture/index.html       a gallery / presenter view (arrow keys, F for full screen)
and refreshes every markdown block wrapped in
    <!-- diagram:<name> --> ... <!-- /diagram -->
so the diagrams embedded in READMEs (rendered natively by GitHub) never drift from the source.

    python scripts/render_diagrams.py            # everything
    python scripts/render_diagrams.py --no-images  # only .mmd + markdown (no Node/Chrome needed)

Images use Mermaid CLI via npx. It reuses a local Chrome/Chromium when it finds one
(set CHROME_PATH to override) instead of downloading its own.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "docs" / "architecture"
SRC = ARCH / "src"
MERMAID_CLI = "@mermaid-js/mermaid-cli@12"
MERMAID_CONFIG = {
    "theme": "base",
    "themeVariables": {"fontFamily": "Inter, Helvetica, Arial, sans-serif", "fontSize": "15px",
                       "lineColor": "#495057", "primaryTextColor": "#212529"},
    "flowchart": {"curve": "basis", "htmlLabels": True, "padding": 12, "nodeSpacing": 40, "rankSpacing": 50,
                  "wrappingWidth": 280},
}
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
]


def build(name: str, styles: str) -> str:
    """Insert the shared classDefs right after the `flowchart ...` line (after any front matter)."""
    text = (SRC / f"{name}.mmd").read_text()
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if re.match(r"\s*(flowchart|graph)\s+(LR|RL|TB|TD|BT)\b", line))
    classdefs = [s for s in styles.splitlines() if s.strip() and not s.lstrip().startswith("%%")]
    return "\n".join(lines[: i + 1] + classdefs + lines[i + 1:]) + "\n"


def sync_markdown(diagrams: dict[str, str]) -> list[Path]:
    changed = []
    pattern = re.compile(r"(<!-- diagram:([\w-]+) -->\n)(.*?)(<!-- /diagram -->)", re.DOTALL)
    for md in [ROOT / "README.md", *ROOT.glob("docs/**/*.md"), *ROOT.glob("levels/*/README.md")]:
        text = md.read_text()

        def repl(m):
            if m.group(2) not in diagrams:
                raise SystemExit(f"{md}: unknown diagram '{m.group(2)}'")
            return f"{m.group(1)}```mermaid\n{diagrams[m.group(2)]}```\n{m.group(4)}"

        new = pattern.sub(repl, text)
        if new != text:
            md.write_text(new)
            changed.append(md.relative_to(ROOT))
    return changed


GALLERY = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Self-Improving Agent Architecture</title>
<style>
  :root { --bg:#ffffff; --fg:#212529; --muted:#868e96; --accent:#e8590c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 Inter,Helvetica,Arial,sans-serif; }
  header { padding:16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; border-bottom:1px solid #e9ecef; }
  header h1 { font-size:18px; margin:0 12px 0 0; }
  header button { border:1px solid #ced4da; background:#fff; border-radius:6px; padding:4px 10px; cursor:pointer; }
  header button[aria-current=true] { border-color:var(--accent); color:var(--accent); }
  .hint { color:var(--muted); margin-left:auto; font-size:13px; }
  section { padding:24px 16px 48px; max-width:1400px; margin:0 auto; overflow-x:auto; }
  pre.mermaid { margin:0; }
  section .mermaid { display:flex; justify-content:center; }
  section .mermaid svg { max-width:100%; height:auto; }
  body.present header { display:none; }
  body.present section { display:none; max-width:none; height:100vh; padding:24px; }
  body.present section.on { display:block; }
  body.present section.on pre.mermaid { width:100%; height:100%; }
  body.present section.on pre.mermaid svg { width:100% !important; height:100% !important; max-width:none !important; }
</style></head>
<body>
<header><h1>Self-improving agent: architecture</h1>__NAV__
<span class="hint">&larr; &rarr; step &middot; P present &middot; Esc exit</span></header>
__SECTIONS__
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@12/dist/mermaid.esm.min.mjs";
mermaid.initialize(Object.assign({ startOnLoad:false }, __CONFIG__));
await mermaid.run({ querySelector: ".mermaid", suppressErrors: true });
const secs = [...document.querySelectorAll("section")], btns = [...document.querySelectorAll("header button")];
let i = 0;
function show(n) {
  i = (n + secs.length) % secs.length;
  secs.forEach((s, k) => s.classList.toggle("on", k === i));
  btns.forEach((b, k) => b.setAttribute("aria-current", k === i));
  if (!document.body.classList.contains("present")) secs[i].scrollIntoView({ behavior: "smooth" });
}
btns.forEach((b, k) => b.onclick = () => show(k));
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight" || e.key === " ") show(i + 1);
  else if (e.key === "ArrowLeft") show(i - 1);
  else if (e.key === "p" || e.key === "P") { document.body.classList.toggle("present"); show(i); }
  else if (e.key === "Escape") document.body.classList.remove("present");
});
const start = secs.findIndex(s => "#" + s.id === location.hash);
show(start < 0 ? 0 : start);
</script></body></html>
"""


def write_gallery(diagrams: dict[str, str]) -> None:
    import html

    def title(src: str, name: str) -> str:
        m = re.search(r'^title:\s*"?(.*?)"?\s*$', src, re.MULTILINE)
        return m.group(1) if m else name

    nav = "".join(f'<button>{html.escape(title(src, n).split(":")[0])}</button>' for n, src in diagrams.items())
    sections = "\n".join(f'<section id="{n}"><pre class="mermaid">{html.escape(src)}</pre></section>'
                         for n, src in diagrams.items())
    page = (GALLERY.replace("__NAV__", nav).replace("__SECTIONS__", sections)
            .replace("__CONFIG__", json.dumps(MERMAID_CONFIG)))
    (ARCH / "index.html").write_text(page)


def render_images(names: list[str]) -> None:
    if not shutil.which("npx"):
        raise SystemExit("npx not found: install Node.js, or run with --no-images")
    chrome = os.getenv("CHROME_PATH") or next((c for c in CHROME_CANDIDATES if Path(c).exists()), None)
    env = dict(os.environ)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "mermaid.json"
        cfg.write_text(json.dumps(MERMAID_CONFIG))
        pup = Path(tmp) / "puppeteer.json"
        launch = {"protocolTimeout": 900_000}  # rendering many diagrams in one page can be slow
        if chrome:
            launch |= {"executablePath": chrome, "headless": "new", "args": ["--no-sandbox", "--disable-gpu"]}
        pup.write_text(json.dumps(launch))
        if chrome:
            env["PUPPETEER_SKIP_DOWNLOAD"] = "1"
        for fmt, extra in (("svg", []), ("png", ["--scale", "2"])):
            (ARCH / fmt).mkdir(exist_ok=True)
            for name in names:
                src, out = ARCH / "mmd" / f"{name}.mmd", ARCH / fmt / f"{name}.{fmt}"
                if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
                    continue  # up to date
                cmd = ["npx", "-y", MERMAID_CLI, "-i", str(src), "-o", str(out), "-c", str(cfg),
                       "-p", str(pup), "-b", "white", *extra]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise SystemExit(f"mermaid-cli failed on {out.name}:\n{proc.stderr[-2000:]}")
                print(f"  {out.relative_to(ROOT)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    styles = (SRC / "_styles.mmd").read_text()
    names = sorted(p.stem for p in SRC.glob("*.mmd") if not p.stem.startswith("_"))
    (ARCH / "mmd").mkdir(exist_ok=True)
    diagrams = {}
    for name in names:
        diagrams[name] = build(name, styles)
        target = ARCH / "mmd" / f"{name}.mmd"
        if not target.exists() or target.read_text() != diagrams[name]:
            target.write_text(diagrams[name])
    write_gallery(diagrams)
    print(f"{len(names)} diagrams -> docs/architecture/mmd/ and docs/architecture/index.html")
    for md in sync_markdown(diagrams):
        print(f"  updated {md}")
    if not args.no_images:
        render_images(names)


if __name__ == "__main__":
    main()
