"""Build a browsable HTML gallery of all rendered run videos.

Scans for <repo>/*.mp4 and <repo>/runs/**/*.mp4 and writes gallery.html at the
repo root. Open it in a browser to review every recording in one page.

Usage:
    python scripts/build_gallery.py
    python scripts/build_gallery.py --open   # also open it in the default browser
"""

import argparse
import glob
import os
import webbrowser

REPO = os.path.dirname(os.path.dirname(__file__))


def find_videos() -> list[str]:
    vids = glob.glob(os.path.join(REPO, "*.mp4"))
    vids += glob.glob(os.path.join(REPO, "runs", "**", "*.mp4"), recursive=True)
    # De-dupe, sort newest-first by mtime.
    vids = sorted(set(vids), key=lambda p: os.path.getmtime(p), reverse=True)
    return vids


def build(open_after: bool) -> None:
    vids = find_videos()
    cards = []
    for v in vids:
        rel = os.path.relpath(v, REPO).replace("\\", "/")
        mtime = __import__("datetime").datetime.fromtimestamp(
            os.path.getmtime(v)
        ).strftime("%Y-%m-%d %H:%M")
        size_mb = os.path.getsize(v) / 1e6
        cards.append(
            f"""    <figure class="card">
      <video src="{rel}" controls preload="metadata" loop muted></video>
      <figcaption><b>{rel}</b><br><span>{mtime} &middot; {size_mb:.1f} MB</span></figcaption>
    </figure>"""
        )
    grid = "\n".join(cards) if cards else "<p>No .mp4 videos found yet.</p>"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Battle Royale — run videos</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background:#111; color:#eee; }}
  h1 {{ font-weight: 600; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:1.2rem; }}
  .card {{ margin:0; background:#1c1c1f; border-radius:10px; padding:.6rem; }}
  video {{ width:100%; border-radius:6px; background:#000; }}
  figcaption {{ font-size:.85rem; margin-top:.4rem; word-break:break-all; }}
  figcaption span {{ color:#999; }}
</style></head><body>
  <h1>Battle Royale — run videos ({len(vids)})</h1>
  <p>Newest first. Click a video to play; they loop muted on hover-play.</p>
  <div class="grid">
{grid}
  </div>
</body></html>"""
    out = os.path.join(REPO, "gallery.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out} with {len(vids)} video(s).")
    if open_after:
        webbrowser.open("file://" + out.replace("\\", "/"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build an HTML gallery of run videos")
    p.add_argument("--open", action="store_true", help="open in the default browser")
    a = p.parse_args()
    build(a.open)
