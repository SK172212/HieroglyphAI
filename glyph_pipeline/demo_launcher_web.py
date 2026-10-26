# Run with:
#   cd /Users/sohankilaru/Desktop/hieroglyphProject  /usr/bin/python3 glyph_pipeline/demo_launcher_web.py

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import subprocess
import threading
import webbrowser
import os
import sys
import html

PROJECT_ROOT = Path("/Users/sohankilaru/Desktop/hieroglyphProject")

SCRIPT_PATH = PROJECT_ROOT / "glyph_pipeline" / "detect_and_classify.py"
MODEL_PYTHON = PROJECT_ROOT / "pipeline_env" / "bin" / "python"
OUTPUT_DIR = PROJECT_ROOT / "glyph_pipeline" / "output"
BOARD_IMAGE = PROJECT_ROOT / "glyph_pipeline" / "demo_board.png"


IMAGE_MAP = {
    "img1": PROJECT_ROOT / "glyph_pipeline" / "test" / "ptest_10.png",  # left tall
    "img2": PROJECT_ROOT / "glyph_pipeline" / "test" / "ptest_16.png",  # second tall
    "img3": PROJECT_ROOT / "glyph_pipeline" / "test" / "ptest_26.png",  # thin middle
    "img4": PROJECT_ROOT / "glyph_pipeline" / "test" / "ptest_6.png",   # top-right
    "img5": PROJECT_ROOT / "glyph_pipeline" / "test" / "ptest_24.png",  # middle-right
    "img6": PROJECT_ROOT / "glyph_pipeline" / "test" / "ptest_11.png",  # bottom-right
}

# Coordinates for demo_board.png (960 x 540)
CLICK_REGIONS = {
    "img1": (15, 10, 165, 525),   # ptest_10
    "img2": (175, 10, 365, 525),  # ptest_16
    "img3": (375, 10, 500, 525),  # ptest_26
    "img4": (515, 10, 915, 205),  # ptest_6
    "img5": (515, 215, 915, 390), # ptest_24
    "img6": (515, 400, 915, 525), # ptest_11
}

HOST = "127.0.0.1"
PORT = 8008


def open_file(path: Path):
    path = str(path.resolve())
    if sys.platform.startswith("darwin"):
        subprocess.run(["open", path], check=False)
    elif os.name == "nt":
        os.startfile(path)
    else:
        subprocess.run(["xdg-open", path], check=False)


def validate_paths():
    for p in [SCRIPT_PATH, MODEL_PYTHON, BOARD_IMAGE]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
    for key, p in IMAGE_MAP.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing image for {key}: {p}")


def run_model_for_key(key: str):
    selected_image = IMAGE_MAP[key]

    result = subprocess.run(
        [str(MODEL_PYTHON), str(SCRIPT_PATH), str(selected_image)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    image_stem = selected_image.stem
    out_folder = OUTPUT_DIR / image_stem
    out_img = out_folder / f"{image_stem}_detcls.png"

    if not out_img.exists():
        raise FileNotFoundError(
            f"Model ran, but output image was not found: {out_img}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    open_file(out_img)
    return selected_image.name, out_img.name


def build_html(status_text: str = "Click an inscription to run the model."):
    status_text = html.escape(status_text)

    areas = []
    for key, (x1, y1, x2, y2) in CLICK_REGIONS.items():
        coords = f"{x1},{y1},{x2},{y2}"
        areas.append(
            f'<area shape="rect" coords="{coords}" href="/run?img={key}" alt="{key}" title="{key}">'
        )
    areas_html = "\n".join(areas)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Full Hieroglyh Pipeline Demo</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background: #ffffff;
      text-align: center;
      margin: 20px;
    }}
    h1 {{
      font-size: 28px;
      margin-bottom: 10px;
    }}
    .status {{
      font-size: 18px;
      margin: 14px 0 20px 0;
    }}
    .board {{
      display: inline-block;
      border: 2px solid #333;
    }}
    .note {{
      margin-top: 14px;
      color: #444;
      font-size: 15px;
    }}
  </style>
</head>
<body>
  <h1>Full Hieroglyph Pipeline Demo</h1>
  <div class="status">{status_text}</div>

  <div class="board">
    <img src="/board.png" usemap="#inscriptions" alt="Demo board">
    <map name="inscriptions">
      {areas_html}
    </map>
  </div>

  <div class="note">Click any inscription.</div>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            page = build_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))
            return

        if parsed.path == "/board.png":
            try:
                data = BOARD_IMAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_error(500, f"Could not load board image: {e}")
            return

        if parsed.path == "/run":
            qs = parse_qs(parsed.query)
            key = qs.get("img", [None])[0]

            if key not in IMAGE_MAP:
                self.send_error(400, "Invalid image key.")
                return

            try:
                selected_name, out_name = run_model_for_key(key)
                page = build_html(
                    f"Done: {selected_name}. Output opened automatically ({out_name})."
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page.encode("utf-8"))
            except subprocess.CalledProcessError as e:
                page = build_html(
                    "Run failed.<br><pre>"
                    + html.escape(f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}")
                    + "</pre>"
                )
                self.send_response(500)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page.encode("utf-8"))
            except Exception as e:
                page = build_html(f"Error: {html.escape(str(e))}")
                self.send_response(500)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page.encode("utf-8"))
            return

        self.send_error(404, "Not found.")

    def log_message(self, format, *args):
        return


def main():
    validate_paths()
    server = HTTPServer((HOST, PORT), DemoHandler)
    url = f"http://{HOST}:{PORT}/"

    print(f"Serving demo at {url}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()