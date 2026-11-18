import os
import re
import sys
import shutil
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap import config

cfg = config.load()
trips = sys.argv[1:] or ["00000056", "00000077"]
chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
if not chrome:
    sys.exit("needs Google Chrome or Chromium on PATH; otherwise open maps/*.html in a browser and screenshot manually")
for t in trips:
    src = os.path.join(cfg["maps_dir"], f"student_test_{t}.html")
    out = os.path.join(cfg["figures_dir"], f"folium_screenshot_{t}.png")
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--window-size=1400,950",
                    "--virtual-time-budget=20000", f"--screenshot={out}", "file://" + os.path.abspath(src)], check=True, capture_output=True)
    print(out)
