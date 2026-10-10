#!/usr/bin/env python3
"""
Generate the animated GitHub profile banners (Dark and Light modes).

Run from the repository root:
    python scripts/banner/generate.py
"""

from __future__ import annotations

import html
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/avatar.png"   # Put your photo here (png or jpg)
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

W, H = 1180, 610
LOOP_SECONDS = 14.2

# ==============================================================================
# USER METADATA (SYSTEM.INFO)
# ==============================================================================
SYSTEM_DATA = [
    ("Subject", "Divyansh Verma"),
    ("Role", "Full Stack Engineer · Systems"),
    ("Origin", "India (Delhi-NCR)"),
    ("Education", "B.Tech CSE · BML Munjal Univ"),
    ("Status", "Building + Shipping + Problem Solving"),
    ("ToolChain", "VS Code · Git · Docker · Postman"),
    ("Core.Lang", "C++ · Python · JavaScript · Java · SQL"),
    ("Core.Frontend", "React · Next.js · Tailwind · Compose"),
    ("Core.Backend", "FastAPI · Node.js · Express · Flask"),
    ("Core.Database", "MySQL · Redis · MongoDB · SQLite"),
    ("Core.Infra", "Docker · Linux · AWS S3 · CI/CD"),
    ("Grid.Mail", "vermadivyansh703@gmail.com"),
    ("Grid.LinkedIn", "/in/divyanshverma153v"),
    ("Grid.GitHub", "adamkoji"),
    ("Grid.LeetCode", "divyansh153"),
]

# ==============================================================================
# THEME PALETTES
# ==============================================================================
THEMES = {
    "dark": {
        "filename": "banner-dark.v9.svg",
        "bg": "#090d16",
        "border": "#1a2638",
        "title_gray": "#64748b",
        "accent_cyan": "#38bdf8",
        "dot_color": "#aa9bef",       # Lilac purple particle matrix
        "text_label": "#718096",
        "text_val": "#f1f5f9",
        "pill_bg": "#0e2a3a",
        "pill_text": "#38bdf8",
        "status_green": "#10b981",
        "live_red": "#ef4444"
    },
    "light": {
        "filename": "banner-light.v9.svg",
        "bg": "#f8fafc",
        "border": "#cbd5e1",
        "title_gray": "#64748b",
        "accent_cyan": "#0284c7",
        "dot_color": "#6366f1",       # Indigo particle matrix for light mode
        "text_label": "#475569",
        "text_val": "#0f172a",
        "pill_bg": "#e0f2fe",
        "pill_text": "#0284c7",
        "status_green": "#059669",
        "live_red": "#dc2626"
    }
}

# ==============================================================================
# SERPENTINE FLOYD-STEINBERG DITHERING
# ==============================================================================
def dither_serpentine(image: Image.Image, target_w: int = 150, target_h: int = 180) -> list[tuple[float, float]]:
    """Converts input image into 2D coordinates using Serpentine Floyd-Steinberg dithering."""
    # Enhance contrast and resize
    img = image.convert("L")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.35)
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    pixels = np.array(img, dtype=float)
    h, w = pixels.shape
    dots = []

    for y in range(h):
        # Serpentine order: alternate scanning direction per row
        row_range = range(w) if (y % 2 == 0) else range(w - 1, -1, -1)
        direction = 1 if (y % 2 == 0) else -1
        
        for x in row_range:
            old_val = pixels[y, x]
            new_val = 255 if old_val > 128 else 0
            pixels[y, x] = new_val
            err = old_val - new_val

            # Distribute error to neighboring pixels
            if 0 <= x + direction < w:
                pixels[y, x + direction] += err * (7.0 / 16.0)
            if y + 1 < h:
                if 0 <= x - direction < w:
                    pixels[y + 1, x - direction] += err * (3.0 / 16.0)
                pixels[y + 1, x] += err * (5.0 / 16.0)
                if 0 <= x + direction < w:
                    pixels[y + 1, x + direction] += err * (1.0 / 16.0)

            # Keep dark pixels (the subject portrait)
            if new_val == 0:
                dots.append((x, y))
                
    return dots

# ==============================================================================
# SVG BUILDER
# ==============================================================================
def render_banner(theme_key: str):
    cfg = THEMES[theme_key]
    out_path = ASSETS / cfg["filename"]

    # Process photo dots
    dot_elements = []
    pt_count = 0
    box_w, box_h = 300, 360
    left_offset_x = 75
    left_offset_y = 125

    if SOURCE.exists():
        raw_img = Image.open(SOURCE)
        coords = dither_serpentine(raw_img, target_w=150, target_h=190)
        pt_count = len(coords)

        for i, (gx, gy) in enumerate(coords):
            # Map grid to coordinates inside VISUAL.MAP
            px = left_offset_x + gx * 1.85
            py = left_offset_y + gy * 1.85
            
            # Subtle random opacity animation delay to create a holographic live shimmer
            delay = round(random.uniform(0.0, LOOP_SECONDS), 2)
            duration = round(random.uniform(2.5, 4.5), 2)
            
            dot_elements.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.1" fill="{cfg["dot_color"]}" class="shimmer" '
                f'style="animation-delay: -{delay}s; animation-duration: {duration}s;"/>'
            )
    else:
        print(f"Warning: {SOURCE} not found. Generated without portrait.")

    rendered_dots = "\n        ".join(dot_elements)

    # Build SYSTEM.INFO list
    system_rows = []
    start_y = 158
    row_gap = 21

    for idx, (label, val) in enumerate(SYSTEM_DATA):
        y_pos = start_y + idx * row_gap
        label_escaped = html.escape(label)
        val_escaped = html.escape(val)
        
        system_rows.append(f'''
        <text x="500" y="{y_pos}" fill="{cfg["text_label"]}" font-size="12">{label_escaped}</text>
        <line x1="610" y1="{y_pos - 4}" x2="660" y2="{y_pos - 4}" stroke="{cfg["border"]}" stroke-dasharray="2,3" />
        <text x="1100" y="{y_pos}" text-anchor="end" fill="{cfg["text_val"]}" font-size="12" font-weight="600">{val_escaped}</text>
        ''')

    rendered_rows = "\n".join(system_rows)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">
    <defs>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&amp;display=swap');
            text {{ font-family: 'JetBrains Mono', monospace; }}
            
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.25; }}
            }}
            @keyframes shimmer {{
                0%, 100% {{ opacity: 0.85; }}
                50% {{ opacity: 0.35; }}
            }}
            .live-dot {{ animation: pulse 2s infinite ease-in-out; }}
            .shimmer {{ animation: shimmer ease-in-out infinite; }}
        </style>
    </defs>

    <!-- Outer Terminal Shell -->
    <rect x="12" y="12" width="{W - 24}" height="{H - 24}" rx="16" fill="{cfg["bg"]}" stroke="{cfg["border"]}" stroke-width="1.6"/>

    <!-- Traffic Lights -->
    <circle cx="42" cy="40" r="5.5" fill="#ef4444" />
    <circle cx="60" cy="40" r="5.5" fill="#f59e0b" />
    <circle cx="78" cy="40" r="5.5" fill="#10b981" />
    <text x="{W / 2}" y="44" text-anchor="middle" fill="{cfg["title_gray"]}" font-size="13" font-weight="500">profile.sh --live</text>

    <!-- ================= LEFT PANEL: VISUAL.MAP ================= -->
    <rect x="42" y="75" width="410" height="500" rx="10" fill="none" stroke="{cfg["border"]}" stroke-width="1.2"/>
    <text x="65" y="105" fill="{cfg["accent_cyan"]}" font-size="13" font-weight="700">VISUAL.MAP</text>
    <text x="430" y="105" text-anchor="end" fill="{cfg["title_gray"]}" font-size="11">300×340 / 1-BIT</text>

    <!-- Corner Brackets -->
    <path d="M 65 130 L 65 118 L 77 118" fill="none" stroke="{cfg["border"]}" stroke-width="1.8"/>
    <path d="M 430 130 L 430 118 L 418 118" fill="none" stroke="{cfg["border"]}" stroke-width="1.8"/>
    <path d="M 65 525 L 65 537 L 77 537" fill="none" stroke="{cfg["border"]}" stroke-width="1.8"/>
    <path d="M 430 525 L 430 537 L 418 537" fill="none" stroke="{cfg["border"]}" stroke-width="1.8"/>

    <!-- Dithered Portrait Points -->
    <g>
        {rendered_dots}
    </g>

    <text x="65" y="555" fill="{cfg["title_gray"]}" font-size="11">PTS {pt_count if pt_count else 18000} · FS/SERPENTINE</text>

    <!-- ================= RIGHT PANEL: SYSTEM.INFO ================= -->
    <rect x="475" y="75" width="663" height="500" rx="10" fill="none" stroke="{cfg["border"]}" stroke-width="1.2"/>
    <text x="500" y="105" fill="{cfg["accent_cyan"]}" font-size="13" font-weight="700">SYSTEM.INFO</text>

    <!-- LIVE Indicator -->
    <circle cx="980" cy="101" r="4.5" fill="{cfg["live_red"]}" class="live-dot"/>
    <text x="993" y="105" fill="{cfg["live_red"]}" font-size="12" font-weight="700">LIVE</text>
    <rect x="1035" y="90" width="85" height="24" rx="12" fill="{cfg["pill_bg"]}"/>
    <text x="1077" y="106" text-anchor="middle" fill="{cfg["pill_text"]}" font-size="11" font-weight="700">@adamkoji</text>

    <!-- System Rows -->
    {rendered_rows}

    <!-- Panel Footer -->
    <line x1="500" y1="535" x2="1100" y2="535" stroke="{cfg["border"]}" stroke-width="1"/>
    <circle cx="508" cy="554" r="4" fill="{cfg["status_green"]}"/>
    <text x="520" y="558" fill="{cfg["status_green"]}" font-size="11" font-weight="700">ALL SYSTEMS NOMINAL</text>
    <text x="1100" y="558" text-anchor="end" fill="{cfg["title_gray"]}" font-size="11">UTC+5:30 · NCR NODE</text>
</svg>'''

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Generated: {out_path.relative_to(ROOT)}")

def main():
    print("Generating profile banners...")
    render_banner("dark")
    render_banner("light")
    print("Done! Both banner-dark.v9.svg and banner-light.v9.svg are ready.")

if __name__ == "__main__":
    main()