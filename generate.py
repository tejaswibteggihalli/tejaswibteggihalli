#!/usr/bin/env python3
"""Mobius contribution ribbon.

Each cell is one day: 7 rows across the strip (Mon..Sun), one column per week
around the loop. A small long-legged walker walks the centreline, two laps
(a Mobius strip is one-sided, so he ends up back where he began).

usage: python generate.py daily.json out_dir
daily.json = {"YYYY-MM-DD": commit_count, ...}
"""
import json, math, sys, os, colorsys, datetime as dt
import numpy as np

W, H = 1200, 400
CX, CY = 600, 196
SCALE = 232
R, HALF_W = 1.0, 0.36
ROWS = 7
ELEV = math.radians(31)
YAW = math.radians(-24)
START = dt.date(2025, 9, 1)  # a Monday
TODAY = dt.date(2026, 11, 9)
LAP_SECONDS = 16
FACE_MIN = 0.55

THEMES = {
    "light": dict(
        empty="#E4DFEE", lv=["#8FD8CC", "#4DB6C9", "#8B6FD8", "#E0468E"],
        ink="#2A2540", eye="#FFFFFF", text="#59636e", faint="#9198a1",
        hi=1.10, lo=0.80),
    "dark": dict(
        empty="#2C2946", lv=["#7BD3C6", "#4FBBD0", "#9A80E6", "#F0589F"],
        ink="#EDE9FA", eye="#2A2540", text="#9198a1", faint="#59636e",
        hi=1.06, lo=0.70),
}


def mobius(u, v):
    c = R + v * np.cos(u / 2)
    return np.array([c * np.cos(u), c * np.sin(u), v * np.sin(u / 2)])


def cam(p):
    """world -> (sx, sy, depth toward viewer)"""
    cy, sy_ = math.cos(YAW), math.sin(YAW)
    x = p[0] * cy - p[1] * sy_
    y = p[0] * sy_ + p[1] * cy
    z = p[2]
    sx = x
    sy = -(y * math.sin(ELEV) + z * math.cos(ELEV))
    d = -y * math.cos(ELEV) + z * math.sin(ELEV)
    return np.array([CX + SCALE * sx, CY + SCALE * sy, d])


def cam_vec(n):
    cy, sy_ = math.cos(YAW), math.sin(YAW)
    x = n[0] * cy - n[1] * sy_
    y = n[0] * sy_ + n[1] * cy
    return np.array([x, y, n[2]])


VIEW = np.array([0, -math.cos(ELEV), math.sin(ELEV)])
UPV = np.array([0, math.sin(ELEV), math.cos(ELEV)])
LIGHT = np.array([-0.45, -0.35, 0.82])
LIGHT /= np.linalg.norm(LIGHT)


def level(n):
    return 0 if n == 0 else 1 if n <= 2 else 2 if n <= 5 else 3 if n <= 12 else 4


def tint(hex_, f):
    r, g, b = (int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(0.95, max(0.05, l * f))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def build(daily, theme):
    T = THEMES[theme]
    ncols = ((TODAY - START).days // 7) + 1
    quads = []
    for i in range(ncols):
        u0, u1 = 2 * math.pi * i / ncols, 2 * math.pi * (i + 1) / ncols
        for j in range(ROWS):
            day = START + dt.timedelta(days=7 * i + j)
            if day > TODAY:
                continue
            v0 = -HALF_W + 2 * HALF_W * j / ROWS
            v1 = -HALF_W + 2 * HALF_W * (j + 1) / ROWS
            pts3 = [mobius(u0, v0), mobius(u1, v0), mobius(u1, v1), mobius(u0, v1)]
            n = np.cross(pts3[1] - pts3[0], pts3[3] - pts3[0])
            n = cam_vec(n / np.linalg.norm(n))
            if n @ VIEW < 0:
                n = -n
            shade = float(n @ LIGHT)  # -1..1
            f = T["lo"] + (T["hi"] - T["lo"]) * (shade + 0.2) / 1.2
            f = min(T["hi"], max(T["lo"], f))
            pts = [cam(p) for p in pts3]
            depth = sum(p[2] for p in pts) / 4
            c = np.mean([p[:2] for p in pts], axis=0)
            poly = [(c + (p[:2] - c) * 0.92) for p in pts]  # tile gap
            lv = level(daily.get(day.isoformat(), 0))
            base = T["empty"] if lv == 0 else T["lv"][lv - 1]
            quads.append((depth, poly, tint(base, f)))
    quads.sort(key=lambda q: q[0])

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-label="Commit activity from September 2025 to November 2026, drawn as a Mobius strip with one cell per day">']
    for _, poly, col in quads:
        d = "M" + "L".join(f"{p[0]:.1f},{p[1]:.1f}" for p in poly) + "Z"
        out.append(f'<path d="{d}" fill="{col}"/>')

    # ---- walker: paces back and forth along the stretch of strip facing the viewer ----
    def state(u):
        P = mobius(u, 0)
        e = 1e-4
        pu = (mobius(u + e, 0) - mobius(u - e, 0)) / (2 * e)
        pv = (mobius(u, e) - mobius(u, -e)) / (2 * e)
        n = cam_vec(np.cross(pu, pv))
        n /= np.linalg.norm(n)
        return P, n, float(n @ VIEW)

    us = np.linspace(0, 2 * math.pi, 2000, endpoint=False)
    ok = np.array([state(u)[2] > FACE_MIN and cam(mobius(u, 0))[1] > 0 for u in us])
    # longest contiguous run (circular) of "facing the viewer"
    best, run_start, bl, bs = 0, None, 0, 0
    doubled = np.concatenate([ok, ok])
    for i, o in enumerate(doubled):
        if o:
            if run_start is None: run_start = i
            if i - run_start + 1 > bl and i - run_start + 1 <= len(ok): bl, bs = i - run_start + 1, run_start
        else:
            run_start = None
    ua, ub = us[bs % len(us)], us[bs % len(us)] + bl * (us[1] - us[0])
    ua += 0.22; ub -= 0.10
    K = 300
    pos, ang = [], []
    for k in range(K + 1):
        ph = k / K  # 0..1: there and back
        tri = 0.5 - 0.5 * math.cos(2 * math.pi * ph)  # eased 0->1->0
        u = ua + (ub - ua) * tri
        P, n, face = state(u)
        s_ = cam(P)
        ns = np.array([n[0], -(n @ UPV)])
        v = ns + np.array([0, -0.7])
        a = 0.55 * math.degrees(math.atan2(v[0], -v[1]))
        pos.append((s_[0], s_[1]))
        ang.append(a)
    for k in range(1, len(ang)):
        while ang[k] - ang[k - 1] > 180: ang[k] -= 360
        while ang[k] - ang[k - 1] < -180: ang[k] += 360
    dur = LAP_SECONDS
    kt = ";".join(f"{k / K:.4f}" for k in range(K + 1))
    tv = ";".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pos)
    av = ";".join(f"{a:.1f}" for a in ang)
    ink, eye = T["ink"], T["eye"]
    out.append(f'''<g><animateTransform attributeName="transform" type="translate" values="{tv}" keyTimes="{kt}" dur="{dur}s" repeatCount="indefinite"/>
<g><animateTransform attributeName="transform" type="rotate" values="{av}" keyTimes="{kt}" dur="{dur}s" repeatCount="indefinite"/>
<g transform="scale(1.35)">
<g fill="none" stroke="{ink}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
<path d="M-2.6,-19 L-5.6,-10 L-3.2,0"><animateTransform attributeName="transform" type="rotate" values="-22 -2.6 -19;22 -2.6 -19;-22 -2.6 -19" dur="0.9s" repeatCount="indefinite"/></path>
<path d="M2.6,-19 L5.6,-10 L3.2,0"><animateTransform attributeName="transform" type="rotate" values="22 2.6 -19;-22 2.6 -19;22 2.6 -19" dur="0.9s" repeatCount="indefinite"/></path>
</g>
<rect x="-6" y="-31" width="12" height="12" rx="1.5" fill="{ink}"/>
<circle cx="-2.2" cy="-26" r="1.25" fill="{eye}"/><circle cx="2.2" cy="-26" r="1.25" fill="{eye}"/>
</g></g></g>''')

    # ---- legend + caption ----
    lx, ly = W - 40, H - 26
    out.append(f'<g font-family="\'Google Sans\',\'Product Sans\',\'Segoe UI\',system-ui,-apple-system,sans-serif" font-size="13">')
    n_c = sum(daily.values())
    n_d = len(daily)
    out.append(f'<text x="40" y="{ly + 4}" fill="{T["text"]}">{n_c} commits on {n_d} days · Sep 2025 to Nov 2026</text>')
    out.append(f'<text x="{lx - 148}" y="{ly + 4}" text-anchor="end" fill="{T["faint"]}">Less</text>')
    for i, col in enumerate([T["empty"]] + T["lv"]):
        out.append(f'<rect x="{lx - 140 + i * 20}" y="{ly - 6}" width="14" height="14" rx="3" fill="{col}"/>')
    out.append(f'<text x="{lx - 140 + 5 * 20 + 6}" y="{ly + 4}" fill="{T["faint"]}">More</text>')
    out.append('</g></svg>')
    return "\n".join(out)


if __name__ == "__main__":
    daily = json.load(open(sys.argv[1]))
    od = sys.argv[2]
    os.makedirs(od, exist_ok=True)
    for th in THEMES:
        open(os.path.join(od, f"ribbon-{th}.svg"), "w").write(build(daily, th))
        print("wrote", th)
