import os
import re
import json
import base64
import secrets
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
from functools import lru_cache

import streamlit as st


# ==========================================================
# CONFIG + CSS (MOBILE PRO SPORT UI - GIF BACKGROUND OPTIONAL)
# ==========================================================
st.set_page_config(page_title="TennisStats", page_icon="🎾", layout="centered")


def _read_gif_data_uri():
    """
    Busca primero el GIF definitivo (tennis_ball_slowmo.gif).
    Fallback: tennis_bg.gif (por compatibilidad).
    """
    candidates = [
        Path("assets/tennis_ball_slowmo.gif"),
        Path("assets/tennis_bg.gif"),
        Path("tennis_ball_slowmo.gif"),
        Path("tennis_bg.gif"),
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                b = p.read_bytes()
                b64 = base64.b64encode(b).decode("utf-8")
                return f"data:image/gif;base64,{b64}"
        except Exception:
            continue
    return ""


BG_GIF = _read_gif_data_uri()

PRO_CSS = f"""
<style>
:root{{
  --bg:#f6f8fb;
  --bg2:#eef2f7;
  --card: rgba(255,255,255,0.86);
  --text:#0b1220;
  --muted:#475569;
  --muted2:#64748b;
  --stroke: rgba(2,6,23,0.10);
  --stroke2: rgba(2,6,23,0.06);

  --accent:#16a34a;      /* green */
  --accent2:#2563eb;     /* blue */
  --danger:#dc2626;      /* red */
  --warn:#f59e0b;        /* amber */

  --radius: 18px;
  --shadow: 0 18px 40px rgba(2,6,23,.10);
  --shadow2: 0 10px 22px rgba(2,6,23,.08);
  --focus: 0 0 0 3px rgba(37,99,235,.16);
}}

* {{
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}}

html, body, [data-testid="stAppViewContainer"]{{
  background:
    radial-gradient(900px 380px at 15% -5%, rgba(22,163,74,.14), transparent 60%),
    radial-gradient(850px 360px at 85% 0%, rgba(37,99,235,.12), transparent 60%),
    linear-gradient(180deg, var(--bg), var(--bg2));
  color: var(--text);
}}

/* Subtle grid overlay */
[data-testid="stAppViewContainer"]::before{{
  content:"";
  position: fixed;
  inset: 0;
  pointer-events:none;
  opacity: .22;
  background:
    linear-gradient(90deg, rgba(2,6,23,.028) 1px, transparent 1px) 0 0 / 160px 160px,
    linear-gradient(0deg, rgba(2,6,23,.024) 1px, transparent 1px) 0 0 / 160px 160px;
  mask-image: radial-gradient(circle at 50% 0%, black 22%, transparent 62%);
  z-index: 0;
}}

{"" if not BG_GIF else f"""
/* Optional Tennis GIF background layer */
[data-testid="stAppViewContainer"]::after{{
  content:"";
  position: fixed;
  inset: -12%;
  background-image: url("{BG_GIF}");
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  opacity: 0.12;              /* subtle so it doesn't hurt readability */
  filter: saturate(1.05) contrast(1.06);
  pointer-events: none;
  z-index: -1;
}}
"""}

/* Keep app content above overlays */
.block-container, header, section, footer {{
  position: relative;
  z-index: 1;
}}

/* ===== Mobile rhythm / spacing ===== */
.block-container{{
  padding-top: .62rem !important;
  padding-bottom: 1.05rem !important;
  max-width: 980px;
}}
div[data-testid="stVerticalBlock"] > div {{ gap: .45rem !important; }}
header[data-testid="stHeader"]{{ height: 0.35rem; background: transparent; }}

h1,h2,h3{{ letter-spacing: .2px; }}
h2{{ margin-bottom: .15rem !important; }}
h3{{ margin-top: .35rem !important; margin-bottom: .20rem !important; }}

.stMarkdown p {{ margin-bottom: .35rem !important; }}
.stCaption, [data-testid="stCaptionContainer"]{{ color: var(--muted2) !important; }}

.small-note{{ color: var(--muted); font-size: .92rem; line-height: 1.25rem; }}
.kpi{{ font-size: 1.02rem; font-weight: 950; }}
.mono{{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}

hr, [data-testid="stDivider"]{{
  border-color: var(--stroke2) !important;
  margin: 0.40rem 0 !important;
}}

/* ===== Inputs: less "boxy" (remove redundant casillas) ===== */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div{{
  background: rgba(255,255,255,0.72) !important;
  border: 1px solid rgba(2,6,23,0.08) !important;
  border-radius: 16px !important;
  box-shadow: none !important;
  backdrop-filter: blur(6px);
}}

div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea{{ color: var(--text) !important; }}

label, .stTextInput label, .stSelectbox label, .stNumberInput label{{
  color: var(--muted2) !important;
  font-weight: 850 !important;
}}

div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within{{
  outline: none !important;
  box-shadow: var(--focus) !important;
  border-color: rgba(37,99,235,.28) !important;
}}

/* ===== Buttons: sporty + cleaner ===== */
.stButton>button{{
  width: 100%;
  padding: 0.56rem 0.92rem;
  border-radius: 14px;
  border: 1px solid rgba(2,6,23,0.10) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,248,252,0.90));
  color: var(--text);
  font-weight: 980;
  box-shadow: 0 10px 18px rgba(2,6,23,.07) !important;
  transition: transform .08s ease, box-shadow .14s ease, border-color .14s ease, filter .14s ease;
}}
.stButton>button:hover{{
  border-color: rgba(22,163,74,.28) !important;
  box-shadow: 0 14px 26px rgba(2,6,23,.10) !important;
  transform: translateY(-1px);
}}
.stButton>button:active{{ transform: translateY(0px) scale(0.99); }}
.stButton>button:focus{{ outline: none !important; box-shadow: 0 10px 18px rgba(2,6,23,.07), var(--focus) !important; }}

[data-testid="stDownloadButton"] > button{{
  border-radius: 14px !important;
  border: 1px solid rgba(37,99,235,.18) !important;
  background: linear-gradient(180deg, rgba(37,99,235,.14), rgba(255,255,255,0.98)) !important;
  color: var(--text) !important;
  font-weight: 980 !important;
  box-shadow: 0 10px 18px rgba(2,6,23,.07) !important;
}}

/* ===== Expanders ===== */
[data-testid="stExpander"]{{
  border: 1px solid rgba(2,6,23,0.08) !important;
  border-radius: var(--radius) !important;
  background: rgba(255,255,255,0.78) !important;
  box-shadow: 0 12px 26px rgba(2,6,23,.08);
  overflow: hidden;
}}
[data-testid="stExpander"] summary{{ font-weight: 980 !important; }}
[data-testid="stExpander"] details {{ padding-bottom: 0 !important; }}

/* ===== Tabs ===== */
[data-baseweb="tab-list"]{{
  background: rgba(255,255,255,0.66);
  border: 1px solid rgba(2,6,23,0.08);
  border-radius: 16px;
  padding: 6px;
  gap: 6px;
  box-shadow: 0 10px 16px rgba(2,6,23,.07);
}}
button[role="tab"]{{
  border-radius: 12px !important;
  font-weight: 980 !important;
  color: var(--muted) !important;
}}
button[role="tab"][aria-selected="true"]{{
  background: linear-gradient(180deg, rgba(22,163,74,.14), rgba(255,255,255,.92)) !important;
  color: var(--text) !important;
  border: 1px solid rgba(22,163,74,.20) !important;
}}

/* ===== Alerts / uploader ===== */
[data-testid="stAlert"]{{
  border-radius: 16px !important;
  border: 1px solid rgba(2,6,23,0.08) !important;
  background: rgba(255,255,255,0.78) !important;
  box-shadow: 0 10px 16px rgba(2,6,23,.07);
}}
section[data-testid="stFileUploaderDropzone"]{{
  border-radius: 16px !important;
  border: 1px dashed rgba(2,6,23,0.20) !important;
  background: rgba(255,255,255,0.60) !important;
  box-shadow: 0 10px 16px rgba(2,6,23,.07);
}}

/* ===== Header cards ===== */
.ts-header{{
  border: 1px solid rgba(2,6,23,0.08);
  border-radius: 22px;
  padding: 14px 16px;
  background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(255,255,255,0.62));
  box-shadow: var(--shadow2);
  position: relative;
  overflow: hidden;
}}
.ts-header::after{{
  content:"";
  position:absolute;
  inset:-60% -20% auto auto;
  width: 520px; height: 320px;
  background: radial-gradient(circle at 30% 40%, rgba(37,99,235,.16), transparent 62%);
  transform: rotate(12deg);
}}
.ts-title{{ font-size: 1.18rem; font-weight: 1000; margin: 0; }}
.ts-sub{{ margin: 4px 0 0 0; color: var(--muted); font-weight: 750; font-size: .92rem; }}
.ts-badges{{ margin-top: 10px; display:flex; flex-wrap:wrap; gap: 8px; }}
.ts-badge{{
  display:inline-flex; align-items:center; gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(2,6,23,0.08);
  background: rgba(255,255,255,0.62);
  font-weight: 950; font-size: .88rem; color: var(--text);
}}
.ts-dot{{
  width: 9px; height: 9px; border-radius: 999px; background: var(--accent);
  box-shadow: 0 0 0 3px rgba(22,163,74,.14);
}}

/* ===== Cards ===== */
.ts-card{{
  border: 1px solid rgba(2,6,23,0.08);
  border-radius: 18px;
  background: rgba(255,255,255,0.78);
  box-shadow: 0 10px 16px rgba(2,6,23,.07);
  padding: 12px 12px;
  backdrop-filter: blur(6px);
}}
.ts-card > div[data-testid="stVerticalBlock"] {{ gap: .35rem !important; }}
.ts-card .stMarkdown {{ margin: 0 !important; }}

/* ===== Donut rings ===== */
.ring-wrap{{ display:flex; gap: 12px; align-items:center; }}
.ring{{
  width: 58px; height: 58px; border-radius: 999px;
  background: conic-gradient(var(--ringc) var(--deg), rgba(2,6,23,.08) 0);
  position: relative; box-shadow: 0 10px 16px rgba(2,6,23,.07);
}}
.ring::after{{
  content:""; position:absolute; inset: 8px; border-radius: 999px;
  background: rgba(255,255,255,0.88);
  border: 1px solid rgba(2,6,23,.05);
}}
.ring-val{{
  position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-weight: 1000; color: var(--text); font-size: .92rem; z-index: 2;
}}
.ring-txt .t1{{ font-weight: 1000; line-height: 1.05rem; }}
.ring-txt .t2{{ color: var(--muted); font-weight: 800; font-size: .88rem; margin-top: 2px; }}

/* ===== Score pills ===== */
.pills{{ display:flex; gap: 8px; flex-wrap:wrap; margin-top:8px; }}
.pill{{
  display:inline-flex; align-items:center; gap:8px;
  padding: 7px 10px; border-radius: 999px;
  border: 1px solid rgba(2,6,23,0.08);
  background: rgba(255,255,255,0.62);
  font-weight: 950; font-size:.90rem;
}}
.pill b{{ font-weight:1000; }}

/* ===== Last points timeline ===== */
.lp{{ display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; }}
.dot{{
  width: 14px; height: 14px; border-radius: 999px;
  border: 1px solid rgba(2,6,23,.14);
  box-shadow: 0 6px 10px rgba(2,6,23,.06);
}}
.dot.win{{ background: rgba(22,163,74,.95); }}
.dot.lose{{ background: rgba(220,38,38,.90); }}
.dot.pressure{{ outline: 3px solid rgba(245,158,11,.20); }}

/* ===== Segmented nav ===== */
div[data-testid="stSegmentedControl"] > div{{
  border-radius: 16px !important;
  border: 1px solid rgba(2,6,23,0.08) !important;
  background: rgba(255,255,255,0.66) !important;
  box-shadow: 0 10px 16px rgba(2,6,23,.07) !important;
  padding: 6px !important;
}}
div[data-testid="stSegmentedControl"] label{{ font-weight: 950 !important; }}

</style>
"""
st.markdown(PRO_CSS, unsafe_allow_html=True)


# ==========================================================
# STORAGE (multi-usuario privado por fichero)
# ==========================================================
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
HIST_DIR = os.path.join(DATA_DIR, "histories")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HIST_DIR, exist_ok=True)


def safe_user_key(username: str) -> str:
    u = (username or "").strip().lower()
    u = re.sub(r"[^a-z0-9_\-\.]", "_", u)
    u = re.sub(r"_+", "_", u).strip("_")
    return u[:40] if u else ""


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def _b64d(s: str) -> bytes:
    s = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode("utf-8"))


def hash_pin(pin: str, salt_b: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt_b, 200_000)
    return _b64e(dk)


def load_users() -> dict:
    ensure_dirs()
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_users(users: dict) -> None:
    ensure_dirs()
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def history_path_for(user_key: str) -> str:
    ensure_dirs()
    return os.path.join(HIST_DIR, f"history__{user_key}.json")


def load_history_from_disk(user_key: str) -> list:
    path = history_path_for(user_key)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        matches = obj.get("matches", [])
        return matches if isinstance(matches, list) else []
    except Exception:
        return []


def save_history_to_disk(user_key: str, matches: list) -> None:
    ensure_dirs()
    path = history_path_for(user_key)
    payload = {"matches": matches}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ==========================================================
# NOTICIAS (RSS)
# ==========================================================
NEWS_SOURCES = [
    ("ATP Tour", "https://www.atptour.com/en/media/rss-feed/xml-feed"),
    ("WTA", "https://www.wtatennis.com/rss"),
    ("ITF", "https://www.itftennis.com/en/news/rss/"),
    ("BBC Tennis", "https://feeds.bbci.co.uk/sport/tennis/rss.xml"),
]


def _first_text(elem, tags):
    for t in tags:
        x = elem.find(t)
        if x is not None and x.text:
            return x.text.strip()
    return ""


def _attr(elem, tag, attr):
    x = elem.find(tag)
    if x is not None and x.attrib.get(attr):
        return x.attrib.get(attr)
    return ""


@st.cache_data(ttl=900, show_spinner=False)
def fetch_tennis_news(max_items: int = 15):
    items = []
    for source_name, url in NEWS_SOURCES:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Streamlit TennisStats)"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = resp.read()

            root = ET.fromstring(data)
            channel = root.find("channel")
            if channel is not None:
                for it in channel.findall("item"):
                    title = _first_text(it, ["title"])
                    link = _first_text(it, ["link"])
                    pub = _first_text(it, ["pubDate", "{http://purl.org/dc/elements/1.1/}date"])
                    if title and link:
                        items.append({"source": source_name, "title": title, "link": link, "published": pub})
                continue

            if root.tag.endswith("feed"):
                ns = {"a": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("a:entry", ns):
                    title = _first_text(entry, ["{http://www.w3.org/2005/Atom}title"])
                    link = _attr(entry, "{http://www.w3.org/2005/Atom}link", "href")
                    pub = _first_text(entry, ["{http://www.w3.org/2005/Atom}updated", "{http://www.w3.org/2005/Atom}published"])
                    if title and link:
                        items.append({"source": source_name, "title": title, "link": link, "published": pub})
        except Exception:
            continue

    seen = set()
    uniq = []
    for it in items:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        uniq.append(it)

    return uniq[:max_items]


# ==========================================================
# LÓGICA TENIS (MARCADOR)
# ==========================================================
POINT_LABELS = {0: "0", 1: "15", 2: "30", 3: "40"}


def game_point_label(p_me: int, p_opp: int) -> str:
    if p_me >= 3 and p_opp >= 3:
        if p_me == p_opp:
            return "40-40"
        if p_me == p_opp + 1:
            return "AD-40"
        if p_opp == p_me + 1:
            return "40-AD"
    return f"{POINT_LABELS.get(p_me, '40')}-{POINT_LABELS.get(p_opp, '40')}"


def won_game(p_me: int, p_opp: int) -> bool:
    return p_me >= 4 and (p_me - p_opp) >= 2


def won_tiebreak(p_me: int, p_opp: int) -> bool:
    return p_me >= 7 and (p_me - p_opp) >= 2


def is_set_over(g_me: int, g_opp: int) -> bool:
    if g_me >= 6 and (g_me - g_opp) >= 2:
        return True
    if g_me == 7 and g_opp == 6:
        return True
    return False


# ==========================================================
# MODELO REAL: Markov (punto→juego→set→BO3)
# ==========================================================
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@lru_cache(maxsize=None)
def _prob_game_from(p_rounded: float, a: int, b: int) -> float:
    p = max(1e-6, min(1 - 1e-6, float(p_rounded)))
    q = 1.0 - p

    if a >= 4 and a - b >= 2:
        return 1.0
    if b >= 4 and b - a >= 2:
        return 0.0

    if a >= 3 and b >= 3:
        deuce = (p * p) / (p * p + q * q)
        if a == b:
            return deuce
        if a == b + 1:
            return p * 1.0 + q * deuce
        if b == a + 1:
            return p * deuce + q * 0.0
        return deuce

    return p * _prob_game_from(p_rounded, a + 1, b) + q * _prob_game_from(p_rounded, a, b + 1)


@lru_cache(maxsize=None)
def _prob_tiebreak_from(p_rounded: float, a: int, b: int) -> float:
    p = max(1e-6, min(1 - 1e-6, float(p_rounded)))
    q = 1.0 - p

    if a >= 7 and a - b >= 2:
        return 1.0
    if b >= 7 and b - a >= 2:
        return 0.0

    if a >= 6 and b >= 6:
        deuce = (p * p) / (p * p + q * q)
        if a == b:
            return deuce
        if a == b + 1:
            return p * 1.0 + q * deuce
        if b == a + 1:
            return p * deuce + q * 0.0
        return deuce

    return p * _prob_tiebreak_from(p_rounded, a + 1, b) + q * _prob_tiebreak_from(p_rounded, a, b + 1)


@lru_cache(maxsize=None)
def _prob_set_from(p_rounded: float, g_me: int, g_opp: int, pts_me: int, pts_opp: int, in_tb: bool) -> float:
    if is_set_over(g_me, g_opp):
        return 1.0
    if is_set_over(g_opp, g_me):
        return 0.0

    if in_tb:
        return _prob_tiebreak_from(p_rounded, pts_me, pts_opp)

    p_game = _prob_game_from(p_rounded, pts_me, pts_opp)

    def after_game(next_g_me, next_g_opp):
        if next_g_me == 6 and next_g_opp == 6:
            return _prob_set_from(p_rounded, 6, 6, 0, 0, True)
        return _prob_set_from(p_rounded, next_g_me, next_g_opp, 0, 0, False)

    return p_game * after_game(g_me + 1, g_opp) + (1 - p_game) * after_game(g_me, g_opp + 1)


@lru_cache(maxsize=None)
def _prob_match_bo3(p_rounded: float, sets_me: int, sets_opp: int, g_me: int, g_opp: int, pts_me: int, pts_opp: int, in_tb: bool) -> float:
    if sets_me >= 2:
        return 1.0
    if sets_opp >= 2:
        return 0.0

    p_set = _prob_set_from(p_rounded, g_me, g_opp, pts_me, pts_opp, in_tb)
    win_state = (p_rounded, sets_me + 1, sets_opp, 0, 0, 0, 0, False)
    lose_state = (p_rounded, sets_me, sets_opp + 1, 0, 0, 0, 0, False)
    return p_set * _prob_match_bo3(*win_state) + (1 - p_set) * _prob_match_bo3(*lose_state)


# ==========================================================
# ESTADO LIVE
# ==========================================================
@dataclass
class LiveState:
    sets_me: int = 0
    sets_opp: int = 0
    games_me: int = 0
    games_opp: int = 0
    pts_me: int = 0
    pts_opp: int = 0
    in_tiebreak: bool = False


class LiveMatch:
    def __init__(self):
        self.points = []
        self.state = LiveState()
        self.surface = "Tierra batida"
        self._undo = []

    def snapshot(self):
        self._undo.append((deepcopy(self.state), len(self.points), self.surface))

    def undo(self):
        if not self._undo:
            return
        st_, n, surf = self._undo.pop()
        self.state = st_
        self.points = self.points[:n]
        self.surface = surf

    def reset(self):
        self.points = []
        self.state = LiveState()
        self._undo = []

    def points_stats(self):
        total = len(self.points)
        won = sum(1 for p in self.points if p["result"] == "win")
        pct = (won / total * 100.0) if total else 0.0
        return total, won, pct

    def estimate_point_win_prob(self) -> float:
        n = len(self.points)
        w = sum(1 for p in self.points if p["result"] == "win")
        p = (w + 1) / (n + 2) if n >= 0 else 0.5
        return _clamp01(p)

    def match_win_prob(self) -> float:
        p = self.estimate_point_win_prob()
        p_r = round(p, 3)
        st_ = self.state
        return _prob_match_bo3(
            p_r, st_.sets_me, st_.sets_opp, st_.games_me, st_.games_opp, st_.pts_me, st_.pts_opp, st_.in_tiebreak
        )

    def win_prob_series(self):
        probs = []
        tmp = LiveMatch()
        tmp.surface = self.surface
        for p in self.points:
            tmp.add_point(p["result"], {"finish": p.get("finish")})
            probs.append(tmp.match_win_prob() * 100.0)
        return probs

    def _maybe_start_tiebreak(self):
        if self.state.games_me == 6 and self.state.games_opp == 6:
            self.state.in_tiebreak = True
            self.state.pts_me = 0
            self.state.pts_opp = 0

    def _award_game_to_me(self):
        self.state.games_me += 1
        self.state.pts_me = 0
        self.state.pts_opp = 0
        self.state.in_tiebreak = False
        self._maybe_start_tiebreak()
        self._maybe_award_set()

    def _award_game_to_opp(self):
        self.state.games_opp += 1
        self.state.pts_me = 0
        self.state.pts_opp = 0
        self.state.in_tiebreak = False
        self._maybe_start_tiebreak()
        self._maybe_award_set()

    def _maybe_award_set(self):
        if is_set_over(self.state.games_me, self.state.games_opp):
            self.state.sets_me += 1
            self.state.games_me = 0
            self.state.games_opp = 0
            self.state.pts_me = 0
            self.state.pts_opp = 0
            self.state.in_tiebreak = False
        elif is_set_over(self.state.games_opp, self.state.games_me):
            self.state.sets_opp += 1
            self.state.games_me = 0
            self.state.games_opp = 0
            self.state.pts_me = 0
            self.state.pts_opp = 0
            self.state.in_tiebreak = False

    def add_point(self, result: str, meta: dict):
        self.snapshot()
        before = deepcopy(self.state)
        set_idx = before.sets_me + before.sets_opp + 1
        is_pressure = bool(before.in_tiebreak or (before.pts_me >= 3 and before.pts_opp >= 3))

        self.points.append(
            {"result": result, **meta, "surface": self.surface, "before": before.__dict__, "set_idx": set_idx, "pressure": is_pressure}
        )

        if result == "win":
            self.state.pts_me += 1
        else:
            self.state.pts_opp += 1

        if self.state.in_tiebreak:
            if won_tiebreak(self.state.pts_me, self.state.pts_opp):
                self.state.games_me = 7
                self.state.games_opp = 6
                self._maybe_award_set()
            elif won_tiebreak(self.state.pts_opp, self.state.pts_me):
                self.state.games_opp = 7
                self.state.games_me = 6
                self._maybe_award_set()
            return

        if won_game(self.state.pts_me, self.state.pts_opp):
            self._award_game_to_me()
        elif won_game(self.state.pts_opp, self.state.pts_me):
            self._award_game_to_opp()

    def add_game_manual(self, who: str):
        self.snapshot()
        if who == "me":
            self._award_game_to_me()
        else:
            self._award_game_to_opp()

    def add_set_manual(self, who: str):
        self.snapshot()
        if who == "me":
            self.state.sets_me += 1
        else:
            self.state.sets_opp += 1
        self.state.games_me = 0
        self.state.games_opp = 0
        self.state.pts_me = 0
        self.state.pts_opp = 0
        self.state.in_tiebreak = False

    def match_summary(self):
        total = len(self.points)
        won = sum(1 for p in self.points if p["result"] == "win")
        pct = (won / total * 100.0) if total else 0.0

        finishes = {"winner": 0, "unforced": 0, "forced": 0, "ace": 0, "double_fault": 0, "opp_error": 0, "opp_winner": 0}
        pressure_total = sum(1 for p in self.points if p.get("pressure"))
        pressure_won = sum(1 for p in self.points if p.get("pressure") and p.get("result") == "win")

        for p in self.points:
            f = p.get("finish")
            if f in finishes:
                finishes[f] += 1

        return {
            "points_total": total,
            "points_won": won,
            "points_pct": pct,
            "pressure_total": pressure_total,
            "pressure_won": pressure_won,
            "pressure_pct": (pressure_won / pressure_total * 100.0) if pressure_total else 0.0,
            "finishes": finishes,
        }


# ==========================================================
# HISTORIAL
# ==========================================================
class MatchHistory:
    def __init__(self):
        self.matches = []

    def add(self, m: dict):
        self.matches.append(m)

    def filtered_matches(self, n=None, surface=None):
        arr = list(self.matches)
        if surface and surface != "Todas":
            arr = [m for m in arr if m.get("surface") == surface]
        if n is not None and n > 0:
            arr = arr[-n:]
        return arr

    def last_n_results(self, n=10, surface=None):
        matches = self.filtered_matches(n=n, surface=surface)
        return [("W" if m.get("won_match") else "L") for m in matches[-n:]]

    def best_streak(self, surface=None):
        matches = self.filtered_matches(n=None, surface=surface)
        best = 0
        cur = 0
        for m in matches:
            if m.get("won_match"):
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    @staticmethod
    def pct(wins, total):
        return (wins / total * 100.0) if total else 0.0

    def aggregate(self, n=None, surface=None):
        matches = self.filtered_matches(n=n, surface=surface)

        total_m = len(matches)
        win_m = sum(1 for m in matches if m.get("won_match"))

        sets_w = sum(int(m.get("sets_w", 0)) for m in matches)
        sets_l = sum(int(m.get("sets_l", 0)) for m in matches)
        games_w = sum(int(m.get("games_w", 0)) for m in matches)
        games_l = sum(int(m.get("games_l", 0)) for m in matches)

        surfaces = {}
        for m in matches:
            srf = m.get("surface", "Tierra batida")
            surfaces.setdefault(srf, {"w": 0, "t": 0})
            surfaces[srf]["t"] += 1
            if m.get("won_match"):
                surfaces[srf]["w"] += 1

        points_total = sum(int(m.get("points_total", 0)) for m in matches)
        points_won = sum(int(m.get("points_won", 0)) for m in matches)
        pressure_total = sum(int(m.get("pressure_total", 0)) for m in matches)
        pressure_won = sum(int(m.get("pressure_won", 0)) for m in matches)

        finishes_sum = {"winner": 0, "unforced": 0, "forced": 0, "ace": 0, "double_fault": 0, "opp_error": 0, "opp_winner": 0}
        for m in matches:
            fin = (m.get("finishes") or {})
            for k in finishes_sum:
                finishes_sum[k] += int(fin.get(k, 0) or 0)

        return {
            "matches_total": total_m,
            "matches_win": win_m,
            "matches_pct": self.pct(win_m, total_m),
            "sets_w": sets_w,
            "sets_l": sets_l,
            "sets_pct": self.pct(sets_w, sets_w + sets_l),
            "games_w": games_w,
            "games_l": games_l,
            "games_pct": self.pct(games_w, games_w + games_l),
            "points_total": points_total,
            "points_won": points_won,
            "points_pct": self.pct(points_won, points_total),
            "pressure_total": pressure_total,
            "pressure_won": pressure_won,
            "pressure_pct": self.pct(pressure_won, pressure_total),
            "finishes_sum": finishes_sum,
            "surfaces": surfaces,
        }


# ==========================================================
# Resumen tipo entrenador (original, se mantiene)
# ==========================================================
def coach_summary_from_match(m: dict) -> str:
    won = bool(m.get("won_match"))
    res = "Victoria" if won else "Derrota"

    pts_total = int(m.get("points_total", 0) or 0)
    pts_won = int(m.get("points_won", 0) or 0)
    pts_pct = float(m.get("points_pct", 0) or 0)

    pressure_total = int(m.get("pressure_total", 0) or 0)
    pressure_won = int(m.get("pressure_won", 0) or 0)
    pressure_pct = float(m.get("pressure_pct", 0) or 0)

    fin = (m.get("finishes") or {})
    winners = int(fin.get("winner", 0) or 0)
    enf = int(fin.get("unforced", 0) or 0)
    ef = int(fin.get("forced", 0) or 0)
    aces = int(fin.get("ace", 0) or 0)
    df = int(fin.get("double_fault", 0) or 0)
    opp_err = int(fin.get("opp_error", 0) or 0)

    strengths = []
    focus = []

    if pts_pct >= 55:
        strengths.append(f"dominaste el intercambio de puntos ({pts_pct:.0f}%).")
    elif pts_pct <= 45 and pts_total >= 10:
        focus.append(f"subir el % de puntos ganados ({pts_pct:.0f}%).")
    else:
        strengths.append(f"tu % de puntos estuvo equilibrado ({pts_pct:.0f}%).")

    if pressure_total >= 6:
        if pressure_pct >= 55:
            strengths.append(f"gestionaste muy bien la presión ({pressure_won}/{pressure_total}, {pressure_pct:.0f}%).")
        elif pressure_pct <= 45:
            focus.append(f"mejorar puntos de presión ({pressure_won}/{pressure_total}, {pressure_pct:.0f}%).")
        else:
            strengths.append(f"en presión estuviste parejo ({pressure_won}/{pressure_total}, {pressure_pct:.0f}%).")
    elif pressure_total > 0:
        strengths.append(f"en los pocos puntos de presión estuviste {pressure_won}/{pressure_total}.")
    else:
        strengths.append("hubo pocos puntos de presión registrados.")

    if winners >= max(5, enf + 2):
        strengths.append("generaste muchos winners y fuiste ofensivo cuando tocaba.")
    if enf >= max(5, winners + 2):
        focus.append("reducir errores no forzados (ENF) en momentos clave.")
    if df >= 3:
        focus.append("controlar dobles faltas (ritual de saque + margen).")
    if aces >= 3:
        strengths.append("el saque fue un arma (aces).")

    if (enf + df) > (winners + aces) and pts_total >= 15:
        focus.append("buscar más margen: altura/profundidad y seleccionar mejor el riesgo.")
    if opp_err >= 5 and winners < 3:
        strengths.append("sacaste puntos provocando error del rival: buena consistencia.")

    plan = []
    plan.append("1) Prioriza consistencia (altura/profundidad) y acelera solo con bola clara.")
    plan.append("2) En puntos importantes: rutina corta (respira, objetivo simple, juega al %).")
    plan.append("3) Saque: 1º con dirección; 2º con más efecto/altura, mismo ritual siempre.")

    s_txt = " ".join(strengths) if strengths else "buen partido en líneas generales."
    f_txt = " ".join(focus) if focus else "pocos puntos débiles claros: sigue consolidando lo que funcionó."

    return (
        f"**Resumen del entrenador ({res})**\n\n"
        f"- **Qué funcionó:** {s_txt}\n"
        f"- **Qué mejorar:** {f_txt}\n\n"
        f"**Claves para el próximo partido**\n"
        f"{plan[0]}\n{plan[1]}\n{plan[2]}\n\n"
        f"**Datos rápidos:** Puntos {pts_won}/{pts_total} ({pts_pct:.0f}%) · "
        f"Presión {pressure_won}/{pressure_total} ({pressure_pct:.0f}%) · "
        f"Winners {winners} · ENF {enf} · EF {ef} · Ace {aces} · DF {df} · ErrRival {opp_err}"
    )


# ==========================================================
# Resumen IA (usa OpenAI REST si hay OPENAI_API_KEY; si no, no rompe)
# ==========================================================
def ai_coach_summary_from_match(m: dict) -> str:
    """
    Generates an AI-style coaching summary. If OPENAI_API_KEY is available,
    calls OpenAI chat completions API via HTTPS. Otherwise returns a helpful message
    + fallback (original deterministic summary kept).
    """
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    base_summary = coach_summary_from_match(m)

    if not api_key:
        return (
            "⚠️ **Resumen IA no disponible** (falta `OPENAI_API_KEY`).\n\n"
            "Para activarlo, añade la key en `st.secrets` o variable de entorno.\n\n"
            "Mientras tanto, aquí tienes el resumen estándar:\n\n"
            + base_summary
        )

    fin = (m.get("finishes") or {})
    payload = {
        "resultado": "Victoria" if m.get("won_match") else "Derrota",
        "superficie": m.get("surface", "—"),
        "marcador_sets": f"{m.get('sets_w',0)}-{m.get('sets_l',0)}",
        "marcador_juegos": f"{m.get('games_w',0)}-{m.get('games_l',0)}",
        "puntos": f"{m.get('points_won',0)}/{m.get('points_total',0)} ({m.get('points_pct',0):.0f}%)",
        "presion": f"{m.get('pressure_won',0)}/{m.get('pressure_total',0)} ({m.get('pressure_pct',0):.0f}%)",
        "finishes": {
            "winners": int(fin.get("winner", 0) or 0),
            "enf": int(fin.get("unforced", 0) or 0),
            "ef": int(fin.get("forced", 0) or 0),
            "ace": int(fin.get("ace", 0) or 0),
            "df": int(fin.get("double_fault", 0) or 0),
            "error_rival": int(fin.get("opp_error", 0) or 0),
            "winner_rival": int(fin.get("opp_winner", 0) or 0),
        },
    }

    try:
        url = "https://api.openai.com/v1/chat/completions"
        req_payload = {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Eres un entrenador de tenis experto y motivador. "
                        "Da un resumen tipo IA: muy claro, corto, accionable, y con tono profesional. "
                        "Estructura: 1) Diagnóstico 2) Qué repetir 3) Qué ajustar 4) Plan próximo partido (3 bullets) "
                        "5) Una frase final motivadora. "
                        "NO inventes datos: usa solo los stats proporcionados."
                    ),
                },
                {"role": "user", "content": f"Stats del partido:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"},
            ],
        }
        data = json.dumps(req_payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        txt = obj["choices"][0]["message"]["content"].strip()
        return "🤖 **Resumen IA (coach)**\n\n" + txt
    except Exception as e:
        return (
            "⚠️ **No se pudo generar el resumen IA** (error de conexión o API).\n\n"
            f"Detalle: `{e}`\n\n"
            "Resumen estándar:\n\n" + base_summary
        )


# ==========================================================
# SESSION STATE INIT
# ==========================================================
def ss_init():
    if "live" not in st.session_state:
        st.session_state.live = LiveMatch()
    if "history" not in st.session_state:
        st.session_state.history = MatchHistory()
    if "finish" not in st.session_state:
        st.session_state.finish = None
    if "page" not in st.session_state:
        st.session_state.page = "LIVE"
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "auth_key" not in st.session_state:
        st.session_state.auth_key = None
    if "authed" not in st.session_state:
        st.session_state.authed = False


ss_init()

SURFACES = ("Tierra batida", "Pista rápida", "Hierba", "Indoor")
FINISH_ITEMS = [
    ("winner", "Winner"),
    ("unforced", "ENF"),
    ("forced", "EF"),
    ("ace", "Ace"),
    ("double_fault", "Doble falta"),
    ("opp_error", "Error rival"),
    ("opp_winner", "Winner rival"),
]


def small_note(txt: str):
    st.markdown(f"<div class='small-note'>{txt}</div>", unsafe_allow_html=True)


def title_h(txt: str):
    st.markdown(f"## {txt}")


# ==========================================================
# VISUAL HELPERS (NO FUNCTIONAL CHANGES)
# ==========================================================
def ring(label: str, value: float, sub: str = "", color: str = "var(--accent)"):
    v = 0.0 if value is None else float(value)
    v = max(0.0, min(100.0, v))
    deg = v * 3.6
    html = f"""
    <div class="ring-wrap">
      <div class="ring" style="--deg:{deg}deg; --ringc:{color};">
        <div class="ring-val">{v:.0f}%</div>
      </div>
      <div class="ring-txt">
        <div class="t1">{label}</div>
        <div class="t2">{sub}</div>
      </div>
    </div>
    """
    st.markdown(f"<div class='ts-card'>{html}</div>", unsafe_allow_html=True)


def score_pills(sets_me, sets_opp, games_me, games_opp, pts_label, surface):
    html = f"""
    <div class="pills">
      <div class="pill">🧱 <b>{surface}</b></div>
      <div class="pill">🎾 Sets <b>{sets_me}-{sets_opp}</b></div>
      <div class="pill">🧾 Juegos <b>{games_me}-{games_opp}</b></div>
      <div class="pill">🔢 Puntos <b>{pts_label}</b></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def last_points_timeline(points, n=18):
    arr = points[-n:] if points else []
    dots = []
    for p in arr:
        cls = "win" if p.get("result") == "win" else "lose"
        if p.get("pressure"):
            cls += " pressure"
        dots.append(f"<span class='dot {cls}' title='{p.get('result','')}'></span>")
    html = f"""
    <div style="font-weight:1000;">Últimos puntos</div>
    <div class="small-note">Verde=ganado · Rojo=perdido · Borde=presión</div>
    <div class="lp">{''.join(dots) if dots else "<span class='small-note'>Aún no hay puntos.</span>"}</div>
    """
    st.markdown(f"<div class='ts-card'>{html}</div>", unsafe_allow_html=True)


def court_svg(surface: str):
    surf_color = {
        "Tierra batida": "#c2410c",
        "Pista rápida": "#1d4ed8",
        "Hierba": "#16a34a",
        "Indoor": "#334155",
    }.get(surface, "#1d4ed8")
    html = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <div style="font-weight:1000;">Pista</div>
      <div class="small-note">Decorativa</div>
    </div>
    <svg viewBox="0 0 400 210" width="100%" height="150" style="margin-top:8px; border-radius:16px; overflow:hidden;">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="{surf_color}" stop-opacity="0.92"/>
          <stop offset="1" stop-color="{surf_color}" stop-opacity="0.75"/>
        </linearGradient>
        <pattern id="grain" width="6" height="6" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="2" r="0.6" fill="rgba(255,255,255,0.10)"/>
          <circle cx="4" cy="5" r="0.6" fill="rgba(0,0,0,0.10)"/>
        </pattern>
      </defs>
      <rect x="0" y="0" width="400" height="210" fill="url(#g)"/>
      <rect x="0" y="0" width="400" height="210" fill="url(#grain)" opacity="0.50"/>
      <rect x="20" y="15" width="360" height="180" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="3"/>
      <line x1="200" y1="15" x2="200" y2="195" stroke="rgba(255,255,255,0.75)" stroke-width="3"/>
      <rect x="60" y="45" width="280" height="120" fill="none" stroke="rgba(255,255,255,0.75)" stroke-width="3"/>
      <line x1="60" y1="105" x2="340" y2="105" stroke="rgba(255,255,255,0.75)" stroke-width="3"/>
      <circle cx="200" cy="105" r="5" fill="rgba(255,255,255,0.85)"/>
    </svg>
    """
    st.markdown(f"<div class='ts-card'>{html}</div>", unsafe_allow_html=True)


def icon_svg(kind: str):
    icons = {
        "score": """
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <path d="M4 7h16M4 12h16M4 17h16" stroke="rgba(2,6,23,.72)" stroke-width="2" stroke-linecap="round"/>
        </svg>""",
        "bolt": """
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <path d="M13 2L3 14h8l-1 8 11-14h-8l0-6z" fill="rgba(37,99,235,.85)"/>
        </svg>""",
        "shield": """
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <path d="M12 2l8 4v6c0 6-4 9-8 10-4-1-8-4-8-10V6l8-4z" fill="rgba(245,158,11,.85)"/>
        </svg>""",
        "trophy": """
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <path d="M7 4h10v3a5 5 0 0 1-10 0V4z" fill="rgba(22,163,74,.85)"/>
          <path d="M9 18h6v2H9z" fill="rgba(2,6,23,.55)"/>
          <path d="M10 13h4v5h-4z" fill="rgba(2,6,23,.35)"/>
        </svg>""",
    }
    return icons.get(kind, "")


# ==========================================================
# AUTH UI
# ==========================================================
def auth_block():
    st.markdown(
        """
        <div class="ts-header">
          <div class="ts-title">🎾 TennisStats</div>
          <div class="ts-sub">Acceso privado por usuario. Optimizado para móvil.</div>
          <div class="ts-badges">
            <div class="ts-badge"><span class="ts-dot"></span> Live tracking</div>
            <div class="ts-badge"><span class="ts-dot" style="background: var(--accent2); box-shadow:0 0 0 3px rgba(37,99,235,.14);"></span> Markov Win Prob</div>
            <div class="ts-badge"><span class="ts-dot" style="background: var(--warn); box-shadow:0 0 0 3px rgba(245,158,11,.14);"></span> Export / Import</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    users = load_users()
    tab_login, tab_register = st.tabs(["🔑 Entrar", "🆕 Crear usuario"])

    with tab_login:
        u = st.text_input("Usuario", value="", placeholder="Ej: ruben")
        pin = st.text_input("PIN", value="", type="password", placeholder="4-12 dígitos")
        if st.button("Entrar", use_container_width=True):
            key = safe_user_key(u)
            if not key or key not in users:
                st.error("Usuario no existe.")
                return
            if not pin:
                st.error("Introduce el PIN.")
                return
            rec = users[key]
            try:
                salt = _b64d(rec["salt"])
                want = rec["hash"]
                got = hash_pin(pin, salt)
            except Exception:
                st.error("Error leyendo credenciales. (users.json corrupto?)")
                return
            if secrets.compare_digest(got, want):
                st.session_state.authed = True
                st.session_state.auth_user = rec.get("display", u.strip() or key)
                st.session_state.auth_key = key
                st.session_state.history.matches = load_history_from_disk(key)
                st.success("Acceso correcto ✅")
                st.rerun()
            else:
                st.error("PIN incorrecto.")

    with tab_register:
        new_u = st.text_input("Nuevo usuario", value="", placeholder="Solo letras/números (mejor corto)")
        new_pin = st.text_input("Nuevo PIN", value="", type="password", placeholder="4-12 dígitos")
        new_pin2 = st.text_input("Repite PIN", value="", type="password")
        if st.button("Crear usuario", use_container_width=True):
            key = safe_user_key(new_u)
            if not key:
                st.error("El usuario no puede estar vacío.")
                return
            if key in users:
                st.error("Ese usuario ya existe.")
                return
            if not (new_pin.isdigit() and 4 <= len(new_pin) <= 12):
                st.error("El PIN debe ser numérico (4 a 12 dígitos).")
                return
            if new_pin != new_pin2:
                st.error("Los PIN no coinciden.")
                return

            salt = os.urandom(16)
            rec = {
                "display": new_u.strip(),
                "salt": _b64e(salt),
                "hash": hash_pin(new_pin, salt),
                "created": datetime.now().isoformat(timespec="seconds"),
            }
            users[key] = rec
            save_users(users)
            save_history_to_disk(key, [])
            st.success("Usuario creado ✅ Ya puedes entrar en la pestaña 'Entrar'.")


# ==========================================================
# MAIN: requiere login
# ==========================================================
if not st.session_state.authed:
    auth_block()
    st.stop()

live: LiveMatch = st.session_state.live
history: MatchHistory = st.session_state.history
user_key = st.session_state.auth_key
user_display = st.session_state.auth_user

# NAV
page_map = {"🎾": "LIVE", "📈": "ANALYSIS", "📊": "STATS", "📰": "NEWS", "🧠": "PSICO"}
labels = list(page_map.keys())
current_label = next((k for k, v in page_map.items() if v == st.session_state.page), "🎾")
nav = st.segmented_control(" ", options=labels, default=current_label, label_visibility="collapsed")
if nav and page_map.get(nav) != st.session_state.page:
    st.session_state.page = page_map[nav]

with st.sidebar:
    st.markdown("### 🎾 TennisStats")
    st.caption("Panel (en móvil puedes colapsarlo)")
    st.markdown(f"**👤 Usuario:** `{user_display}`")
    full_map = {"🎾 LIVE": "LIVE", "📈 Analysis": "ANALYSIS", "📊 Stats": "STATS", "📰 Noticias": "NEWS", "🧠 Psico": "PSICO"}
    cur_full = next((k for k, v in full_map.items() if v == st.session_state.page), "🎾 LIVE")
    choice = st.radio("Página", list(full_map.keys()), index=list(full_map.keys()).index(cur_full))
    st.session_state.page = full_map[choice]
    st.divider()
    if st.button("🚪 Salir", use_container_width=True):
        st.session_state.authed = False
        st.session_state.auth_user = None
        st.session_state.auth_key = None
        st.session_state.page = "LIVE"
        st.session_state.finish = None
        st.rerun()

# TOP DASHBOARD (visual, compact)
total_pts, won_pts, pct_pts = live.points_stats()
p_point = live.estimate_point_win_prob()
p_match = live.match_win_prob() * 100.0

st.markdown(
    f"""
    <div class="ts-header">
      <div class="ts-title">🎾 Dashboard</div>
      <div class="ts-sub">Visual, rápido y pensado para móvil.</div>
      <div class="ts-badges">
        <div class="ts-badge"><span class="ts-dot"></span> {user_display}</div>
        <div class="ts-badge"><span class="ts-dot" style="background: var(--accent2); box-shadow:0 0 0 3px rgba(37,99,235,.14);"></span> Win Prob {p_match:.1f}%</div>
        <div class="ts-badge"><span class="ts-dot" style="background: var(--warn); box-shadow:0 0 0 3px rgba(245,158,11,.14);"></span> p(punto) {p_point:.2f}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top1, top2 = st.columns(2, gap="small")
with top1:
    ring("Puntos ganados", pct_pts, f"{won_pts}/{total_pts}", "var(--accent)")
with top2:
    ring("Prob. victoria", p_match, "Modelo Markov", "var(--accent2)")


# ==========================================================
# PAGE: LIVE
# ==========================================================
if st.session_state.page == "LIVE":
    title_h("LIVE MATCH")

    st_ = live.state
    pts_label = f"TB {st_.pts_me}-{st_.pts_opp}" if st_.in_tiebreak else game_point_label(st_.pts_me, st_.pts_opp)

    a, b = st.columns([1.05, 0.95], gap="small")
    with a:
        live.surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(live.surface))
        small_note("Tip: usa el panel lateral para navegar rápido.")
    with b:
        st.markdown(
            f"""
            <div class="ts-card" style="display:flex; gap:10px; align-items:center;">
              <div>{icon_svg("score")}</div>
              <div>
                <div style="font-weight:1000;">Marcador</div>
                <div class="small-note"><span class="mono">Sets {st_.sets_me}-{st_.sets_opp} · Juegos {st_.games_me}-{st_.games_opp} · Puntos {pts_label}</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    score_pills(st_.sets_me, st_.sets_opp, st_.games_me, st_.games_opp, pts_label, live.surface)

    cL, cR = st.columns([1, 1], gap="small")
    with cL:
        court_svg(live.surface)
    with cR:
        last_points_timeline(live.points, n=18)

    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.subheader("Registrar punto", anchor=False)
    c1, c2 = st.columns(2, gap="small")
    with c1:
        if st.button("🟩 Punto Yo", use_container_width=True):
            live.add_point("win", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()
    with c2:
        if st.button("🟥 Punto Rival", use_container_width=True):
            live.add_point("lose", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    
        # Acciones manuales
st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
st.subheader("Acciones manuales", anchor=False)
c3, c4 = st.columns(2, gap="small")
with c3:
    if st.button("➕ Juego Yo", use_container_width=True):
        live.add_game_manual("me")
        st.rerun()
    if st.button("➕ Set Yo", use_container_width=True):
        live.add_set_manual("me")
        st.rerun()
with c4:
    if st.button("➕ Juego Rival", use_container_width=True):
        live.add_game_manual("opp")
        st.rerun()
    if st.button("➕ Set Rival", use_container_width=True):
        live.add_set_manual("opp")
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)




