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
# CONFIG + CSS (MOBILE PRO SPORT UI - MORE VISUAL)
# ==========================================================
st.set_page_config(page_title="TennisStats", page_icon="🎾", layout="centered")

PRO_CSS = """
<style>
:root{
  --bg:#f6f8fb;
  --bg2:#eef2f7;
  --card:#ffffff;
  --card2:#fbfcff;
  --text:#0b1220;
  --muted:#475569;
  --muted2:#64748b;
  --stroke: rgba(2,6,23,0.10);
  --stroke2: rgba(2,6,23,0.08);
  --accent:#16a34a;      /* green */
  --accent2:#2563eb;     /* blue */
  --danger:#dc2626;      /* red */
  --warn:#f59e0b;        /* amber */
  --radius: 18px;
  --shadow: 0 18px 40px rgba(2,6,23,.10);
  --shadow2: 0 10px 22px rgba(2,6,23,.08);
  --focus: 0 0 0 3px rgba(37,99,235,.16);
}

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 380px at 15% -5%, rgba(22,163,74,.14), transparent 60%),
    radial-gradient(850px 360px at 85% 0%, rgba(37,99,235,.12), transparent 60%),
    linear-gradient(180deg, var(--bg), var(--bg2));
  color: var(--text);
}

[data-testid="stAppViewContainer"]::before{
  content:"";
  position: fixed;
  inset: 0;
  pointer-events:none;
  opacity: .30;
  background:
    linear-gradient(90deg, rgba(2,6,23,.035) 1px, transparent 1px) 0 0 / 140px 140px,
    linear-gradient(0deg, rgba(2,6,23,.028) 1px, transparent 1px) 0 0 / 140px 140px;
  mask-image: radial-gradient(circle at 50% 0%, black 22%, transparent 62%);
}

.block-container{
  padding-top: 0.75rem;
  padding-bottom: 1.25rem;
  max-width: 980px;
}

div[data-testid="stVerticalBlock"] > div { gap: 0.65rem; }
header[data-testid="stHeader"]{ height: 0.45rem; background: transparent; }

* { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
h1,h2,h3{ letter-spacing: .2px; }
.stCaption, [data-testid="stCaptionContainer"]{ color: var(--muted2) !important; }

.small-note{ color: var(--muted); font-size: .92rem; line-height: 1.25rem; }
.kpi{ font-size: 1.03rem; font-weight: 950; }
.mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }

hr, [data-testid="stDivider"]{
  border-color: var(--stroke2) !important;
  margin: 0.55rem 0;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div{
  background: linear-gradient(180deg, var(--card), var(--card2)) !important;
  border: 1px solid var(--stroke) !important;
  border-radius: 14px !important;
  box-shadow: var(--shadow2);
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea{ color: var(--text) !important; }
label, .stTextInput label, .stSelectbox label, .stNumberInput label{
  color: var(--muted2) !important;
  font-weight: 850 !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within{
  outline: none !important;
  box-shadow: var(--shadow2), var(--focus) !important;
  border-color: rgba(37,99,235,.35) !important;
}

/* Buttons */
.stButton>button{
  width: 100%;
  padding: 0.56rem 0.92rem;
  border-radius: 14px;
  border: 1px solid var(--stroke);
  background: linear-gradient(180deg, #ffffff, #f4f7fb);
  color: var(--text);
  font-weight: 950;
  box-shadow: var(--shadow2);
  transition: transform .06s ease, box-shadow .12s ease, border-color .12s ease;
}
.stButton>button:hover{
  border-color: rgba(22,163,74,.35);
  box-shadow: 0 14px 28px rgba(2,6,23,.12);
}
.stButton>button:active{ transform: translateY(1px) scale(0.99); }
.stButton>button:focus{ outline: none !important; box-shadow: var(--shadow2), var(--focus) !important; }

[data-testid="stDownloadButton"] > button{
  border-radius: 14px !important;
  border: 1px solid rgba(37,99,235,.22) !important;
  background: linear-gradient(180deg, rgba(37,99,235,.14), rgba(255,255,255,0.98)) !important;
  color: var(--text) !important;
  font-weight: 980 !important;
  box-shadow: var(--shadow2) !important;
}
[data-testid="stDownloadButton"] > button:hover{ border-color: rgba(37,99,235,.38) !important; }

/* Expanders */
[data-testid="stExpander"]{
  border: 1px solid var(--stroke) !important;
  border-radius: var(--radius) !important;
  background: linear-gradient(180deg, var(--card), var(--card2)) !important;
  box-shadow: var(--shadow);
  overflow: hidden;
}
[data-testid="stExpander"] summary{ font-weight: 980 !important; }

/* Tabs */
[data-baseweb="tab-list"]{
  background: rgba(255,255,255,0.75);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 6px;
  gap: 6px;
  box-shadow: var(--shadow2);
}
button[role="tab"]{
  border-radius: 12px !important;
  font-weight: 980 !important;
  color: var(--muted) !important;
}
button[role="tab"][aria-selected="true"]{
  background: linear-gradient(180deg, rgba(22,163,74,.16), rgba(255,255,255,.95)) !important;
  color: var(--text) !important;
  border: 1px solid rgba(22,163,74,.25) !important;
}

/* Alerts / uploader */
[data-testid="stAlert"]{
  border-radius: 16px !important;
  border: 1px solid var(--stroke) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.82)) !important;
  box-shadow: var(--shadow2);
}
section[data-testid="stFileUploaderDropzone"]{
  border-radius: 16px !important;
  border: 1px dashed rgba(2,6,23,0.20) !important;
  background: rgba(255,255,255,0.65) !important;
  box-shadow: var(--shadow2);
}

/* Header card */
.ts-header{
  border: 1px solid var(--stroke);
  border-radius: 22px;
  padding: 14px 16px;
  background: linear-gradient(180deg, rgba(255,255,255,0.90), rgba(255,255,255,0.72));
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
}
.ts-header::after{
  content:"";
  position:absolute;
  inset:-60% -20% auto auto;
  width: 520px; height: 320px;
  background: radial-gradient(circle at 30% 40%, rgba(37,99,235,.18), transparent 62%);
  transform: rotate(12deg);
}
.ts-title{
  font-size: 1.18rem;
  font-weight: 1000;
  margin: 0;
}
.ts-sub{
  margin: 4px 0 0 0;
  color: var(--muted);
  font-weight: 700;
  font-size: .92rem;
}
.ts-badges{
  margin-top: 10px;
  display:flex;
  flex-wrap:wrap;
  gap: 8px;
}
.ts-badge{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.72);
  font-weight: 900;
  font-size: .88rem;
  color: var(--text);
}
.ts-dot{
  width: 9px; height: 9px;
  border-radius: 999px;
  background: var(--accent);
  box-shadow: 0 0 0 3px rgba(22,163,74,.16);
}

/* Cards */
.ts-card{
  border: 1px solid var(--stroke);
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.82));
  box-shadow: var(--shadow2);
  padding: 12px 12px;
}

/* Donut rings */
.ring-wrap{ display:flex; gap: 12px; align-items:center; }
.ring{
  width: 58px; height: 58px; border-radius: 999px;
  background: conic-gradient(var(--ringc) var(--deg), rgba(2,6,23,.08) 0);
  position: relative; box-shadow: 0 10px 18px rgba(2,6,23,.08);
}
.ring::after{
  content:""; position:absolute; inset: 8px; border-radius: 999px;
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.86));
  border: 1px solid rgba(2,6,23,.06);
}
.ring-val{
  position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-weight: 1000; color: var(--text); font-size: .92rem; z-index: 2;
}
.ring-txt .t1{ font-weight: 1000; line-height: 1.05rem; }
.ring-txt .t2{ color: var(--muted); font-weight: 800; font-size: .88rem; margin-top: 2px; }

/* Score pills */
.pills{ display:flex; gap: 8px; flex-wrap:wrap; margin-top:8px; }
.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding: 7px 10px; border-radius: 999px;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.70);
  font-weight: 950; font-size:.90rem;
}
.pill b{ font-weight:1000; }

/* Last points timeline */
.lp{
  display:flex; gap:6px; flex-wrap:wrap;
  margin-top:10px;
}
.dot{
  width: 14px; height: 14px; border-radius: 999px;
  border: 1px solid rgba(2,6,23,.14);
  box-shadow: 0 6px 12px rgba(2,6,23,.08);
}
.dot.win{ background: rgba(22,163,74,.95); }
.dot.lose{ background: rgba(220,38,38,.90); }
.dot.pressure{ outline: 3px solid rgba(245,158,11,.22); }

/* Segmented nav */
div[data-testid="stSegmentedControl"] > div{
  border-radius: 16px !important;
  border: 1px solid var(--stroke) !important;
  background: rgba(255,255,255,0.72) !important;
  box-shadow: var(--shadow2) !important;
  padding: 6px !important;
}
div[data-testid="stSegmentedControl"] label{ font-weight: 950 !important; }

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
        return _prob_match_bo3(p_r, st_.sets_me, st_.sets_opp, st_.games_me, st_.games_opp, st_.pts_me, st_.pts_opp, st_.in_tiebreak)

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

        self.points.append({"result": result, **meta, "surface": self.surface, "before": before.__dict__, "set_idx": set_idx, "pressure": is_pressure})

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
# Resumen tipo entrenador (basado en stats)
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
    <div class="ts-card">
      <div class="ring-wrap">
        <div class="ring" style="--deg:{deg}deg; --ringc:{color};">
          <div class="ring-val">{v:.0f}%</div>
        </div>
        <div class="ring-txt">
          <div class="t1">{label}</div>
          <div class="t2">{sub}</div>
        </div>
      </div>
    </div>
    """
    st.components.v1.html(html, height=92)


def mini_card(title: str, body_html: str):
    st.markdown(
        f"""
        <div class="ts-card">
          <div style="font-weight:1000; margin-bottom:4px;">{title}</div>
          <div class="small-note">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    <div class="ts-card">
      <div style="font-weight:1000;">Últimos puntos</div>
      <div class="small-note">Verde=ganado · Rojo=perdido · Borde=presión</div>
      <div class="lp">{''.join(dots) if dots else "<span class='small-note'>Aún no hay puntos.</span>"}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def court_svg(surface: str):
    # simple decorative court (no input)
    surf_color = {
        "Tierra batida": "#c2410c",
        "Pista rápida": "#1d4ed8",
        "Hierba": "#16a34a",
        "Indoor": "#334155",
    }.get(surface, "#1d4ed8")
    html = f"""
    <div class="ts-card" style="padding:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-weight:1000;">Vista rápida</div>
        <div class="small-note">Pista (decorativa)</div>
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
        <rect x="0" y="0" width="400" height="210" fill="url(#grain)" opacity="0.55"/>
        <rect x="20" y="15" width="360" height="180" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="3"/>
        <line x1="200" y1="15" x2="200" y2="195" stroke="rgba(255,255,255,0.75)" stroke-width="3"/>
        <rect x="60" y="45" width="280" height="120" fill="none" stroke="rgba(255,255,255,0.75)" stroke-width="3"/>
        <line x1="60" y1="105" x2="340" y2="105" stroke="rgba(255,255,255,0.75)" stroke-width="3"/>
        <circle cx="200" cy="105" r="5" fill="rgba(255,255,255,0.85)"/>
      </svg>
    </div>
    """
    st.components.v1.html(html, height=230)


def finish_bar_current_match(live_match: LiveMatch):
    # purely visual, derived from current live points
    counts = {k: 0 for k, _ in FINISH_ITEMS}
    for p in live_match.points:
        f = p.get("finish")
        if f in counts:
            counts[f] += 1
    total = sum(counts.values())
    if total <= 0:
        return
    labels = [lbl for _, lbl in FINISH_ITEMS]
    values = [counts[k] for k, _ in FINISH_ITEMS]
    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.subheader("Finish (partido actual)", anchor=False)
    small_note("Distribución de finishes registrados en este partido.")
    st.bar_chart({labels[i]: values[i] for i in range(len(labels))}, height=240)
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# AUTH UI
# ==========================================================
def auth_block():
    st.markdown(
        """
        <div class="ts-header">
          <div class="ts-title">🎾 TennisStats</div>
          <div class="ts-sub">Acceso privado por usuario. Interfaz optimizada para móvil.</div>
          <div class="ts-badges">
            <div class="ts-badge"><span class="ts-dot"></span> Live tracking</div>
            <div class="ts-badge"><span class="ts-dot" style="background: var(--accent2); box-shadow:0 0 0 3px rgba(37,99,235,.14);"></span> Markov Win Prob</div>
            <div class="ts-badge"><span class="ts-dot" style="background: var(--warn); box-shadow:0 0 0 3px rgba(245,158,11,.14);"></span> Export / Import</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("")

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

# TOP DASHBOARD (visual)
total_pts, won_pts, pct_pts = live.points_stats()
p_point = live.estimate_point_win_prob()
p_match = live.match_win_prob() * 100.0

st.markdown(
    f"""
    <div class="ts-header">
      <div class="ts-title">🎾 Dashboard</div>
      <div class="ts-sub">Intuitivo y visual para uso en móvil: anillos, timeline de puntos y pistas.</div>
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

    # Contexto arriba: superficie + marcador visual
    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    a, b = st.columns([1.05, 0.95], gap="small")
    with a:
        live.surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(live.surface))
        small_note("Tip: usa el panel lateral para cambiar de página rápido.")
    with b:
        mini_card("Estado del partido", f"<span class='mono'>Sets {st_.sets_me}-{st_.sets_opp} · Juegos {st_.games_me}-{st_.games_opp} · Puntos {pts_label}</span><br/>Win Prob <b>{p_match:.1f}%</b>")
    score_pills(st_.sets_me, st_.sets_opp, st_.games_me, st_.games_opp, pts_label, live.surface)
    st.markdown("</div>", unsafe_allow_html=True)

    # Pista decorativa + timeline de puntos
    cL, cR = st.columns([1, 1], gap="small")
    with cL:
        court_svg(live.surface)
    with cR:
        last_points_timeline(live.points, n=18)

    # Botones principales (mismo comportamiento)
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

    # Finish selector (igual lógica)
    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.subheader("Finish (opcional)", anchor=False)
    small_note("Selecciona 1 (se aplica al siguiente punto). Puedes deseleccionar tocando de nuevo.")
    fcols = st.columns(2, gap="small")
    for i, (key, label) in enumerate(FINISH_ITEMS):
        with fcols[i % 2]:
            selected = (st.session_state.finish == key)
            txt = f"✅ {label}" if selected else label
            if st.button(txt, key=f"finish_{key}", use_container_width=True):
                st.session_state.finish = None if selected else key
                st.rerun()

    colx, coly = st.columns([1, 1], gap="small")
    with colx:
        if st.button("🧼 Limpiar", use_container_width=True):
            st.session_state.finish = None
            st.rerun()
    with coly:
        small_note(f"Seleccionado: **{st.session_state.finish or '—'}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # Visual extra: finishes del partido actual (solo si hay datos)
    finish_bar_current_match(live)

    # Acciones rápidas + finalizar (misma funcionalidad)
    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.subheader("Acciones", anchor=False)
    a1, a2, a3 = st.columns(3, gap="small")
    with a1:
        if st.button("↩️ Deshacer", use_container_width=True):
            live.undo()
            st.rerun()
    with a2:
        if st.button("📈 Ir a Analysis", use_container_width=True):
            st.session_state.page = "ANALYSIS"
            st.rerun()
    with a3:
        if st.button("🏁 Finalizar", use_container_width=True):
            st.session_state._open_finish = True

    if st.session_state.get("_open_finish", False):
        with st.expander("Finalizar partido", expanded=True):
            st.write("Introduce el marcador final y guarda el partido.")
            sw = st.number_input("Sets Yo", 0, 5, value=int(live.state.sets_me), step=1)
            sl = st.number_input("Sets Rival", 0, 5, value=int(live.state.sets_opp), step=1)
            gw = st.number_input("Juegos Yo", 0, 50, value=int(live.state.games_me), step=1)
            gl = st.number_input("Juegos Rival", 0, 50, value=int(live.state.games_opp), step=1)
            surf_save = st.selectbox("Superficie (guardar)", SURFACES, index=SURFACES.index(live.surface))

            s_left, s_right = st.columns(2, gap="small")
            with s_left:
                if st.button("Cancelar", use_container_width=True):
                    st.session_state._open_finish = False
                    st.rerun()
            with s_right:
                if st.button("Guardar partido", use_container_width=True):
                    won_match = (sw > sl)
                    report = live.match_summary()

                    history.add({
                        "id": f"m_{datetime.now().timestamp()}",
                        "date": datetime.now().isoformat(timespec="seconds"),
                        "won_match": bool(won_match),
                        "sets_w": int(sw), "sets_l": int(sl),
                        "games_w": int(gw), "games_l": int(gl),
                        "surface": surf_save,
                        **report,
                    })
                    save_history_to_disk(user_key, history.matches)

                    live.surface = surf_save
                    live.reset()
                    st.session_state.finish = None
                    st.session_state._open_finish = False
                    st.success("Partido guardado ✅")
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Historial + export/import (mismo)
    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.subheader("Historial y exportación", anchor=False)
    small_note("Tu historial privado (solo tu usuario). Puedes editar/borrar y exportar/importar en JSON.")

    if not history.matches:
        st.info("Aún no hay partidos guardados.")
    else:
        last10 = history.filtered_matches(n=10, surface=None)
        w10 = sum(1 for m in last10 if m.get("won_match"))
        t10 = len(last10)
        pct10 = (w10 / t10 * 100.0) if t10 else 0.0

        x1, x2 = st.columns(2, gap="small")
        with x1:
            ring("Últimos 10", pct10, f"{w10}/{t10} victorias", "var(--accent)")
        with x2:
            best_all = history.best_streak(surface=None)
            mini_card("🔥 Mejor racha", f"<b>{best_all}</b> victorias seguidas")

        matches = list(reversed(history.matches))
        for idx, m in enumerate(matches):
            real_i = len(history.matches) - 1 - idx
            date = m.get("date", "")
            surf = m.get("surface", "—")
            res = "✅ W" if m.get("won_match") else "❌ L"
            score = f"{m.get('sets_w',0)}-{m.get('sets_l',0)} sets · {m.get('games_w',0)}-{m.get('games_l',0)} juegos"
            pts = f"{m.get('points_won',0)}/{m.get('points_total',0)} pts ({m.get('points_pct',0):.0f}%)"

            with st.expander(f"{res} · {score} · {surf} · {date}", expanded=False):
                st.write(f"**{score}**")
                small_note(f"{pts} · Presión: {m.get('pressure_won',0)}/{m.get('pressure_total',0)} ({m.get('pressure_pct',0):.0f}%)")

                fin = (m.get("finishes") or {})
                fin_line = f"Winners {fin.get('winner',0)} · ENF {fin.get('unforced',0)} · EF {fin.get('forced',0)} · Ace {fin.get('ace',0)} · DF {fin.get('double_fault',0)}"
                small_note(fin_line)

                if st.button("🧠 Resumen tipo entrenador", key=f"coach_{m.get('id',real_i)}", use_container_width=True):
                    st.session_state._coach_open = True
                    st.session_state._coach_text = coach_summary_from_match(m)
                    st.rerun()

                e1, e2 = st.columns(2, gap="small")
                with e1:
                    if st.button("✏️ Editar", key=f"edit_btn_{m.get('id',real_i)}", use_container_width=True):
                        st.session_state._edit_index = real_i
                        st.session_state._edit_open = True
                        st.rerun()
                with e2:
                    if st.button("🗑️ Borrar", key=f"del_btn_{m.get('id',real_i)}", use_container_width=True):
                        history.matches.pop(real_i)
                        save_history_to_disk(user_key, history.matches)
                        st.success("Partido borrado.")
                        st.rerun()

        if st.session_state.get("_coach_open", False):
            with st.expander("🧠 Resumen del entrenador", expanded=True):
                st.markdown(st.session_state.get("_coach_text", ""))
                if st.button("Cerrar resumen", use_container_width=True):
                    st.session_state._coach_open = False
                    st.session_state._coach_text = ""
                    st.rerun()

        if st.session_state.get("_edit_open", False):
            i = st.session_state.get("_edit_index", None)
            if i is not None and 0 <= i < len(history.matches):
                m = history.matches[i]
                with st.expander("✏️ Editar partido", expanded=True):
                    st.write("Modifica los campos y guarda.")
                    col1, col2 = st.columns(2, gap="small")
                    with col1:
                        won_match = st.toggle("Victoria", value=bool(m.get("won_match", False)), key=f"edit_victoria_{m.get('id', i)}")
                        sets_w = st.number_input("Sets Yo", 0, 5, value=int(m.get("sets_w", 0)), step=1, key=f"edit_sets_w_{m.get('id', i)}")
                        games_w = st.number_input("Juegos Yo", 0, 50, value=int(m.get("games_w", 0)), step=1, key=f"edit_games_w_{m.get('id', i)}")
                    with col2:
                        sets_l = st.number_input("Sets Rival", 0, 5, value=int(m.get("sets_l", 0)), step=1, key=f"edit_sets_l_{m.get('id', i)}")
                        games_l = st.number_input("Juegos Rival", 0, 50, value=int(m.get("games_l", 0)), step=1, key=f"edit_games_l_{m.get('id', i)}")
                        surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(m.get("surface", SURFACES[0])), key=f"edit_surface_{m.get('id', i)}")

                    date = st.text_input("Fecha (ISO)", value=str(m.get("date", "")), key=f"edit_date_{m.get('id', i)}")

                    bL, bR = st.columns(2, gap="small")
                    with bL:
                        if st.button("Cancelar edición", use_container_width=True, key=f"edit_cancel_{m.get('id', i)}"):
                            st.session_state._edit_open = False
                            st.session_state._edit_index = None
                            st.rerun()
                    with bR:
                        if st.button("Guardar cambios", use_container_width=True, key=f"edit_save_{m.get('id', i)}"):
                            m["won_match"] = bool(won_match)
                            m["sets_w"] = int(sets_w)
                            m["sets_l"] = int(sets_l)
                            m["games_w"] = int(games_w)
                            m["games_l"] = int(games_l)
                            m["surface"] = surface
                            m["date"] = date
                            history.matches[i] = m

                            save_history_to_disk(user_key, history.matches)

                            st.session_state._edit_open = False
                            st.session_state._edit_index = None
                            st.success("Cambios guardados ✅")
                            st.rerun()

    export_obj = {"matches": history.matches}
    export_json = json.dumps(export_obj, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Descargar historial (JSON)",
        data=export_json,
        file_name=f"tennis_history__{user_key}.json",
        mime="application/json",
        use_container_width=True,
    )

    up = st.file_uploader("⬆️ Importar historial (JSON)", type=["json"], label_visibility="visible")
    if up is not None:
        try:
            obj = json.loads(up.read().decode("utf-8"))
            matches = obj.get("matches", [])
            if not isinstance(matches, list):
                raise ValueError("Formato incorrecto: 'matches' debe ser una lista.")
            for mm in matches:
                if "id" not in mm:
                    mm["id"] = f"m_{datetime.now().timestamp()}"
            history.matches = matches
            save_history_to_disk(user_key, history.matches)
            st.success("Historial importado ✅")
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo importar: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# PAGE: ANALYSIS
# ==========================================================
elif st.session_state.page == "ANALYSIS":
    title_h("Analysis")

    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.markdown("**Win Probability (modelo real)**")
    small_note(f"p(punto)≈{p_point:.2f} · Win Prob≈{p_match:.1f}%")
    small_note("Modelo: Markov (punto→juego→set→BO3). p(punto) se estima con tus puntos del partido.")
    st.markdown("</div>", unsafe_allow_html=True)

    probs = live.win_prob_series()
    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.subheader("Evolución Win Prob", anchor=False)
    if len(probs) < 2:
        st.info("Aún no hay suficientes puntos para dibujar la gráfica (mínimo 2).")
    else:
        st.area_chart(probs, height=280)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.subheader("Puntos de presión (live)", anchor=False)
    pressure_total = sum(1 for p in live.points if p.get("pressure"))
    pressure_won = sum(1 for p in live.points if p.get("pressure") and p.get("result") == "win")
    pressure_pct = (pressure_won / pressure_total * 100.0) if pressure_total else 0.0

    a, b = st.columns(2, gap="small")
    with a:
        ring("Presión", pressure_pct, f"{pressure_won}/{pressure_total} ganados", "var(--warn)")
    with b:
        last_points_timeline(live.points, n=12)
    st.write(f"**{pressure_won}/{pressure_total}** ganados ({pressure_pct:.0f}%) en deuce/tiebreak.")
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# PAGE: STATS
# ==========================================================
elif st.session_state.page == "STATS":
    title_h("Stats")

    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    colF1, colF2 = st.columns([1.1, 0.9], gap="small")
    with colF1:
        n_choice = st.selectbox("Rango", ["Últ. 10", "Últ. 30", "Todos"], index=0)
    with colF2:
        surf_filter = st.selectbox("Superficie", ["Todas", *SURFACES], index=0)
    n = 10 if n_choice == "Últ. 10" else (30 if n_choice == "Últ. 30" else None)
    agg = history.aggregate(n=n, surface=surf_filter)
    st.markdown("</div>", unsafe_allow_html=True)

    # Visual principal: 3 anillos
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        ring("Partidos", agg["matches_pct"], f"{agg['matches_win']} / {agg['matches_total']}", "var(--accent)")
    with c2:
        ring("Sets", agg["sets_pct"], f"{agg['sets_w']} / {agg['sets_w'] + agg['sets_l']}", "var(--accent2)")
    with c3:
        ring("Juegos", agg["games_pct"], f"{agg['games_w']} / {agg['games_w'] + agg['games_l']}", "var(--warn)")

    # Resumen + finishes
    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.subheader("Resumen", anchor=False)
    st.write(
        f"**Puntos:** {agg['points_won']}/{agg['points_total']} ({agg['points_pct']:.0f}%) · "
        f"**Presión:** {agg['pressure_won']}/{agg['pressure_total']} ({agg['pressure_pct']:.0f}%)"
    )
    fin = agg["finishes_sum"]
    small_note(
        f"Winners {fin['winner']} · ENF {fin['unforced']} · EF {fin['forced']} · "
        f"Aces {fin['ace']} · Dobles faltas {fin['double_fault']}"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Rachas (visual)
    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.subheader("Rachas", anchor=False)
    results = history.last_n_results(10, surface=(None if surf_filter == "Todas" else surf_filter))
    if not results:
        st.info("Aún no hay partidos guardados.")
    else:
        row = ["✅ W" if r == "W" else "⬛ L" for r in results]
        st.write(" · ".join(row))
    best = history.best_streak(surface=(None if surf_filter == "Todas" else surf_filter))
    st.write(f"**🔥 Mejor racha:** {best} victorias seguidas")
    st.markdown("</div>", unsafe_allow_html=True)

    # Superficies + gráfico
    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    st.subheader("Superficies", anchor=False)
    surf = agg["surfaces"]
    for srf in SURFACES:
        w = surf.get(srf, {}).get("w", 0)
        t_ = surf.get(srf, {}).get("t", 0)
        pct = (w / t_ * 100.0) if t_ else 0.0
        st.write(f"**{srf}:** {pct:.0f}%  ({w} de {t_})")

    chart_data = {}
    for srf in SURFACES:
        w = surf.get(srf, {}).get("w", 0)
        t_ = surf.get(srf, {}).get("t", 0)
        chart_data[srf] = (w / t_ * 100.0) if t_ else 0.0

    if any(v > 0 for v in chart_data.values()):
        st.bar_chart(chart_data, height=260)
    else:
        small_note("Aún no hay datos suficientes para mostrar el gráfico por superficies.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Extra visual: evolución de resultados por fecha (si hay)
    if history.matches:
        try:
            st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
            st.subheader("Evolución (historial)", anchor=False)
            # serie: 1 si victoria, 0 si derrota; con etiqueta por fecha (simple)
            series = []
            for m in history.matches[-30:]:
                series.append(100.0 if m.get("won_match") else 0.0)
            if len(series) >= 2:
                st.line_chart(series, height=220)
                small_note("Serie: 100=Victoria, 0=Derrota (últimos 30).")
            else:
                small_note("Añade más partidos para ver evolución.")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            pass


# ==========================================================
# PAGE: NEWS
# ==========================================================
elif st.session_state.page == "NEWS":
    title_h("Noticias (tenis)")
    small_note("Últimas noticias desde fuentes públicas (RSS). Si alguna fuente falla, se muestra el resto.")

    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    cL, cR = st.columns([1, 1], gap="small")
    with cL:
        max_items = st.selectbox("Cuántas noticias", [8, 12, 15, 20], index=1)
    with cR:
        if st.button("🔄 Actualizar", use_container_width=True):
            fetch_tennis_news.clear()
            st.rerun()
    news = fetch_tennis_news(max_items=int(max_items))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    if not news:
        st.info("No se pudieron cargar noticias ahora mismo. Prueba a recargar en unos segundos.")
    else:
        for it in news:
            src = it.get("source", "—")
            title = it.get("title", "Noticia")
            link = it.get("link", "#")
            pub = it.get("published", "")
            if pub:
                st.markdown(f"- **[{title}]({link})**  \n  <span class='small-note'>{src} · {pub}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"- **[{title}]({link})**  \n  <span class='small-note'>{src}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# PAGE: PSICO
# ==========================================================
else:
    title_h("Psico")
    small_note("Material en PDF (visible y descargable).")

    st.markdown("<div class='ts-card'>", unsafe_allow_html=True)
    psico_dir = Path("psico_pdfs")
    pdfs = []
    if psico_dir.exists() and psico_dir.is_dir():
        pdfs = sorted([p for p in psico_dir.glob("*.pdf") if p.is_file()], key=lambda x: x.name.lower())

    if not pdfs:
        st.info("No se han encontrado PDFs en la carpeta `psico_pdfs/`. Sube los archivos al repo y redeploy.")
    else:
        for p in pdfs:
            k = hashlib.md5(p.name.encode("utf-8")).hexdigest()[:10]
            with st.expander(f"📄 {p.name}", expanded=False):
                try:
                    data = p.read_bytes()
                except Exception as e:
                    st.error(f"No se pudo leer el PDF: {e}")
                    continue

                st.download_button(
                    "⬇️ Descargar PDF",
                    data=data,
                    file_name=p.name,
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"psico_dl_{k}",
                )

                b64 = base64.b64encode(data).decode("utf-8")
                html = f"""
                <iframe
                    src="data:application/pdf;base64,{b64}"
                    width="100%"
                    height="650"
                    style="border: 1px solid rgba(2,6,23,0.12); border-radius: 14px; background: rgba(255,255,255,0.6);"
                ></iframe>
                """
                st.components.v1.html(html, height=680, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)
