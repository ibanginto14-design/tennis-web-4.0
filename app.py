import os, re, json, base64, secrets, hashlib, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
import streamlit as st


# =========================
# CONFIG (same app, new design only)
# =========================
st.set_page_config(page_title="TennisStats", page_icon="🎾", layout="centered")


def _read_gif_data_uri():
    for p in [Path("assets/tennis_ball_slowmo.gif"), Path("assets/tennis_bg.gif"),
              Path("tennis_ball_slowmo.gif"), Path("tennis_bg.gif")]:
        try:
            if p.exists() and p.is_file():
                b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                return f"data:image/gif;base64,{b64}"
        except Exception:
            pass
    return ""


BG_GIF = _read_gif_data_uri()

# ---- NEW DESIGN: "FITCOURT" (clean + professional, minimal CSS) ----
st.markdown(f"""
<style>
:root {{
  --bg:#F6F7FB; --card:#FFFFFF; --text:#0B1220; --muted:#5B6475;
  --stroke:rgba(10,20,40,.12); --accent:#00C389; --accent2:#2F6BFF; --warn:#F59E0B; --danger:#EF4444;
  --r:18px; --shadow: 0 18px 46px rgba(10,20,40,.10);
}}
html, body, [data-testid="stAppViewContainer"] {{
  color: var(--text);
  background:
    radial-gradient(900px 420px at 12% -10%, rgba(47,107,255,.14), transparent 60%),
    radial-gradient(900px 420px at 86% -10%, rgba(0,195,137,.12), transparent 60%),
    linear-gradient(180deg, var(--bg), #FFFFFF);
}}
header[data-testid="stHeader"]{{height:0;background:transparent;}}
.block-container{{max-width:1060px;padding-top:.6rem;padding-bottom:1.1rem;}}
/* subtle animated background (optional) */
{"[data-testid='stAppViewContainer']::after{content:'';position:fixed;inset:-12%;background-image:url('"+BG_GIF+"');background-size:cover;background-position:center;opacity:.06;filter:blur(2px) saturate(1.05);pointer-events:none;z-index:0;}" if BG_GIF else ""}
.block-container, section, footer {{position:relative; z-index:1;}}

.tsTop {{
  position: sticky; top: 0; z-index: 20;
  padding: 8px 0 10px 0;
}}
.tsTopInner {{
  background: rgba(255,255,255,.78);
  border: 1px solid var(--stroke);
  border-radius: calc(var(--r) + 6px);
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
  padding: 10px 12px;
}}
.tsRow{{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap;}}
.badge{{display:inline-flex;gap:8px;align-items:center;padding:6px 10px;border-radius:999px;border:1px solid var(--stroke);background:rgba(255,255,255,.9);font-weight:800;font-size:.88rem;}}
.dot{{width:10px;height:10px;border-radius:999px;background:var(--accent);box-shadow:0 0 0 3px rgba(0,195,137,.16);}}

.card {{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: calc(var(--r) + 4px);
  box-shadow: var(--shadow);
  padding: 12px 12px;
}}
.card.tight{{padding:10px 10px;}}
.h2{{font-size:1.2rem;font-weight:900;margin:.2rem 0 .25rem 0;}}
.sub{{color:var(--muted);font-weight:650;margin:0 0 .25rem 0;}}
.small{{color:var(--muted);font-size:.92rem;line-height:1.25rem;}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;}}

.stButton>button {{
  width:100%;
  border-radius: 16px;
  border: 1px solid var(--stroke);
  background: linear-gradient(180deg, #FFFFFF, rgba(255,255,255,.92));
  box-shadow: 0 14px 28px rgba(10,20,40,.10);
  color: var(--text);
  font-weight: 900;
  padding: .72rem .95rem;
}}
.stButton>button:hover {{border-color: rgba(47,107,255,.26);}}
.stButton>button:focus {{outline:none; box-shadow: 0 14px 28px rgba(10,20,40,.10), 0 0 0 3px rgba(47,107,255,.14);}}

div[data-baseweb="tab-list"] {{
  background: rgba(255,255,255,.72);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 6px;
}}
button[role="tab"]{{border-radius:12px!important;font-weight:900!important;color:var(--muted)!important;}}
button[role="tab"][aria-selected="true"] {{
  background: rgba(47,107,255,.12)!important;
  border: 1px solid rgba(47,107,255,.22)!important;
  color: var(--text)!important;
}}
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div, div[data-baseweb="textarea"]>div {{
  background: rgba(255,255,255,.9)!important;
  border: 1px solid var(--stroke)!important;
  border-radius: 14px!important;
}}
[data-testid="stExpander"] {{
  border: 1px solid var(--stroke)!important;
  border-radius: calc(var(--r) + 6px)!important;
  background: rgba(255,255,255,.86)!important;
  box-shadow: var(--shadow);
}}
[data-testid="stAlert"] {{
  border-radius: 16px!important;
  border: 1px solid var(--stroke)!important;
  background: rgba(255,255,255,.86)!important;
}}
/* ring */
.rwrap{{display:flex;gap:12px;align-items:center;}}
.rsvg{{width:62px;height:62px;}}
.rtrk{{stroke:rgba(10,20,40,.10);stroke-width:10;}}
.rprg{{stroke:var(--rc);stroke-width:10;stroke-linecap:round;transform:rotate(-90deg);transform-origin:50% 50%;
  stroke-dasharray:150.7964; stroke-dashoffset: calc(150.7964*(1 - var(--p)));
}}
.rval{{font-weight:900;fill:rgba(10,20,40,.92);font-size:12px;}}
/* last points */
.lp{{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;}}
.pdot{{width:14px;height:14px;border-radius:999px;border:1px solid rgba(10,20,40,.10);}}
.pdot.w{{background:rgba(0,195,137,.95);}}
.pdot.l{{background:rgba(239,68,68,.90);}}
.pdot.pr{{outline:3px solid rgba(245,158,11,.22);}}
</style>
""", unsafe_allow_html=True)


# =========================
# STORAGE
# =========================
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
    with open(history_path_for(user_key), "w", encoding="utf-8") as f:
        json.dump({"matches": matches}, f, ensure_ascii=False, indent=2)


# =========================
# NEWS (RSS)
# =========================
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
    seen, uniq = set(), []
    for it in items:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        uniq.append(it)
    return uniq[:max_items]


# =========================
# TENNIS LOGIC
# =========================
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

    def after_game(ngm, ngo):
        if ngm == 6 and ngo == 6:
            return _prob_set_from(p_rounded, 6, 6, 0, 0, True)
        return _prob_set_from(p_rounded, ngm, ngo, 0, 0, False)

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


# =========================
# LIVE STATE
# =========================
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
            self.state.games_me = self.state.games_opp = 0
            self.state.pts_me = self.state.pts_opp = 0
            self.state.in_tiebreak = False
        elif is_set_over(self.state.games_opp, self.state.games_me):
            self.state.sets_opp += 1
            self.state.games_me = self.state.games_opp = 0
            self.state.pts_me = self.state.pts_opp = 0
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
                self.state.games_me, self.state.games_opp = 7, 6
                self._maybe_award_set()
            elif won_tiebreak(self.state.pts_opp, self.state.pts_me):
                self.state.games_opp, self.state.games_me = 7, 6
                self._maybe_award_set()
            return

        if won_game(self.state.pts_me, self.state.pts_opp):
            self._award_game_to_me()
        elif won_game(self.state.pts_opp, self.state.pts_me):
            self._award_game_to_opp()

    def add_game_manual(self, who: str):
        self.snapshot()
        self._award_game_to_me() if who == "me" else self._award_game_to_opp()

    def add_set_manual(self, who: str):
        self.snapshot()
        if who == "me":
            self.state.sets_me += 1
        else:
            self.state.sets_opp += 1
        self.state.games_me = self.state.games_opp = 0
        self.state.pts_me = self.state.pts_opp = 0
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
            "points_total": total, "points_won": won, "points_pct": pct,
            "pressure_total": pressure_total, "pressure_won": pressure_won,
            "pressure_pct": (pressure_won / pressure_total * 100.0) if pressure_total else 0.0,
            "finishes": finishes,
        }


# =========================
# HISTORY
# =========================
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
        best = cur = 0
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
            "matches_total": total_m, "matches_win": win_m, "matches_pct": self.pct(win_m, total_m),
            "sets_w": sets_w, "sets_l": sets_l, "sets_pct": self.pct(sets_w, sets_w + sets_l),
            "games_w": games_w, "games_l": games_l, "games_pct": self.pct(games_w, games_w + games_l),
            "points_total": points_total, "points_won": points_won, "points_pct": self.pct(points_won, points_total),
            "pressure_total": pressure_total, "pressure_won": pressure_won, "pressure_pct": self.pct(pressure_won, pressure_total),
            "finishes_sum": finishes_sum, "surfaces": surfaces,
        }


# =========================
# COACH SUMMARY (unchanged)
# =========================
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

    strengths, focus = [], []
    if pts_pct >= 55: strengths.append(f"dominaste el intercambio de puntos ({pts_pct:.0f}%).")
    elif pts_pct <= 45 and pts_total >= 10: focus.append(f"subir el % de puntos ganados ({pts_pct:.0f}%).")
    else: strengths.append(f"tu % de puntos estuvo equilibrado ({pts_pct:.0f}%).")

    if pressure_total >= 6:
        if pressure_pct >= 55: strengths.append(f"gestionaste muy bien la presión ({pressure_won}/{pressure_total}, {pressure_pct:.0f}%).")
        elif pressure_pct <= 45: focus.append(f"mejorar puntos de presión ({pressure_won}/{pressure_total}, {pressure_pct:.0f}%).")
        else: strengths.append(f"en presión estuviste parejo ({pressure_won}/{pressure_total}, {pressure_pct:.0f}%).")
    elif pressure_total > 0:
        strengths.append(f"en los pocos puntos de presión estuviste {pressure_won}/{pressure_total}.")
    else:
        strengths.append("hubo pocos puntos de presión registrados.")

    if winners >= max(5, enf + 2): strengths.append("generaste muchos winners y fuiste ofensivo cuando tocaba.")
    if enf >= max(5, winners + 2): focus.append("reducir errores no forzados (ENF) en momentos clave.")
    if df >= 3: focus.append("controlar dobles faltas (ritual de saque + margen).")
    if aces >= 3: strengths.append("el saque fue un arma (aces).")
    if (enf + df) > (winners + aces) and pts_total >= 15:
        focus.append("buscar más margen: altura/profundidad y seleccionar mejor el riesgo.")
    if opp_err >= 5 and winners < 3:
        strengths.append("sacaste puntos provocando error del rival: buena consistencia.")

    plan = [
        "1) Prioriza consistencia (altura/profundidad) y acelera solo con bola clara.",
        "2) En puntos importantes: rutina corta (respira, objetivo simple, juega al %).",
        "3) Saque: 1º con dirección; 2º con más efecto/altura, mismo ritual siempre.",
    ]
    s_txt = " ".join(strengths) if strengths else "buen partido en líneas generales."
    f_txt = " ".join(focus) if focus else "pocos puntos débiles claros: sigue consolidando lo que funcionó."
    return (
        f"**Resumen del entrenador ({res})**\n\n"
        f"- **Qué funcionó:** {s_txt}\n"
        f"- **Qué mejorar:** {f_txt}\n\n"
        f"**Claves para el próximo partido**\n" + "\n".join(plan) + "\n\n"
        f"**Datos rápidos:** Puntos {pts_won}/{pts_total} ({pts_pct:.0f}%) · "
        f"Presión {pressure_won}/{pressure_total} ({pressure_pct:.0f}%) · "
        f"Winners {winners} · ENF {enf} · EF {ef} · Ace {aces} · DF {df} · ErrRival {opp_err}"
    )


def ai_coach_summary_from_match(m: dict) -> str:
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_summary = coach_summary_from_match(m)
    if not api_key:
        return "⚠️ **Resumen IA no disponible** (falta `OPENAI_API_KEY`).\n\n" + base_summary

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
                {"role": "system", "content": "Eres un entrenador de tenis experto. Resumen claro y accionable. No inventes datos."},
                {"role": "user", "content": f"Stats del partido:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"},
            ],
        }
        data = json.dumps(req_payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        txt = obj["choices"][0]["message"]["content"].strip()
        return "🤖 **Resumen IA (coach)**\n\n" + txt
    except Exception as e:
        return f"⚠️ **No se pudo generar el resumen IA**\n\nDetalle: `{e}`\n\n" + base_summary


# =========================
# SESSION STATE INIT
# =========================
def ss_init():
    if "live" not in st.session_state: st.session_state.live = LiveMatch()
    if "history" not in st.session_state: st.session_state.history = MatchHistory()
    if "finish" not in st.session_state: st.session_state.finish = None
    if "page" not in st.session_state: st.session_state.page = "LIVE"
    if "auth_user" not in st.session_state: st.session_state.auth_user = None
    if "auth_key" not in st.session_state: st.session_state.auth_key = None
    if "authed" not in st.session_state: st.session_state.authed = False
    if "_last_p_match" not in st.session_state: st.session_state._last_p_match = None


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
    st.markdown(f"<div class='small'>{txt}</div>", unsafe_allow_html=True)


def ring(label: str, value: float, sub: str = "", color: str = "var(--accent)"):
    v = max(0.0, min(100.0, float(value or 0.0)))
    p = v / 100.0
    st.markdown(f"""
    <div class="card tight">
      <div class="rwrap">
        <svg class="rsvg" viewBox="0 0 64 64" style="--rc:{color};--p:{p}">
          <circle class="rtrk" cx="32" cy="32" r="24" fill="none"></circle>
          <circle class="rprg" cx="32" cy="32" r="24" fill="none"></circle>
          <text class="rval" x="32" y="36" text-anchor="middle">{v:.0f}%</text>
        </svg>
        <div>
          <div style="font-weight:900;">{label}</div>
          <div class="small">{sub}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def last_points_timeline(points, n=18):
    arr = points[-n:] if points else []
    dots = []
    for p in arr:
        cls = "w" if p.get("result") == "win" else "l"
        if p.get("pressure"): cls += " pr"
        dots.append(f"<span class='pdot {cls}' title='{p.get('result','')}'></span>")
    st.markdown(f"""
    <div class="card">
      <div style="font-weight:900;">Últimos puntos</div>
      <div class="small">Verde=ganado · Rojo=perdido · Borde=presión</div>
      <div class="lp">{''.join(dots) if dots else "<span class='small'>Aún no hay puntos.</span>"}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# AUTH (same logic, cleaner UI)
# =========================
def auth_block():
    users = load_users()
    st.markdown(f"""
    <div class="card">
      <div class="h2">FITCOURT — TennisStats</div>
      <div class="sub">Interfaz limpia, estilo “fitness app”. Funciones intactas.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

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
                st.error("Error leyendo credenciales.")
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
            users[key] = {
                "display": new_u.strip(),
                "salt": _b64e(salt),
                "hash": hash_pin(new_pin, salt),
                "created": datetime.now().isoformat(timespec="seconds"),
            }
            save_users(users)
            save_history_to_disk(key, [])
            st.success("Usuario creado ✅ Ya puedes entrar.")


# =========================
# MAIN (login required)
# =========================
if not st.session_state.authed:
    auth_block()
    st.stop()

live: LiveMatch = st.session_state.live
history: MatchHistory = st.session_state.history
user_key = st.session_state.auth_key
user_display = st.session_state.auth_user

# Sidebar (same actions)
with st.sidebar:
    st.markdown("### 🎾 TennisStats")
    st.caption("Panel")
    st.markdown(f"**👤 Usuario:** `{user_display}`")
    st.divider()
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

# Top bar (visual only)
st_ = live.state
pts_label_now = f"TB {st_.pts_me}-{st_.pts_opp}" if st_.in_tiebreak else game_point_label(st_.pts_me, st_.pts_opp)
total_pts, won_pts, pct_pts = live.points_stats()
p_point = live.estimate_point_win_prob()
p_match = live.match_win_prob() * 100.0

st.markdown(f"""
<div class="tsTop">
  <div class="tsTopInner">
    <div class="tsRow">
      <div class="badge"><span class="dot"></span>{user_display}</div>
      <div class="badge"><span class="dot" style="background:var(--accent2);box-shadow:0 0 0 3px rgba(47,107,255,.14);"></span>WinProb <b>{p_match:.1f}%</b></div>
      <div class="badge"><span class="dot" style="background:var(--warn);box-shadow:0 0 0 3px rgba(245,158,11,.14);"></span>Sets <b>{st_.sets_me}-{st_.sets_opp}</b></div>
      <div class="badge"><span class="dot" style="background:var(--danger);box-shadow:0 0 0 3px rgba(239,68,68,.12);"></span>Juegos <b>{st_.games_me}-{st_.games_opp}</b></div>
      <div class="badge"><span class="dot" style="background:rgba(10,20,40,.40);box-shadow:none;"></span>Puntos <b>{pts_label_now}</b></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Dashboard quick rings (same numbers)
c1, c2 = st.columns(2, gap="small")
with c1: ring("Puntos ganados", pct_pts, f"{won_pts}/{total_pts}", "var(--accent)")
with c2: ring("Prob. victoria", p_match, "Modelo Markov", "var(--accent2)")


# =========================
# PAGES
# =========================
if st.session_state.page == "LIVE":
    st.markdown("<div class='card'><div class='h2'>LIVE MATCH</div><div class='sub'>Registro rápido y claro (sin tocar lógica)</div></div>", unsafe_allow_html=True)

    a, b = st.columns([1.05, 0.95], gap="small")
    with a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        live.surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(live.surface))
        small_note("Tip: usa el sidebar para cambiar de sección.")
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        pts_label = f"TB {st_.pts_me}-{st_.pts_opp}" if st_.in_tiebreak else game_point_label(st_.pts_me, st_.pts_opp)
        st.markdown(f"<div style='font-weight:900;'>Marcador</div><div class='small mono'>Sets {st_.sets_me}-{st_.sets_opp} · Juegos {st_.games_me}-{st_.games_opp} · Puntos {pts_label}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    last_points_timeline(live.points, n=18)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Registrar punto", anchor=False)
    r1, r2 = st.columns(2, gap="small")
    with r1:
        if st.button("🟩 Punto Yo", use_container_width=True):
            live.add_point("win", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()
    with r2:
        if st.button("🟥 Punto Rival", use_container_width=True):
            live.add_point("lose", {"finish": st.session_state.finish})
            st.session_state.finish = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Acciones manuales", anchor=False)
    m1, m2 = st.columns(2, gap="small")
    with m1:
        if st.button("➕ Juego Yo", use_container_width=True):
            live.add_game_manual("me"); st.rerun()
        if st.button("➕ Set Yo", use_container_width=True):
            live.add_set_manual("me"); st.rerun()
    with m2:
        if st.button("➕ Juego Rival", use_container_width=True):
            live.add_game_manual("opp"); st.rerun()
        if st.button("➕ Set Rival", use_container_width=True):
            live.add_set_manual("opp"); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Finish (opcional)", anchor=False)
    small_note("Selecciona 1 (se aplica al siguiente punto). Toca de nuevo para deseleccionar.")
    fcols = st.columns(2, gap="small")
    for i, (key, label) in enumerate(FINISH_ITEMS):
        with fcols[i % 2]:
            selected = (st.session_state.finish == key)
            txt = f"✅ {label}" if selected else label
            if st.button(txt, key=f"finish_{key}", use_container_width=True):
                st.session_state.finish = None if selected else key
                st.rerun()
    x1, x2 = st.columns([1, 1], gap="small")
    with x1:
        if st.button("🧼 Limpiar", use_container_width=True):
            st.session_state.finish = None; st.rerun()
    with x2:
        small_note(f"Seleccionado: **{st.session_state.finish or '—'}**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Acciones", anchor=False)
    a1, a2, a3 = st.columns(3, gap="small")
    with a1:
        if st.button("↩️ Deshacer", use_container_width=True):
            live.undo(); st.rerun()
    with a2:
        if st.button("📈 Ir a Analysis", use_container_width=True):
            st.session_state.page = "ANALYSIS"; st.rerun()
    with a3:
        if st.button("🏁 Finalizar", use_container_width=True):
            st.session_state._open_finish = True

    if st.session_state.get("_open_finish", False):
        with st.expander("Finalizar partido", expanded=True):
            sw = st.number_input("Sets Yo", 0, 5, value=int(live.state.sets_me), step=1)
            sl = st.number_input("Sets Rival", 0, 5, value=int(live.state.sets_opp), step=1)
            gw = st.number_input("Juegos Yo", 0, 50, value=int(live.state.games_me), step=1)
            gl = st.number_input("Juegos Rival", 0, 50, value=int(live.state.games_opp), step=1)
            surf_save = st.selectbox("Superficie (guardar)", SURFACES, index=SURFACES.index(live.surface))
            sL, sR = st.columns(2, gap="small")
            with sL:
                if st.button("Cancelar", use_container_width=True):
                    st.session_state._open_finish = False; st.rerun()
            with sR:
                if st.button("Guardar partido", use_container_width=True):
                    report = live.match_summary()
                    history.add({
                        "id": f"m_{datetime.now().timestamp()}",
                        "date": datetime.now().isoformat(timespec="seconds"),
                        "won_match": bool(sw > sl),
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

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Historial y exportación", anchor=False)
    small_note("Historial privado por usuario. Puedes borrar/editar, exportar/importar JSON.")
    if not history.matches:
        st.info("Aún no hay partidos guardados.")
    else:
        matches = list(reversed(history.matches))
        for idx, m in enumerate(matches):
            real_i = len(history.matches) - 1 - idx
            res = "✅ W" if m.get("won_match") else "❌ L"
            score = f"{m.get('sets_w',0)}-{m.get('sets_l',0)} sets · {m.get('games_w',0)}-{m.get('games_l',0)} juegos"
            surf = m.get("surface", "—")
            date = m.get("date", "")
            with st.expander(f"{res} · {score} · {surf} · {date}", expanded=False):
                st.write(f"**{score}**")
                small_note(f"Puntos: {m.get('points_won',0)}/{m.get('points_total',0)} ({m.get('points_pct',0):.0f}%) · "
                           f"Presión: {m.get('pressure_won',0)}/{m.get('pressure_total',0)} ({m.get('pressure_pct',0):.0f}%)")
                fin = (m.get("finishes") or {})
                small_note(f"Winners {fin.get('winner',0)} · ENF {fin.get('unforced',0)} · EF {fin.get('forced',0)} · "
                           f"Ace {fin.get('ace',0)} · DF {fin.get('double_fault',0)}")
                if st.button("🤖 Resumen IA (coach)", key=f"coach_{m.get('id',real_i)}", use_container_width=True):
                    st.session_state._coach_open = True
                    st.session_state._coach_text = ai_coach_summary_from_match(m)
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
            with st.expander("🤖 Resumen IA", expanded=True):
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
                    col1, col2 = st.columns(2, gap="small")
                    with col1:
                        won_match = st.toggle("Victoria", value=bool(m.get("won_match", False)))
                        sets_w = st.number_input("Sets Yo", 0, 5, value=int(m.get("sets_w", 0)), step=1)
                        games_w = st.number_input("Juegos Yo", 0, 50, value=int(m.get("games_w", 0)), step=1)
                    with col2:
                        sets_l = st.number_input("Sets Rival", 0, 5, value=int(m.get("sets_l", 0)), step=1)
                        games_l = st.number_input("Juegos Rival", 0, 50, value=int(m.get("games_l", 0)), step=1)
                        surface = st.selectbox("Superficie", SURFACES, index=SURFACES.index(m.get("surface", SURFACES[0])))
                    date = st.text_input("Fecha (ISO)", value=str(m.get("date", "")))
                    bL, bR = st.columns(2, gap="small")
                    with bL:
                        if st.button("Cancelar edición", use_container_width=True):
                            st.session_state._edit_open = False
                            st.session_state._edit_index = None
                            st.rerun()
                    with bR:
                        if st.button("Guardar cambios", use_container_width=True):
                            m["won_match"] = bool(won_match)
                            m["sets_w"] = int(sets_w); m["sets_l"] = int(sets_l)
                            m["games_w"] = int(games_w); m["games_l"] = int(games_l)
                            m["surface"] = surface; m["date"] = date
                            history.matches[i] = m
                            save_history_to_disk(user_key, history.matches)
                            st.session_state._edit_open = False
                            st.session_state._edit_index = None
                            st.success("Cambios guardados ✅")
                            st.rerun()

    export_json = json.dumps({"matches": history.matches}, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("⬇️ Descargar historial (JSON)", data=export_json,
                       file_name=f"tennis_history__{user_key}.json", mime="application/json",
                       use_container_width=True)

    up = st.file_uploader("⬆️ Importar historial (JSON)", type=["json"])
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

elif st.session_state.page == "ANALYSIS":
    st.markdown("<div class='card'><div class='h2'>Analysis</div><div class='sub'>Modelo real: Markov (punto→juego→set→BO3)</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    small_note(f"p(punto)≈{p_point:.2f} · Win Prob≈{p_match:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    probs = live.win_prob_series()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Evolución Win Prob", anchor=False)
    if len(probs) < 2:
        st.info("Aún no hay suficientes puntos (mínimo 2).")
    else:
        st.area_chart(probs, height=280)
    st.markdown("</div>", unsafe_allow_html=True)

    pressure_total = sum(1 for p in live.points if p.get("pressure"))
    pressure_won = sum(1 for p in live.points if p.get("pressure") and p.get("result") == "win")
    pressure_pct = (pressure_won / pressure_total * 100.0) if pressure_total else 0.0
    ring("Presión", pressure_pct, f"{pressure_won}/{pressure_total} ganados", "var(--warn)")

elif st.session_state.page == "STATS":
    st.markdown("<div class='card'><div class='h2'>Stats</div><div class='sub'>Historial · Superficies · Rachas</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    colF1, colF2 = st.columns([1.1, 0.9], gap="small")
    with colF1:
        n_choice = st.selectbox("Rango", ["Últ. 10", "Últ. 30", "Todos"], index=0)
    with colF2:
        surf_filter = st.selectbox("Superficie", ["Todas", *SURFACES], index=0)
    n = 10 if n_choice == "Últ. 10" else (30 if n_choice == "Últ. 30" else None)
    agg = history.aggregate(n=n, surface=surf_filter)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    with c1: ring("Partidos", agg["matches_pct"], f"{agg['matches_win']} / {agg['matches_total']}", "var(--accent)")
    with c2: ring("Sets", agg["sets_pct"], f"{agg['sets_w']} / {agg['sets_w'] + agg['sets_l']}", "var(--accent2)")
    with c3: ring("Juegos", agg["games_pct"], f"{agg['games_w']} / {agg['games_w'] + agg['games_l']}", "var(--warn)")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Resumen", anchor=False)
    st.write(f"**Puntos:** {agg['points_won']}/{agg['points_total']} ({agg['points_pct']:.0f}%) · "
             f"**Presión:** {agg['pressure_won']}/{agg['pressure_total']} ({agg['pressure_pct']:.0f}%)")
    fin = agg["finishes_sum"]
    small_note(f"Winners {fin['winner']} · ENF {fin['unforced']} · EF {fin['forced']} · Aces {fin['ace']} · DF {fin['double_fault']}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Rachas", anchor=False)
    results = history.last_n_results(10, surface=(None if surf_filter == "Todas" else surf_filter))
    st.write(" · ".join(["✅ W" if r == "W" else "⬛ L" for r in results]) if results else "—")
    st.write(f"**🔥 Mejor racha:** {history.best_streak(surface=(None if surf_filter == 'Todas' else surf_filter))} victorias seguidas")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Superficies", anchor=False)
    surf = agg["surfaces"]
    chart_data = {}
    for srf in SURFACES:
        w = surf.get(srf, {}).get("w", 0)
        t_ = surf.get(srf, {}).get("t", 0)
        pct = (w / t_ * 100.0) if t_ else 0.0
        st.write(f"**{srf}:** {pct:.0f}%  ({w} de {t_})")
        chart_data[srf] = pct
    st.bar_chart(chart_data, height=260) if any(v > 0 for v in chart_data.values()) else small_note("Aún no hay datos suficientes.")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "NEWS":
    st.markdown("<div class='card'><div class='h2'>Noticias</div><div class='sub'>RSS de tenis</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cL, cR = st.columns([1, 1], gap="small")
    with cL:
        max_items = st.selectbox("Cuántas noticias", [8, 12, 15, 20], index=1)
    with cR:
        if st.button("🔄 Actualizar", use_container_width=True):
            fetch_tennis_news.clear(); st.rerun()
    news = fetch_tennis_news(int(max_items))
    if not news:
        st.info("No se pudieron cargar noticias ahora mismo.")
    else:
        for it in news:
            src = it.get("source", "—")
            title = it.get("title", "Noticia")
            link = it.get("link", "#")
            pub = it.get("published", "")
            st.markdown(f"- **[{title}]({link})**  \n  <span class='small'>{src}{(' · ' + pub) if pub else ''}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:  # PSICO
    st.markdown("<div class='card'><div class='h2'>Psico</div><div class='sub'>PDFs en la carpeta psico_pdfs/</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    psico_dir = Path("psico_pdfs")
    pdfs = sorted([p for p in psico_dir.glob("*.pdf") if p.is_file()], key=lambda x: x.name.lower()) if psico_dir.exists() else []
    if not pdfs:
        st.info("No se han encontrado PDFs en `psico_pdfs/`.")
    else:
        for p in pdfs:
            k = hashlib.md5(p.name.encode("utf-8")).hexdigest()[:10]
            with st.expander(f"📄 {p.name}", expanded=False):
                try:
                    data = p.read_bytes()
                except Exception as e:
                    st.error(f"No se pudo leer el PDF: {e}")
                    continue
                st.download_button("⬇️ Descargar PDF", data=data, file_name=p.name, mime="application/pdf",
                                   use_container_width=True, key=f"psico_dl_{k}")
                b64 = base64.b64encode(data).decode("utf-8")
                st.components.v1.html(
                    f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='650' "
                    f"style='border:1px solid rgba(10,20,40,.12);border-radius:16px;background:#fff;'></iframe>",
                    height=680, scrolling=False
                )
    st.markdown("</div>", unsafe_allow_html=True)
