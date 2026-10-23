"""
HUMAN NEXT visual theme: palette constants and global CSS.

No external fonts or assets are loaded, so the app works offline and makes
no requests beyond localhost. Animations are subtle and are switched off
when the operating system asks for reduced motion.
"""

from __future__ import annotations

import streamlit as st

PALETTE = {
    "bg": "#07080D",
    "bg2": "#0D101A",
    "card": "#111522",
    "blue": "#36C5FF",
    "cyan": "#65F2FF",
    "magenta": "#FF38D1",
    "purple": "#8B5CF6",
    "text": "#F4F7FF",
    "muted": "#9AA6BD",
    "success": "#4ADE80",
    "warning": "#FBBF24",
    "error": "#FB7185",
}

BRAND = "HUMAN NEXT"
TAGLINE = "Decode Your Interview Potential"

_CSS = """
<style>
:root{
  --hn-bg:%(bg)s;--hn-bg2:%(bg2)s;--hn-card:%(card)s;--hn-blue:%(blue)s;--hn-cyan:%(cyan)s;
  --hn-magenta:%(magenta)s;--hn-purple:%(purple)s;--hn-text:%(text)s;--hn-muted:%(muted)s;
  --hn-ok:%(success)s;--hn-warn:%(warning)s;--hn-err:%(error)s;
  --hn-grad:linear-gradient(100deg,var(--hn-blue) 0%%,var(--hn-purple) 55%%,var(--hn-magenta) 100%%);
  --hn-border:rgba(101,242,255,.12);--hn-radius:16px;
  --hn-font:"Segoe UI Variable Text","Segoe UI",Inter,system-ui,-apple-system,sans-serif;
}
html,body,[data-testid="stAppViewContainer"]{background:
  radial-gradient(1200px 600px at 85%% -10%%,rgba(139,92,246,.16),transparent 60%%),
  radial-gradient(900px 500px at -10%% 10%%,rgba(54,197,255,.10),transparent 60%%),
  var(--hn-bg)!important;color:var(--hn-text);font-family:var(--hn-font)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none!important}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1280px}
h1,h2,h3,h4{font-family:var(--hn-font);letter-spacing:-.01em;color:var(--hn-text)}
p,li,label,.stMarkdown{color:var(--hn-text)}
small,.hn-muted{color:var(--hn-muted)}

/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid var(--hn-border);
  background:rgba(17,21,34,.85);color:var(--hn-text);font-weight:600;transition:all .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--hn-blue);
  box-shadow:0 0 0 1px rgba(54,197,255,.35),0 0 18px rgba(54,197,255,.18);color:var(--hn-text)}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primaryFormSubmit"]{
  background:var(--hn-grad);border:none;color:#05060A;box-shadow:0 6px 24px rgba(139,92,246,.30)}
.stButton>button[kind="primary"]:hover,.stFormSubmitButton>button[kind="primaryFormSubmit"]:hover{
  filter:brightness(1.08);color:#05060A;box-shadow:0 8px 30px rgba(255,56,209,.30)}
[data-testid="stForm"]{border:1px solid var(--hn-border);border-radius:16px;background:rgba(17,21,34,.6)}
.stButton>button:disabled{opacity:.45;box-shadow:none;filter:grayscale(.4)}
.stButton>button:focus-visible{outline:2px solid var(--hn-cyan);outline-offset:2px}

/* inputs */
[data-baseweb="select"]>div,.stTextInput input,.stMultiSelect [data-baseweb="select"]>div{
  background:var(--hn-card)!important;border-color:var(--hn-border)!important;border-radius:12px!important}
[data-testid="stMetric"]{background:rgba(17,21,34,.7);border:1px solid var(--hn-border);
  border-radius:14px;padding:12px 14px}
[data-testid="stMetricLabel"] p{color:var(--hn-muted)!important;font-size:.8rem}
.stTabs [data-baseweb="tab"]{color:var(--hn-muted)}
.stTabs [aria-selected="true"]{color:var(--hn-text)!important}
.stProgress>div>div>div>div{background:var(--hn-grad)!important}
[data-testid="stExpander"]{background:rgba(17,21,34,.6);border:1px solid var(--hn-border);border-radius:14px}
[data-testid="stImage"] img{border-radius:14px;border:1px solid rgba(54,197,255,.25)}

/* brand bar */
.hn-bar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 18px}
.hn-logo{display:flex;align-items:center;gap:10px;font-weight:800;letter-spacing:.22em;font-size:.86rem}
.hn-logo-mark{width:28px;height:28px;border-radius:9px;background:var(--hn-grad);
  box-shadow:0 0 18px rgba(101,242,255,.35);position:relative}
.hn-logo-mark:after{content:"";position:absolute;inset:7px;border-radius:50%%;border:2px solid #05060A}
.hn-bar-right{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}

/* cards */
.hn-card{background:linear-gradient(180deg,rgba(17,21,34,.92),rgba(13,16,26,.92));
  border:1px solid var(--hn-border);border-radius:var(--hn-radius);padding:18px 20px;margin:0 0 14px;
  box-shadow:0 10px 30px rgba(0,0,0,.35);backdrop-filter:blur(8px)}
.hn-card.eq{min-height:230px}.hn-card.eq-lg{min-height:300px}
@media (max-width: 760px){.hn-card.eq,.hn-card.eq-lg{min-height:0}}
.hn-card.glow{box-shadow:0 0 0 1px rgba(54,197,255,.18),0 12px 40px rgba(54,197,255,.10)}
.hn-card h4{margin:0 0 8px;font-size:1rem}
.hn-card [data-testid="stHeaderActionElements"],.hn-hero [data-testid="stHeaderActionElements"],
.hn-stage [data-testid="stHeaderActionElements"]{display:none}
.hn-label{font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;color:var(--hn-cyan);font-weight:700;margin-bottom:6px}
.hn-card p{color:var(--hn-muted);margin:.2rem 0;line-height:1.55}

/* hero */
.hn-hero{position:relative;overflow:hidden;border-radius:24px;padding:56px 44px;margin:6px 0 22px;
  background:radial-gradient(600px 300px at 80%% 20%%,rgba(255,56,209,.18),transparent 60%%),
             radial-gradient(700px 360px at 10%% 90%%,rgba(54,197,255,.16),transparent 60%%),
             linear-gradient(160deg,#0B0E18,#07080D);border:1px solid var(--hn-border)}
.hn-hero:before{content:"";position:absolute;inset:0;background-image:
  linear-gradient(rgba(101,242,255,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(101,242,255,.06) 1px,transparent 1px);
  background-size:42px 42px;mask-image:radial-gradient(ellipse at 70%% 40%%,#000 20%%,transparent 70%%)}
.hn-hero>*{position:relative}
.hn-hero h1{font-size:clamp(2.1rem,4.4vw,3.6rem);line-height:1.05;margin:10px 0 14px;font-weight:800}
.hn-grad-text{background:var(--hn-grad);background-size:200%% 100%%;-webkit-background-clip:text;
  background-clip:text;color:transparent;animation:hn-shift 8s ease-in-out infinite alternate}
.hn-hero p{color:var(--hn-muted);font-size:1.05rem;max-width:620px;line-height:1.6}
.hn-orb{position:absolute;right:6%%;top:50%%;width:220px;height:220px;transform:translateY(-50%%);border-radius:50%%;
  background:radial-gradient(circle at 35%% 35%%,rgba(101,242,255,.55),rgba(139,92,246,.35) 45%%,rgba(255,56,209,.15) 70%%,transparent 72%%);
  filter:blur(1px);box-shadow:0 0 80px rgba(139,92,246,.35)}
.hn-orb:after{content:"";position:absolute;inset:-14px;border-radius:50%%;border:1px dashed rgba(101,242,255,.35);
  animation:hn-spin 40s linear infinite}

/* chips */
.hn-chip{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-size:.72rem;
  font-weight:700;letter-spacing:.06em;text-transform:uppercase;border:1px solid var(--hn-border);
  background:rgba(17,21,34,.85);color:var(--hn-muted);white-space:nowrap}
.hn-chip .dot{width:7px;height:7px;border-radius:50%%;background:currentColor}
.hn-chip.live{color:var(--hn-blue);border-color:rgba(54,197,255,.45)}
.hn-chip.rec{color:var(--hn-magenta);border-color:rgba(255,56,209,.55);background:rgba(255,56,209,.08)}
.hn-chip.live .dot,.hn-chip.rec .dot{animation:hn-pulse 1.3s ease-in-out infinite}
.hn-chip.ok{color:var(--hn-ok);border-color:rgba(74,222,128,.4)}
.hn-chip.warn{color:var(--hn-warn);border-color:rgba(251,191,36,.4)}
.hn-chip.err{color:var(--hn-err);border-color:rgba(251,113,133,.45)}
.hn-chip.measured{color:var(--hn-cyan);border-color:rgba(101,242,255,.35)}
.hn-chip.model{color:#C4B5FD;border-color:rgba(139,92,246,.45)}
.hn-chip.na{color:var(--hn-muted);border-style:dashed}

/* stepper */
.hn-steps{display:flex;gap:6px;flex-wrap:wrap;margin:2px 0 18px}
.hn-step{flex:1 1 120px;padding:8px 12px;border-radius:12px;border:1px solid var(--hn-border);
  font-size:.78rem;color:var(--hn-muted);background:rgba(17,21,34,.6)}
.hn-step b{display:block;font-size:.66rem;letter-spacing:.16em;color:var(--hn-muted)}
.hn-step.done{color:var(--hn-text)}.hn-step.done b{color:var(--hn-ok)}
.hn-step.now{color:var(--hn-text);border-color:rgba(54,197,255,.6);box-shadow:0 0 16px rgba(54,197,255,.15)}
.hn-step.now b{color:var(--hn-blue)}
.hn-steps.compact{flex-wrap:nowrap;gap:4px;margin:4px 0 12px}
.hn-steps.compact .hn-step{flex:1 1 0;min-width:0;padding:6px 8px;font-size:.72rem;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* bars */
.hn-row{margin:10px 0}
.hn-row-head{display:flex;justify-content:space-between;gap:10px;font-size:.86rem;margin-bottom:5px}
.hn-row-head span:last-child{font-weight:700;font-variant-numeric:tabular-nums}
.hn-track{height:8px;border-radius:99px;background:rgba(154,166,189,.14);overflow:hidden}
.hn-fill{height:100%%;border-radius:99px;background:var(--hn-grad);transition:width .5s ease}
.hn-fill.blue{background:linear-gradient(90deg,var(--hn-blue),var(--hn-cyan))}
.hn-track.na{background:repeating-linear-gradient(45deg,rgba(154,166,189,.10) 0 6px,transparent 6px 12px)}
.hn-row-note{font-size:.76rem;color:var(--hn-muted);margin-top:4px;line-height:1.4}

/* score ring */
.hn-ring{--v:0;width:150px;height:150px;border-radius:50%%;display:grid;place-items:center;margin:4px auto 8px;
  background:conic-gradient(var(--hn-magenta) 0,var(--hn-purple) calc(var(--v)*0.5%%),var(--hn-blue) calc(var(--v)*1%%),
  rgba(154,166,189,.14) calc(var(--v)*1%%));box-shadow:0 0 40px rgba(139,92,246,.22)}
.hn-ring>div{width:118px;height:118px;border-radius:50%%;background:var(--hn-bg2);display:grid;place-items:center;text-align:center}
.hn-ring b{font-size:2.2rem;line-height:1}.hn-ring small{display:block;color:var(--hn-muted);font-size:.7rem;letter-spacing:.12em}
.hn-ring.na{background:repeating-conic-gradient(rgba(154,166,189,.16) 0 6deg,transparent 6deg 12deg)}

/* video standby */
.hn-stage{aspect-ratio:16/10;border-radius:16px;border:1px solid rgba(54,197,255,.25);display:grid;place-items:center;
  background:radial-gradient(circle at 50%% 45%%,rgba(54,197,255,.10),transparent 60%%),var(--hn-bg2);text-align:center;padding:20px}
.hn-stage .ring{width:86px;height:86px;border-radius:50%%;border:2px dashed rgba(101,242,255,.45);margin:0 auto 12px;
  animation:hn-spin 24s linear infinite}
.hn-stage p{color:var(--hn-muted);margin:0}
.hn-big{font-size:2.6rem;font-weight:800;font-variant-numeric:tabular-nums}

.hn-list{margin:6px 0 0 0;padding-left:18px;color:var(--hn-text)}
.hn-list li{margin:4px 0;color:var(--hn-text)}
.hn-legend{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 12px}

@keyframes hn-pulse{0%%,100%%{opacity:1}50%%{opacity:.25}}
@keyframes hn-spin{to{transform:rotate(360deg)}}
@keyframes hn-shift{from{background-position:0%% 50%%}to{background-position:100%% 50%%}}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
@media (max-width: 760px){.hn-hero{padding:36px 22px}.hn-orb{display:none}.hn-ring{width:120px;height:120px}
  .hn-ring>div{width:94px;height:94px}}
</style>
"""


def inject_theme() -> None:
    """Add the global stylesheet to the page (call once per script run)."""
    st.markdown(_CSS % PALETTE, unsafe_allow_html=True)
