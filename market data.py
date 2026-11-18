#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
import time
import types
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=Warning, message='.*Glyph.*missing.*')
warnings.filterwarnings('ignore', category=Warning, message='.*highly fragmented.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')

# ============================================================
# INLINED ZONE-LEARNING MODULE
# (equivalent to `import zone_learning as zl`)
# ============================================================
_ZL_SOURCE = r'''
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

CFG = {
    "OTE_FIB_LOW": 0.62, "OTE_FIB_HIGH": 0.79, "OTE_FIB_IDEAL": 0.705,
    "OTE_DISP_MULT": 1.5, "OTE_LEG_LOOKBACK": 20,
    "KILL_ZONES_UTC": {"London KZ": (7, 10), "NY AM KZ": (12, 15), "Silver Bullet": (14, 15)},
    "HOURLY_MIN_DIST_PCT": 0.0003, "HOURLY_MAX_DIST_PCT": 0.02,
    "HOURLY_STOP_BUFFER_PCT": 0.0005,
    "ZONE_TRACK_AGE": 120,
    "ZONE_MAX_AGE": 96,
    "FILL_HORIZON": 24,
    "REACT_WINDOW": 12,
    "REACT_ATR": 1.0,
    "BREAK_ATR": 1.0,
    "TRAIN_STRIDE": 6, "MAX_TRAIN_ROWS": 150000, "MIN_TOUCHED": 1500, "MIN_AUC": 0.53,
    "STRONG_QUANTILE": 0.66,
    "OTE_PRIOR": 0.0,
    "MIN_RR": 1.0,
}

def configure(**kw):
    for k, v in kw.items():
        if k in CFG:
            CFG[k] = v

KINDS = ["fvg", "ob", "ifvg", "breaker", "sweep", "eqhl", "imbalance"]
_KIND_IDX = {k: i for i, k in enumerate(KINDS)}
_LABELS = {"fvg": "FVG", "ob": "OB", "ifvg": "IFVG", "breaker": "Breaker",
           "sweep": "Liquidity Sweep", "eqhl": "Equal High/Low", "imbalance": "Imbalance"}
FEATURES = (["k_" + k for k in KINDS] +
            ["dir", "height_atr", "dist_edge_atr", "dist_mid_atr", "dist_pct",
             "appr3", "appr12", "age", "touches", "mid_hits", "size_c", "kz_c",
             "hr_sin", "hr_cos", "fib_r", "has_leg", "in_ote", "d705", "conf_n",
             "pd_pos", "atr_pct", "trend_align", "struct_align", "mom24"])

EQ_TOL = 0.0015
_MODELS = {}

def _session(hour):
    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 12:
        return "London"
    if 12 <= hour < 21:
        return "New York"
    return "Asia (late)"

def _in_kz(hour):
    for name, (s, e) in CFG["KILL_ZONES_UTC"].items():
        if s <= hour < e:
            return name
    return None

def _prep(h):
    o = h["open"].values.astype(float)
    hi = h["high"].values.astype(float)
    lo = h["low"].values.astype(float)
    c = h["close"].values.astype(float)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
    atr = pd.Series(tr).rolling(14, min_periods=5).mean().bfill().values
    body = c - o
    avgb = pd.Series(np.abs(body)).rolling(20, min_periods=5).mean().shift(1).values
    return {
        "o": o, "hi": hi, "lo": lo, "c": c, "atr": atr, "body": body, "avgb": avgb,
        "ema_f": pd.Series(c).ewm(span=20, adjust=False).mean().values,
        "ema_s": pd.Series(c).ewm(span=100, adjust=False).mean().values,
        "rmax": pd.Series(hi).rolling(50, min_periods=10).max().values,
        "rmin": pd.Series(lo).rolling(50, min_periods=10).min().values,
        "sh": pd.Series(hi).rolling(7, center=True, min_periods=7).max().values == hi,
        "sl": pd.Series(lo).rolling(7, center=True, min_periods=7).min().values == lo,
        "hrs": h["hour_utc"].values.astype(int),
    }

def _struct_state(sh_conf, sl_conf):
    if len(sh_conf) < 2 or len(sl_conf) < 2:
        return 0
    if sh_conf[-1][1] > sh_conf[-2][1] and sl_conf[-1][1] > sl_conf[-2][1]:
        return 1
    if sh_conf[-1][1] < sh_conf[-2][1] and sl_conf[-1][1] < sl_conf[-2][1]:
        return -1
    return 0

def _walk(h, stride=1, last_only=False, dist_filter=True, start=60):
    P = _prep(h)
    o, hi, lo, c = P["o"], P["hi"], P["lo"], P["c"]
    atr, body, avgb, hrs = P["atr"], P["body"], P["avgb"], P["hrs"]
    n = len(c)
    TA, MA = CFG["ZONE_TRACK_AGE"], CFG["ZONE_MAX_AGE"]
    dmin, dmax = CFG["HOURLY_MIN_DIST_PCT"], CFG["HOURLY_MAX_DIST_PCT"]
    FLO, FHI, FID = CFG["OTE_FIB_LOW"], CFG["OTE_FIB_HIGH"], CFG["OTE_FIB_IDEAL"]
    DM, LB = CFG["OTE_DISP_MULT"], CFG["OTE_LEG_LOOKBACK"]
    fwd = CFG["FILL_HORIZON"] + CFG["REACT_WINDOW"] + 2

    zones, open_fvgs, obc = [], [], []
    sh_conf, sl_conf = [], []
    last_sh = last_sl = np.nan
    last_sh_k = last_sl_k = -1
    swept_sh = swept_sl = -2
    bleg = sleg = None
    rows, metas = [], []

    def add(i, kind, d, top, bottom, mid, size, hour_c):
        if not (np.isfinite(top) and np.isfinite(bottom) and np.isfinite(mid)) or top < bottom:
            return
        zones.append({"kind": kind, "dir": d, "top": float(top), "bottom": float(bottom),
                      "mid": float(mid), "created": i, "hour_c": int(hour_c),
                      "size_c": float(size), "touches": 0, "mid_hits": 0})

    for i in range(n):
        ci = c[i]
        a = atr[i] if atr[i] > 0 else 1e-9

        alive = []
        for z in zones:
            if i - z["created"] > TA:
                continue
            if z["dir"] > 0:
                if lo[i] <= z["top"]:
                    z["touches"] += 1
                if lo[i] <= z["mid"]:
                    z["mid_hits"] += 1
                if ci < z["bottom"]:
                    continue
            else:
                if hi[i] >= z["bottom"]:
                    z["touches"] += 1
                if hi[i] >= z["mid"]:
                    z["mid_hits"] += 1
                if ci > z["top"]:
                    continue
            alive.append(z)
        zones[:] = alive

        if bleg is not None:
            if lo[i] < bleg[0]:
                bleg = None
            else:
                bleg[1] = max(bleg[1], hi[i])
        if sleg is not None:
            if hi[i] > sleg[0]:
                sleg = None
            else:
                sleg[1] = min(sleg[1], lo[i])

        keep = []
        for f in open_fvgs:
            if i - f["created"] > TA:
                continue
            if f["created"] < i:
                if f["dir"] > 0 and ci < f["bottom"]:
                    add(i, "ifvg", -1, f["top"], f["bottom"], (f["top"] + f["bottom"]) / 2,
                        (f["top"] - f["bottom"]) / a, hrs[i])
                    continue
                if f["dir"] < 0 and ci > f["top"]:
                    add(i, "ifvg", 1, f["top"], f["bottom"], (f["top"] + f["bottom"]) / 2,
                        (f["top"] - f["bottom"]) / a, hrs[i])
                    continue
            keep.append(f)
        open_fvgs[:] = keep

        if i >= 2:
            if lo[i] > hi[i - 2]:
                add(i, "fvg", 1, lo[i], hi[i - 2], (lo[i] + hi[i - 2]) / 2, (lo[i] - hi[i - 2]) / a, hrs[i])
                open_fvgs.append({"dir": 1, "top": lo[i], "bottom": hi[i - 2], "created": i})
            if hi[i] < lo[i - 2]:
                add(i, "fvg", -1, lo[i - 2], hi[i], (lo[i - 2] + hi[i]) / 2, (lo[i - 2] - hi[i]) / a, hrs[i])
                open_fvgs.append({"dir": -1, "top": lo[i - 2], "bottom": hi[i], "created": i})

        nobc = []
        for cd in obc:
            if i - cd["disp"] > 20:
                continue
            if i > cd["disp"]:
                k = cd["ob"]
                if cd["dir"] > 0 and ci < lo[k]:
                    bot, top = min(o[k], c[k]), hi[k]
                    add(i, "breaker", -1, top, bot, (top + bot) / 2, (hi[k] - lo[k]) / a, hrs[i])
                    continue
                if cd["dir"] < 0 and ci > hi[k]:
                    top, bot = max(o[k], c[k]), lo[k]
                    add(i, "breaker", 1, top, bot, (top + bot) / 2, (hi[k] - lo[k]) / a, hrs[i])
                    continue
            nobc.append(cd)
        obc[:] = nobc

        ab = avgb[i]
        if ab == ab and ab > 0 and abs(body[i]) > DM * ab:
            d = 1 if body[i] > 0 else -1
            s = abs(body[i]) / ab
            top_b, bot_b = max(o[i], ci), min(o[i], ci)
            add(i, "imbalance", d, top_b, bot_b, (top_b + bot_b) / 2, s, hrs[i])
            if i >= 1 and body[i - 1] * d < 0:
                add(i, "ob", d, hi[i - 1], lo[i - 1], (hi[i - 1] + lo[i - 1]) / 2, s, hrs[i - 1])
                obc.append({"ob": i - 1, "disp": i, "dir": d})
            st = max(0, i - LB)
            if d > 0:
                org = st + int(np.argmin(lo[st:i + 1]))
                bleg = [lo[org], hi[i]]
            else:
                org = st + int(np.argmax(hi[st:i + 1]))
                sleg = [hi[org], lo[i]]

        k = i - 3
        if k >= 3:
            if P["sh"][k]:
                lvl = hi[k]
                for pk, pl in sh_conf[-8:]:
                    if k - pk <= 200 and abs(lvl - pl) / pl <= EQ_TOL:
                        mean = (lvl + pl) / 2
                        if ci < mean and not any(z["kind"] == "eqhl" and z["dir"] < 0 and
                                                 abs(z["mid"] - mean) / mean < 0.0008 for z in zones):
                            add(i, "eqhl", -1, mean * (1 + EQ_TOL), mean, mean, 2.0, hrs[k])
                        break
                sh_conf.append((k, lvl))
                sh_conf[:] = sh_conf[-20:]
                last_sh, last_sh_k = lvl, k
            if P["sl"][k]:
                lvl = lo[k]
                for pk, pl in sl_conf[-8:]:
                    if k - pk <= 200 and abs(lvl - pl) / pl <= EQ_TOL:
                        mean = (lvl + pl) / 2
                        if ci > mean and not any(z["kind"] == "eqhl" and z["dir"] > 0 and
                                                 abs(z["mid"] - mean) / mean < 0.0008 for z in zones):
                            add(i, "eqhl", 1, mean, mean * (1 - EQ_TOL), mean, 2.0, hrs[k])
                        break
                sl_conf.append((k, lvl))
                sl_conf[:] = sl_conf[-20:]
                last_sl, last_sl_k = lvl, k

        if last_sh == last_sh and last_sh_k != swept_sh and hi[i] > last_sh and ci < last_sh:
            swept_sh = last_sh_k
            add(i, "sweep", -1, hi[i], ci, (hi[i] + ci) / 2, (hi[i] - ci) / a, hrs[i])
        if last_sl == last_sl and last_sl_k != swept_sl and lo[i] < last_sl and ci > last_sl:
            swept_sl = last_sl_k
            add(i, "sweep", 1, ci, lo[i], (ci + lo[i]) / 2, (ci - lo[i]) / a, hrs[i])

        if last_only:
            do_eval = (i == n - 1)
        else:
            do_eval = (i >= start and (i - start) % stride == 0 and i <= n - 1 - fwd)
        if not do_eval or not zones:
            continue

        trend = 1.0 if P["ema_f"][i] > P["ema_s"][i] else -1.0
        h1s = float(_struct_state(sh_conf, sl_conf))
        rng = P["rmax"][i] - P["rmin"][i]
        pd_pos = (ci - P["rmin"][i]) / rng if rng > 0 else 0.5
        hs = np.sin(2 * np.pi * hrs[i] / 24)
        hc = np.cos(2 * np.pi * hrs[i] / 24)
        mids = {1: np.array([z["mid"] for z in zones if z["dir"] > 0]),
                -1: np.array([z["mid"] for z in zones if z["dir"] < 0])}
        mom24 = (ci - c[i - 24]) / a
        ap3 = (ci - c[i - 3]) / a
        ap12 = (ci - c[i - 12]) / a

        for z in zones:
            d = z["dir"]
            if d > 0:
                if ci <= z["top"]:
                    continue
                edge, dist = ci - z["top"], ci - z["mid"]
            else:
                if ci >= z["bottom"]:
                    continue
                edge, dist = z["bottom"] - ci, z["mid"] - ci
            dp = dist / ci
            age = i - z["created"]
            if age > MA:
                continue
            if dist_filter and not (dmin <= dp <= dmax):
                continue
            leg = bleg if d > 0 else sleg
            r, has_leg, in_ote, d705 = -1.0, 0.0, 0.0, 1.0
            if leg is not None:
                span = abs(leg[1] - leg[0])
                if span > 0:
                    r = ((leg[1] - z["mid"]) / span) if d > 0 else ((z["mid"] - leg[1]) / span)
                    has_leg = 1.0
                    in_ote = 1.0 if FLO <= r <= FHI else 0.0
                    d705 = abs(r - FID)
            m = mids[d]
            conf = int((np.abs(m - z["mid"]) <= 0.0015 * z["mid"]).sum()) - 1
            oh = [0.0] * len(KINDS)
            oh[_KIND_IDX[z["kind"]]] = 1.0
            rows.append(oh + [float(d), (z["top"] - z["bottom"]) / a, edge / a, dist / a, dp,
                              -d * ap3, -d * ap12, float(age), float(z["touches"]),
                              float(z["mid_hits"]), z["size_c"],
                              1.0 if _in_kz(z["hour_c"]) else 0.0, hs, hc, r, has_leg,
                              in_ote, d705, float(conf), pd_pos, a / ci, d * trend,
                              d * h1s, d * mom24])
            metas.append({"i": i, "kind": z["kind"], "dir": d, "top": z["top"],
                          "bottom": z["bottom"], "mid": z["mid"], "created": z["created"],
                          "hour_c": z["hour_c"], "age": age, "atr": a, "in_ote": bool(in_ote),
                          "r": r, "dp": dp, "touches": z["touches"]})

    ctx = {"P": P, "atr_last": float(atr[-1]), "sh_conf": list(sh_conf), "sl_conf": list(sl_conf),
           "bleg": None if bleg is None else list(bleg),
           "sleg": None if sleg is None else list(sleg), "n": n}
    return rows, metas, ctx

def _labels(metas, P):
    lo, hi, n = P["lo"], P["hi"], len(P["lo"])
    H, R = CFG["FILL_HORIZON"], CFG["REACT_WINDOW"]
    ra, ba = CFG["REACT_ATR"], CFG["BREAK_ATR"]
    m = len(metas)
    y_touch = np.zeros(m, dtype=int)
    y_react = np.zeros(m, dtype=int)
    pen = np.full(m, np.nan)
    BIG = 10 ** 9
    for q, mt in enumerate(metas):
        i, a, d, mid = mt["i"], mt["atr"], mt["dir"], mt["mid"]
        end = min(n - 1, i + H)
        if d > 0:
            hit = np.flatnonzero(lo[i + 1:end + 1] <= mid)
        else:
            hit = np.flatnonzero(hi[i + 1:end + 1] >= mid)
        if hit.size == 0:
            continue
        j = i + 1 + int(hit[0])
        y_touch[q] = 1
        e2 = min(n - 1, j + R)
        if d > 0:
            broke = np.flatnonzero(lo[j:e2 + 1] <= mid - ba * a)
            react = np.flatnonzero(hi[j + 1:e2 + 1] >= mid + ra * a)
        else:
            broke = np.flatnonzero(hi[j:e2 + 1] >= mid + ba * a)
            react = np.flatnonzero(lo[j + 1:e2 + 1] <= mid - ra * a)
        b = int(broke[0]) if broke.size else BIG
        r_ = int(react[0]) + 1 if react.size else BIG
        if r_ < b:
            y_react[q] = 1
            stop_at = min(e2, j + r_)
            if d > 0:
                pen[q] = max(0.0, (mt["bottom"] - lo[j:stop_at + 1].min()) / a)
            else:
                pen[q] = max(0.0, (hi[j:stop_at + 1].max() - mt["top"]) / a)
    return y_touch, y_react, pen

def _rf():
    return RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=40,
                                  max_features="sqrt", n_jobs=1, random_state=42)

def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")

def fit(index_name, hourly, min_auc=None, verbose=True):
    min_auc = CFG["MIN_AUC"] if min_auc is None else min_auc
    t0 = time.time()
    entry = {"usable": False}
    _MODELS[index_name] = entry
    h = hourly.reset_index(drop=True)
    if len(h) < 1500:
        if verbose:
            print(f"🧱 Zone model [{index_name}]: only {len(h)} hourly bars — rule-based fallback")
        return entry
    rows, metas, ctx = _walk(h, stride=CFG["TRAIN_STRIDE"], last_only=False, dist_filter=True)
    if len(rows) < 2000:
        if verbose:
            print(f"🧱 Zone model [{index_name}]: only {len(rows)} samples — rule-based fallback")
        return entry
    y_t, y_r, pen = _labels(metas, ctx["P"])
    X = np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    bars = np.array([mt["i"] for mt in metas])
    kinds = np.array([_KIND_IDX[mt["kind"]] for mt in metas])
    if len(X) > CFG["MAX_TRAIN_ROWS"]:
        sel = np.sort(np.random.default_rng(42).choice(len(X), CFG["MAX_TRAIN_ROWS"], replace=False))
        X, y_t, y_r, pen, bars, kinds = X[sel], y_t[sel], y_r[sel], pen[sel], bars[sel], kinds[sel]

    gap = CFG["FILL_HORIZON"] + CFG["REACT_WINDOW"] + 2
    cut = np.quantile(bars, 0.75)
    tr, te = bars < cut - gap, bars >= cut
    tr_t = tr & (y_t == 1)
    te_t = te & (y_t == 1)
    if tr_t.sum() < CFG["MIN_TOUCHED"] or te_t.sum() < 200:
        if verbose:
            print(f"🧱 Zone model [{index_name}]: too few filled samples "
                  f"({int(tr_t.sum())} train / {int(te_t.sum())} test) — rule-based fallback")
        return entry

    touch_tr = _rf().fit(X[tr], y_t[tr])
    react_tr = _rf().fit(X[tr_t], y_r[tr_t])
    auc_t = _auc(y_t[te], touch_tr.predict_proba(X[te])[:, 1])
    auc_r = _auc(y_r[te_t], react_tr.predict_proba(X[te_t])[:, 1])
    strong_thr = float(np.quantile(react_tr.predict_proba(X[te])[:, 1], CFG["STRONG_QUANTILE"]))

    allt = y_t == 1
    touch_m = _rf().fit(X, y_t)
    react_m = _rf().fit(X[allt], y_r[allt])
    react_base = float(y_r[allt].mean())

    pen_q = {}
    for kn, ki in _KIND_IDX.items():
        mk = allt & (y_r == 1) & (kinds == ki) & np.isfinite(pen)
        pen_q[kn] = float(np.clip(np.quantile(pen[mk], 0.75), 0.05, 1.0)) if mk.sum() >= 30 else 0.15

    usable = bool(np.isfinite(auc_r) and auc_r >= min_auc)
    entry.update({"usable": usable, "touch": touch_m, "react": react_m, "react_base": react_base,
                  "strong_thr": strong_thr, "pen_q": pen_q, "auc_touch": auc_t, "auc_react": auc_r,
                  "n": int(len(X)), "n_touched": int(allt.sum())})
    if verbose:
        imp = sorted(zip(FEATURES, react_m.feature_importances_), key=lambda x: -x[1])[:5]
        print(f"🧱 Zone reaction model [{index_name}]: {len(X):,} samples ({int(allt.sum()):,} filled) | "
              f"hold-out AUC touch {auc_t:.3f} / react {auc_r:.3f} | base react rate {react_base:.1%} | "
              f"{'ACTIVE' if usable else f'not predictive enough (<{min_auc}) — rule-based fallback'} "
              f"[{time.time() - t0:.0f}s]")
        print("   top reaction drivers: " + ", ".join(f"{k} {v:.2f}" for k, v in imp))
    return entry

def best_entry(hourly, side, daily_close, index_name="", lookback=200,
               fallback=None, daily_target=None):
    def _fb():
        if fallback is None:
            return None
        return fallback(hourly, side, daily_close, index_name=index_name, lookback=lookback)

    m = _MODELS.get(index_name)
    if not m or not m.get("usable") or len(hourly) < 400:
        return _fb()
    try:
        return _best_entry_learned(m, hourly, side, daily_close, daily_target)
    except Exception as e:
        print(f"   ⚠️  Zone-learning entry failed ({e}) — using rule-based entry")
        return _fb()

def _best_entry_learned(m, hourly, side, daily_close, daily_target):
    h = hourly.tail(1500).reset_index(drop=True)
    rows, metas, ctx = _walk(h, last_only=True, dist_filter=False)
    if not rows:
        return None
    X = np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    pt = m["touch"].predict_proba(X)[:, 1]
    pr = m["react"].predict_proba(X)[:, 1]
    for mt, a_, b_ in zip(metas, pt, pr):
        mt["p_touch"], mt["p_react"] = float(a_), float(b_)

    want = 1 if side == "LONG" else -1
    dmin, dmax = CFG["HOURLY_MIN_DIST_PCT"], CFG["HOURLY_MAX_DIST_PCT"]
    cands = []
    for mt in metas:
        if mt["dir"] != want:
            continue
        dist = (daily_close - mt["mid"]) / daily_close if want > 0 else (mt["mid"] - daily_close) / daily_close
        if dist < dmin or dist > dmax:
            continue
        mt["dist_close"] = dist
        mt["score"] = mt["p_touch"] * mt["p_react"] * (1 + CFG["OTE_PRIOR"] * (1.0 if mt["in_ote"] else 0.0))
        cands.append(mt)
    if not cands:
        return None
    best = max(cands, key=lambda z: (z["score"], -z["dist_close"]))

    atr = ctx["atr_last"]
    strong = m["strong_thr"]
    entry = best["mid"]
    buf_pct = CFG["HOURLY_STOP_BUFFER_PCT"]

    anchor_kind = best["kind"]
    if want > 0:
        anchor = best["bottom"]
        prot = [z for z in metas if z["dir"] == want and z is not best and z["bottom"] < best["bottom"]
                and (best["bottom"] - z["bottom"]) / atr <= 2.5 and z["p_react"] >= strong]
        if prot:
            pz = max(prot, key=lambda z: (z["p_react"], z["bottom"]))
            if pz["bottom"] < anchor:
                anchor, anchor_kind = pz["bottom"], pz["kind"]
        liq = [lvl for _, lvl in ctx["sl_conf"][-5:] if lvl < anchor and (anchor - lvl) / atr <= 0.75]
        if liq:
            anchor = min(liq)
        buf = max(buf_pct * anchor, m["pen_q"].get(anchor_kind, 0.15) * atr)
        stop_ref = anchor - buf
    else:
        anchor = best["top"]
        prot = [z for z in metas if z["dir"] == want and z is not best and z["top"] > best["top"]
                and (z["top"] - best["top"]) / atr <= 2.5 and z["p_react"] >= strong]
        if prot:
            pz = max(prot, key=lambda z: (z["p_react"], -z["top"]))
            if pz["top"] > anchor:
                anchor, anchor_kind = pz["top"], pz["kind"]
        liq = [lvl for _, lvl in ctx["sh_conf"][-5:] if lvl > anchor and (lvl - anchor) / atr <= 0.75]
        if liq:
            anchor = max(liq)
        buf = max(buf_pct * anchor, m["pen_q"].get(anchor_kind, 0.15) * atr)
        stop_ref = anchor + buf

    tp_ref = float(daily_target) if daily_target is not None and np.isfinite(daily_target) else None
    tp_note, blocked = None, False
    opp = [z for z in metas if z["dir"] == -want and z["p_react"] >= strong]
    if want > 0:
        opp = [z for z in opp if z["bottom"] > entry and (tp_ref is None or z["bottom"] < tp_ref)]
        blk = min(opp, key=lambda z: z["bottom"]) if opp else None
        if blk is not None:
            new_tp = blk["bottom"] - max(0.1 * atr, buf_pct * blk["bottom"])
            if new_tp <= entry + 0.25 * atr:
                blocked = True
            else:
                tp_ref = new_tp
                tp_note = (f"just inside {_LABELS[blk['kind']]} resistance {blk['bottom']:.2f} "
                           f"(learned react p={blk['p_react']:.0%})")
    else:
        opp = [z for z in opp if z["top"] < entry and (tp_ref is None or z["top"] > tp_ref)]
        blk = max(opp, key=lambda z: z["top"]) if opp else None
        if blk is not None:
            new_tp = blk["top"] + max(0.1 * atr, buf_pct * blk["top"])
            if new_tp >= entry - 0.25 * atr:
                blocked = True
            else:
                tp_ref = new_tp
                tp_note = (f"just inside {_LABELS[blk['kind']]} support {blk['top']:.2f} "
                           f"(learned react p={blk['p_react']:.0%})")

    FLO, FHI, FID = CFG["OTE_FIB_LOW"], CFG["OTE_FIB_HIGH"], CFG["OTE_FIB_IDEAL"]
    ote_lo = ote_hi = ote_705 = None
    if want > 0 and ctx["bleg"] is not None:
        lo_, hi_ = ctx["bleg"]
        sp = hi_ - lo_
        if sp > 0:
            ote_lo, ote_hi, ote_705 = hi_ - FHI * sp, hi_ - FLO * sp, hi_ - FID * sp
    elif want < 0 and ctx["sleg"] is not None:
        hi_, lo_ = ctx["sleg"]
        sp = hi_ - lo_
        if sp > 0:
            ote_lo, ote_hi, ote_705 = lo_ + FLO * sp, lo_ + FHI * sp, lo_ + FID * sp

    session = _session(best["hour_c"])
    kz = _in_kz(best["hour_c"])
    st = _struct_state(ctx["sh_conf"], ctx["sl_conf"])
    want_dir = "bull" if want > 0 else "bear"
    label = _LABELS[best["kind"]]
    ote_note = ""
    if best["in_ote"] and ote_705 is not None:
        ote_note = f", OTE-aligned (0.705 = {ote_705:.2f})"
    elif ote_705 is not None:
        ote_note = ", outside OTE band"
    kz_note = f", kill zone: {kz}" if kz else ""
    st_note = {1: ", H1 structure bullish", -1: ", H1 structure bearish", 0: ""}[st]
    dist_pct = best["dist_close"] * 100

    return {
        "kind": best["kind"],
        "kind_label": f"{want_dir.upper()} {label}",
        "entry_price": round(entry, 4),
        "zone_top": round(best["top"], 4),
        "zone_bottom": round(best["bottom"], 4),
        "hour_utc": best["hour_c"],
        "session": session,
        "bars_since_zone": int(best["age"]),
        "distance_pct": round(dist_pct, 3),
        "stop_reference": round(stop_ref, 4),
        "priority": round(best["p_react"], 2),
        "score": round(best["score"], 3),
        "ote_aligned": bool(best["in_ote"]),
        "ote_705": round(ote_705, 4) if ote_705 is not None else None,
        "ote_band_low": round(ote_lo, 4) if ote_lo is not None else None,
        "ote_band_high": round(ote_hi, 4) if ote_hi is not None else None,
        "kill_zone": kz,
        "session_weight": 1.0,
        "reason": (f"{label.upper()} zone on hourly chart, {best['hour_c']:02d}:00 UTC ({session}), "
                   f"{best['age']} bars ago, {dist_pct:.2f}% from current price "
                   f"(confluence score {best['score']:.2f}, learned react p={best['p_react']:.0%}, "
                   f"fill p={best['p_touch']:.0%}{ote_note}{kz_note}{st_note})."),
        "learned": True,
        "react_prob": best["p_react"],
        "touch_prob": best["p_touch"],
        "react_base": m["react_base"],
        "tp_reference": tp_ref,
        "tp_note": tp_note,
        "blocked": blocked,
        "stop_note": f"behind {_LABELS[anchor_kind]} zone + learned {m['pen_q'].get(anchor_kind, 0.15):.2f} ATR wick buffer",
    }
'''

_zl = types.ModuleType("zone_learning")
exec(_ZL_SOURCE, _zl.__dict__)
sys.modules["zone_learning"] = _zl
import zone_learning as zl

# ============================================
# 0. SETUP — MULTI-INDEX CONFIG
# ============================================

INDEX_CONFIGS = [
    ("SP500_1d_data.csv",  "S&P 500"),
    ("NASDAQ_1d_data.csv", "NASDAQ"),
    ("DOW_1d_data.csv",    "DOW JONES"),
    ("XAUUSD_1d_data.csv", "GOLD (XAUUSD)"),
]

DXY_CSV_PATH = "DXY_1d_data.csv"

# Hourly CSVs — used by the hourly entry optimiser (model 2)
HOURLY_CONFIGS = {
    "S&P 500":       "SP500_index_H1.csv",
    "NASDAQ":        "NASDAQ100_index_H1.csv",
    "DOW JONES":     "DOW_index_H1.csv",
    "GOLD (XAUUSD)": "XAUUSD_H1.csv",
}

# Hourly entry optimiser (model 2) — structural zone detection
HOURLY_OPT_ENABLED         = True
HOURLY_STRUCT_LOOKBACK     = 200
HOURLY_ZONE_MAX_AGE        = 48
HOURLY_MIN_DIST_PCT        = 0.0003
HOURLY_MAX_DIST_PCT        = 0.02
HOURLY_STOP_BUFFER_PCT     = 0.0005

# ICT OTE settings (hourly execution layer)
OTE_FIB_LOW        = 0.62
OTE_FIB_HIGH       = 0.79
OTE_FIB_IDEAL      = 0.705
OTE_BOOST          = 1.5
OTE_PENALTY        = 0.6
OTE_DISP_MULT      = 1.5
OTE_LEG_LOOKBACK   = 20

# Kill-zone windows (UTC) — soft bonus only
KILL_ZONES_UTC = {
    "London KZ":     (7, 10),
    "NY AM KZ":      (12, 15),
    "Silver Bullet": (14, 15),
}
KILL_ZONE_BOOST    = 0.30
SESSION_WEIGHTS = {
    "S&P 500":       {"London": 0.8, "New York": 1.2, "Asia": 0.4, "Asia (late)": 0.5},
    "NASDAQ":        {"London": 0.9, "New York": 1.3, "Asia": 0.4, "Asia (late)": 0.5},
    "DOW JONES":     {"London": 0.8, "New York": 1.2, "Asia": 0.4, "Asia (late)": 0.5},
    "GOLD (XAUUSD)": {"London": 1.3, "New York": 1.1, "Asia": 0.9, "Asia (late)": 0.7},
}

# Bias gate thresholds (reference only; gate no longer vetoes or flips)
BIAS_MIN_CONF     = 0.42
BIAS_DISCOUNT_MAX = 0.55
BIAS_PREMIUM_MIN  = 0.45

zl.configure(
    OTE_FIB_LOW=OTE_FIB_LOW, OTE_FIB_HIGH=OTE_FIB_HIGH, OTE_FIB_IDEAL=OTE_FIB_IDEAL,
    OTE_DISP_MULT=OTE_DISP_MULT, OTE_LEG_LOOKBACK=OTE_LEG_LOOKBACK,
    KILL_ZONES_UTC=KILL_ZONES_UTC,
    HOURLY_MIN_DIST_PCT=HOURLY_MIN_DIST_PCT, HOURLY_MAX_DIST_PCT=HOURLY_MAX_DIST_PCT,
    HOURLY_STOP_BUFFER_PCT=HOURLY_STOP_BUFFER_PCT,
)

# ============================================
# TRADE JOURNAL ("MEMORY") SETTINGS
# ============================================
USE_JOURNAL = True
JOURNAL_DIR = "journal"
JOURNAL_HOLD_BARS = 1

CAL_MIN_TRADES = 60
CAL_TRAIN_FRACTION = 0.7
CAL_MIN_IMPROVEMENT = 0.02
CAL_COMPLEXITY_PENALTY = 0.01
CAL_SHIFT_GRID = [0.0, 0.10, 0.20, 0.35, 0.50]
CAL_EXTRA_STOP_GRID = [0.0, 0.25, 0.50, 1.00, 1.50]
CAL_TARGET_SCALE_GRID = [0.5, 0.75, 1.0, 1.5]

META_MIN_TRADES = 150
META_MIN_AUC = 0.55
META_MIN_EXPECTED_R = 0.0

os.makedirs(JOURNAL_DIR, exist_ok=True)


def _normalize_path(raw: str) -> str:
    cleaned = "".join(" " if c.isspace() else c for c in raw)
    return " ".join(cleaned.split())


def _safe_savefig(fig, directory: str, filename: str, dpi: int = 120):
    directory = _normalize_path(directory)
    os.makedirs(directory, exist_ok=True)
    attempts = [
        os.path.join(directory, filename),
        os.path.join(str(Path.home()), filename),
    ]
    last_err = None
    for path in attempts:
        for delay in (0, 0.5, 1.5):
            if delay:
                time.sleep(delay)
            try:
                fig.savefig(path, dpi=dpi)
                return path
            except OSError as e:
                last_err = e
                if getattr(e, 'errno', None) == 22:
                    continue
                break
    raise last_err if last_err else RuntimeError("savefig failed with no error")


def _parse_dates(series):
    formats = ["%b %d, %Y", "%d %b %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(series, errors='coerce')


# ============================================
# DXY LOADER
# ============================================

def load_dxy_series(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"ℹ️  DXY file not found at {csv_path} — DXY features will be disabled")
        return None
    try:
        dxy = pd.read_csv(csv_path)
        dxy['Date'] = _parse_dates(dxy['Date'])
        dxy = dxy.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        close_col = None
        for candidate in ['Close', 'DXY_Close', 'Price', 'close']:
            if candidate in dxy.columns:
                close_col = candidate
                break
        if close_col is None:
            print(f"⚠️  DXY CSV missing Close column — DXY features disabled")
            return None
        dxy = dxy[['Date', close_col]].rename(columns={close_col: 'DXY_Close'})
        dxy['DXY_Close'] = pd.to_numeric(
            dxy['DXY_Close'].astype(str).str.replace(',', ''), errors='coerce')
        dxy = dxy.dropna(subset=['DXY_Close']).reset_index(drop=True)
        dxy['DXY_Return'] = dxy['DXY_Close'].pct_change()
        dxy['DXY_MA20'] = dxy['DXY_Close'].rolling(20).mean()
        dxy['DXY_vs_MA20'] = dxy['DXY_Close'] / dxy['DXY_MA20'] - 1
        _delta = dxy['DXY_Close'].diff()
        _gain = _delta.where(_delta > 0, 0).rolling(14).mean()
        _loss = -_delta.where(_delta < 0, 0).rolling(14).mean()
        _rs = _gain / _loss
        dxy['DXY_RSI'] = 100 - (100 / (1 + _rs))
        dxy['DXY_Return_Lag1'] = dxy['DXY_Return'].shift(1)
        dxy['DXY_Return_Lag5'] = dxy['DXY_Close'].pct_change(5)
        print(f"\n💵 DXY loaded: {len(dxy)} rows "
              f"({dxy['Date'].min().date()} → {dxy['Date'].max().date()})")
        return dxy
    except Exception as e:
        print(f"⚠️  Failed to load DXY ({e}) — DXY features will be disabled")
        return None


# ============================================
# HOURLY DATA — LOADING
# ============================================

def load_hourly_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    try:
        h = pd.read_csv(csv_path)
        needed = {"timestamp", "open", "high", "low", "close"}
        if not needed.issubset(h.columns):
            return None
        h = h.drop_duplicates(subset=["timestamp", "open", "high", "low", "close"],
                              keep="first").reset_index(drop=True)
        h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True, errors="coerce")
        h = h.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        for c in ("open", "high", "low", "close"):
            h[c] = pd.to_numeric(h[c], errors="coerce")
        h["volume"] = pd.to_numeric(h.get("volume"), errors="coerce") if "volume" in h else 0.0
        h = h.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        h["hour_utc"] = h["timestamp"].dt.hour
        h["date_utc"] = h["timestamp"].dt.date
        return h
    except Exception:
        return None


def _session_from_hour(hour: int) -> str:
    if 0 <= hour < 7:
        return "Asia"
    elif 7 <= hour < 12:
        return "London"
    elif 12 <= hour < 21:
        return "New York"
    return "Asia (late)"


def _in_kill_zone(hour_utc: int):
    for name, (start, end) in KILL_ZONES_UTC.items():
        if start <= hour_utc < end:
            return name
    return None


# ============================================
# MODEL 2 — HOURLY STRUCTURAL ENTRY OPTIMISER (rule-based original)
# ============================================

def _find_hourly_swings(hourly: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    h = hourly.copy().reset_index(drop=True)
    n = len(h)
    high = h["high"].values
    low = h["low"].values
    is_sh = np.zeros(n, dtype=bool)
    is_sl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        if high[i] == high[i - left:i + right + 1].max():
            is_sh[i] = True
        if low[i] == low[i - left:i + right + 1].min():
            is_sl[i] = True
    h["is_swing_high"] = is_sh
    h["is_swing_low"] = is_sl
    return h


def _find_hourly_fvgs(hourly: pd.DataFrame, lookback: int = 200) -> list:
    h = hourly.tail(lookback).reset_index(drop=True)
    fvgs = []
    for i in range(2, len(h)):
        hi2 = h.loc[i - 2, "high"]
        lo2 = h.loc[i - 2, "low"]
        hi0 = h.loc[i, "high"]
        lo0 = h.loc[i, "low"]
        if lo0 > hi2:
            top, bottom = lo0, hi2
            filled = any(h.loc[j, "low"] <= bottom for j in range(i + 1, len(h)))
            if not filled:
                fvgs.append({
                    "direction": "bull",
                    "top": float(top), "bottom": float(bottom),
                    "mid": float((top + bottom) / 2),
                    "date": h.loc[i, "timestamp"],
                    "hour_utc": int(h.loc[i, "hour_utc"]),
                    "bars_since": len(h) - 1 - i,
                })
        if hi0 < lo2:
            top, bottom = lo2, hi0
            filled = any(h.loc[j, "high"] >= top for j in range(i + 1, len(h)))
            if not filled:
                fvgs.append({
                    "direction": "bear",
                    "top": float(top), "bottom": float(bottom),
                    "mid": float((top + bottom) / 2),
                    "date": h.loc[i, "timestamp"],
                    "hour_utc": int(h.loc[i, "hour_utc"]),
                    "bars_since": len(h) - 1 - i,
                })
    return fvgs


def _find_hourly_order_blocks(hourly: pd.DataFrame, lookback: int = 200) -> list:
    h = hourly.tail(lookback).reset_index(drop=True)
    body = h["close"] - h["open"]
    avg_body = body.abs().rolling(20).mean()
    obs = []
    for i in range(1, len(h)):
        if pd.isna(avg_body.iloc[i - 1]) or avg_body.iloc[i - 1] == 0:
            continue
        if abs(body.iloc[i]) > 1.5 * avg_body.iloc[i - 1]:
            ob_idx = i - 1
            if body.iloc[i] > 0 and body.iloc[ob_idx] < 0:
                obs.append({
                    "direction": "bull",
                    "top": float(h.loc[ob_idx, "high"]),
                    "bottom": float(h.loc[ob_idx, "low"]),
                    "body_top": float(max(h.loc[ob_idx, "open"], h.loc[ob_idx, "close"])),
                    "body_bottom": float(min(h.loc[ob_idx, "open"], h.loc[ob_idx, "close"])),
                    "date": h.loc[ob_idx, "timestamp"],
                    "hour_utc": int(h.loc[ob_idx, "hour_utc"]),
                    "bars_since": len(h) - 1 - ob_idx,
                })
            elif body.iloc[i] < 0 and body.iloc[ob_idx] > 0:
                obs.append({
                    "direction": "bear",
                    "top": float(h.loc[ob_idx, "high"]),
                    "bottom": float(h.loc[ob_idx, "low"]),
                    "body_top": float(max(h.loc[ob_idx, "open"], h.loc[ob_idx, "close"])),
                    "body_bottom": float(min(h.loc[ob_idx, "open"], h.loc[ob_idx, "close"])),
                    "date": h.loc[ob_idx, "timestamp"],
                    "hour_utc": int(h.loc[ob_idx, "hour_utc"]),
                    "bars_since": len(h) - 1 - ob_idx,
                })
    return obs


def _find_hourly_ifvgs(hourly: pd.DataFrame, lookback: int = 200) -> list:
    h = hourly.tail(lookback).reset_index(drop=True)
    out = []
    open_fvgs = []
    for i in range(2, len(h)):
        hi2 = h.loc[i - 2, "high"]; lo2 = h.loc[i - 2, "low"]
        hi0 = h.loc[i, "high"];   lo0 = h.loc[i, "low"]
        cl  = h.loc[i, "close"]
        if lo0 > hi2:
            open_fvgs.append({"dir": "bull", "top": lo0, "bottom": hi2,
                              "idx": i, "hour_utc": int(h.loc[i, "hour_utc"])})
        if hi0 < lo2:
            open_fvgs.append({"dir": "bear", "top": lo2, "bottom": hi0,
                              "idx": i, "hour_utc": int(h.loc[i, "hour_utc"])})
        for fvg in open_fvgs:
            if fvg["dir"] == "bull" and cl < fvg["bottom"]:
                out.append({
                    "direction": "bear",
                    "top": float(fvg["bottom"]),
                    "bottom": float(fvg["bottom"] - (fvg["top"] - fvg["bottom"]) * 0.5),
                    "mid": float(fvg["bottom"] - (fvg["top"] - fvg["bottom"]) * 0.25),
                    "date": h.loc[i, "timestamp"],
                    "hour_utc": int(h.loc[i, "hour_utc"]),
                    "bars_since": len(h) - 1 - i,
                })
                fvg["dir"] = "inverted"
            elif fvg["dir"] == "bear" and cl > fvg["top"]:
                out.append({
                    "direction": "bull",
                    "top": float(fvg["top"] + (fvg["top"] - fvg["bottom"]) * 0.5),
                    "bottom": float(fvg["top"]),
                    "mid": float(fvg["top"] + (fvg["top"] - fvg["bottom"]) * 0.25),
                    "date": h.loc[i, "timestamp"],
                    "hour_utc": int(h.loc[i, "hour_utc"]),
                    "bars_since": len(h) - 1 - i,
                })
                fvg["dir"] = "inverted"
    return out


def _find_hourly_breakers(hourly: pd.DataFrame, lookback: int = 200) -> list:
    h = hourly.tail(lookback).reset_index(drop=True)
    body = h["close"] - h["open"]
    avg_body = body.abs().rolling(20).mean()
    out = []
    for i in range(1, len(h) - 1):
        if pd.isna(avg_body.iloc[i - 1]) or avg_body.iloc[i - 1] == 0:
            continue
        if abs(body.iloc[i]) > 1.5 * avg_body.iloc[i - 1]:
            ob_idx = i - 1
            ob_high = float(h.loc[ob_idx, "high"])
            ob_low  = float(h.loc[ob_idx, "low"])
            for j in range(i + 1, min(i + 20, len(h))):
                close_j = h.loc[j, "close"]
                if body.iloc[ob_idx] < 0 and close_j > ob_high:
                    out.append({
                        "direction": "bear",
                        "top": ob_high,
                        "bottom": float(min(h.loc[ob_idx, "open"], h.loc[ob_idx, "close"])),
                        "mid": float((ob_high + min(h.loc[ob_idx, "open"], h.loc[ob_idx, "close"])) / 2),
                        "date": h.loc[j, "timestamp"],
                        "hour_utc": int(h.loc[j, "hour_utc"]),
                        "bars_since": len(h) - 1 - j,
                    })
                    break
                if body.iloc[ob_idx] > 0 and close_j < ob_low:
                    out.append({
                        "direction": "bull",
                        "top": float(max(h.loc[ob_idx, "open"], h.loc[ob_idx, "close"])),
                        "bottom": ob_low,
                        "mid": float((ob_low + max(h.loc[ob_idx, "open"], h.loc[ob_idx, "close"])) / 2),
                        "date": h.loc[j, "timestamp"],
                        "hour_utc": int(h.loc[j, "hour_utc"]),
                        "bars_since": len(h) - 1 - j,
                    })
                    break
    return out


def _find_hourly_liquidity_sweeps(hourly: pd.DataFrame, lookback: int = 200,
                                   swings: pd.DataFrame = None) -> list:
    h = (swings if swings is not None else _find_hourly_swings(hourly)).tail(lookback).reset_index(drop=True)
    out = []
    last_sh = np.nan
    last_sl = np.nan
    for i in range(len(h)):
        if h.loc[i, "is_swing_high"]:
            last_sh = float(h.loc[i, "high"])
        if h.loc[i, "is_swing_low"]:
            last_sl = float(h.loc[i, "low"])
        if not np.isnan(last_sh) and h.loc[i, "high"] > last_sh and h.loc[i, "close"] < last_sh:
            out.append({
                "direction": "bear",
                "top": float(h.loc[i, "high"]),
                "bottom": float(h.loc[i, "close"]),
                "mid": float(h.loc[i, "high"]),
                "date": h.loc[i, "timestamp"],
                "hour_utc": int(h.loc[i, "hour_utc"]),
                "bars_since": len(h) - 1 - i,
            })
        if not np.isnan(last_sl) and h.loc[i, "low"] < last_sl and h.loc[i, "close"] > last_sl:
            out.append({
                "direction": "bull",
                "top": float(h.loc[i, "close"]),
                "bottom": float(h.loc[i, "low"]),
                "mid": float(h.loc[i, "low"]),
                "date": h.loc[i, "timestamp"],
                "hour_utc": int(h.loc[i, "hour_utc"]),
                "bars_since": len(h) - 1 - i,
            })
    return out


def _find_hourly_equal_levels(hourly: pd.DataFrame, lookback: int = 200,
                               tol_pct: float = 0.0015) -> list:
    h = _find_hourly_swings(hourly).tail(lookback).reset_index(drop=True)
    out = []
    swing_highs = h[h["is_swing_high"]].reset_index(drop=True)
    swing_lows  = h[h["is_swing_low"]].reset_index(drop=True)
    for swings, direction in ((swing_highs, "bear"), (swing_lows, "bull")):
        if len(swings) < 2:
            continue
        used = set()
        for i in range(len(swings)):
            if i in used:
                continue
            lvl_i = float(swings.loc[i, "high" if direction == "bear" else "low"])
            cluster = [i]
            for j in range(i + 1, len(swings)):
                if j in used:
                    continue
                lvl_j = float(swings.loc[j, "high" if direction == "bear" else "low"])
                if abs(lvl_j - lvl_i) / lvl_i <= tol_pct:
                    cluster.append(j)
            if len(cluster) >= 2:
                for k in cluster:
                    used.add(k)
                idx_last = max(cluster)
                lvl_mean = float(np.mean([
                    float(swings.loc[k, "high" if direction == "bear" else "low"])
                    for k in cluster
                ]))
                out.append({
                    "direction": direction,
                    "top": lvl_mean * (1 + tol_pct) if direction == "bear" else lvl_mean,
                    "bottom": lvl_mean if direction == "bear" else lvl_mean * (1 - tol_pct),
                    "mid": lvl_mean,
                    "date": swings.loc[idx_last, "timestamp"],
                    "hour_utc": int(swings.loc[idx_last, "hour_utc"]),
                    "bars_since": int(len(h) - 1 - idx_last),
                })
    return out


def _find_hourly_imbalances(hourly: pd.DataFrame, lookback: int = 200) -> list:
    h = hourly.tail(lookback).reset_index(drop=True)
    body = h["close"] - h["open"]
    avg_body = body.abs().rolling(20).mean()
    out = []
    for i in range(1, len(h)):
        if pd.isna(avg_body.iloc[i - 1]) or avg_body.iloc[i - 1] == 0:
            continue
        if abs(body.iloc[i]) < 1.5 * avg_body.iloc[i - 1]:
            continue
        top = float(max(h.loc[i, "open"], h.loc[i, "close"]))
        bottom = float(min(h.loc[i, "open"], h.loc[i, "close"]))
        mid = (top + bottom) / 2
        if body.iloc[i] > 0:
            filled = any(h.loc[j, "low"] <= mid for j in range(i + 1, len(h)))
            direction = "bull"
        else:
            filled = any(h.loc[j, "high"] >= mid for j in range(i + 1, len(h)))
            direction = "bear"
        if not filled:
            out.append({
                "direction": direction,
                "top": top,
                "bottom": bottom,
                "mid": mid,
                "date": h.loc[i, "timestamp"],
                "hour_utc": int(h.loc[i, "hour_utc"]),
                "bars_since": len(h) - 1 - i,
            })
    return out


def _find_displacement_leg(hourly: pd.DataFrame, side: str,
                           lookback: int = HOURLY_STRUCT_LOOKBACK):
    h = hourly.tail(lookback).reset_index(drop=True)
    if len(h) < 30:
        return None

    body = (h["close"] - h["open"]).values
    avg_body = (h["close"] - h["open"]).abs().rolling(20).mean().shift(1).values

    if side == "LONG":
        disp_mask = (body > OTE_DISP_MULT * avg_body) & (avg_body > 0)
    else:
        disp_mask = (-body > OTE_DISP_MULT * avg_body) & (avg_body > 0)

    idxs = np.where(np.nan_to_num(disp_mask, nan=0).astype(bool))[0]
    if len(idxs) == 0:
        return None
    i_disp = int(idxs[-1])

    start = max(0, i_disp - OTE_LEG_LOOKBACK)
    if side == "LONG":
        i_origin = start + int(np.argmin(h["low"].values[start:i_disp + 1]))
        leg_low  = float(h.loc[i_origin, "low"])
        extension_end = min(i_disp + 5, len(h) - 1)
        leg_high = float(h.loc[i_disp:extension_end + 1, "high"].max())
        if leg_high <= leg_low:
            return None
        fib_top, fib_bottom = leg_high, leg_low
    else:
        i_origin = start + int(np.argmax(h["high"].values[start:i_disp + 1]))
        leg_high = float(h.loc[i_origin, "high"])
        extension_end = min(i_disp + 5, len(h) - 1)
        leg_low  = float(h.loc[i_disp:extension_end + 1, "low"].min())
        if leg_high <= leg_low:
            return None
        fib_top, fib_bottom = leg_high, leg_low

    span = fib_top - fib_bottom
    if side == "LONG":
        ote_low  = fib_bottom + span * OTE_FIB_HIGH
        ote_high = fib_bottom + span * OTE_FIB_LOW
    else:
        ote_low  = fib_top - span * OTE_FIB_HIGH
        ote_high = fib_top - span * OTE_FIB_LOW
    ote_705 = fib_bottom + span * (1 - OTE_FIB_IDEAL) if side == "LONG" \
              else fib_top - span * OTE_FIB_IDEAL

    return {
        "origin_idx": i_origin,
        "disp_idx": i_disp,
        "fib_top": fib_top,
        "fib_bottom": fib_bottom,
        "ote_low": ote_low,
        "ote_high": ote_high,
        "ote_705": ote_705,
        "displacement_size": float(abs(body[i_disp])),
        "bars_ago": len(h) - 1 - i_disp,
    }


def _h1_trend_state(hourly: pd.DataFrame, lookback: int = HOURLY_STRUCT_LOOKBACK) -> int:
    h_swings = _find_hourly_swings(hourly.tail(lookback).reset_index(drop=True))
    last_20 = h_swings.tail(20)
    sh = last_20.loc[last_20["is_swing_high"], "high"].values
    sl = last_20.loc[last_20["is_swing_low"], "low"].values
    if len(sh) < 2 or len(sl) < 2:
        return 0
    if sh[-1] > sh[-2] and sl[-1] > sl[-2]:
        return 1
    if sh[-1] < sh[-2] and sl[-1] < sl[-2]:
        return -1
    return 0


def best_hourly_entry(hourly: pd.DataFrame,
                      side: str,
                      daily_close: float,
                      index_name: str = "",
                      lookback: int = HOURLY_STRUCT_LOOKBACK) -> dict:
    if len(hourly) < 50:
        return None

    h1_trend = _h1_trend_state(hourly, lookback)
    if side == "LONG" and h1_trend == -1:
        return None
    if side == "SHORT" and h1_trend == 1:
        return None

    h_swings = _find_hourly_swings(hourly.tail(lookback).reset_index(drop=True))

    families = [
        ("fvg",        _find_hourly_fvgs(hourly, lookback),       1.00),
        ("ob",         _find_hourly_order_blocks(hourly, lookback), 0.85),
        ("ifvg",       _find_hourly_ifvgs(hourly, lookback),       0.75),
        ("breaker",    _find_hourly_breakers(hourly, lookback),   0.70),
        ("sweep",      _find_hourly_liquidity_sweeps(hourly, lookback, h_swings), 0.80),
        ("eqhl",       _find_hourly_equal_levels(hourly, lookback), 0.55),
        ("imbalance",  _find_hourly_imbalances(hourly, lookback), 0.60),
    ]

    want_dir = "bull" if side == "LONG" else "bear"
    candidates = []
    for kind, zones, weight in families:
        for z in zones:
            if z["direction"] != want_dir:
                continue
            if z["bars_since"] > HOURLY_ZONE_MAX_AGE:
                continue
            entry_price = z.get("mid", (z["top"] + z["bottom"]) / 2)
            if side == "LONG":
                dist = (daily_close - entry_price) / daily_close
            else:
                dist = (entry_price - daily_close) / daily_close
            if dist < HOURLY_MIN_DIST_PCT or dist > HOURLY_MAX_DIST_PCT:
                continue
            candidates.append({
                **z,
                "kind": kind,
                "entry_price": entry_price,
                "distance_pct": dist,
                "priority": weight,
            })

    if not candidates:
        return None

    leg = _find_displacement_leg(hourly, side, lookback)

    for c in candidates:
        c_mid = c["entry_price"]
        boost = c["priority"]
        for other in candidates:
            if other is c:
                continue
            if abs(other["entry_price"] - c_mid) / c_mid <= 0.0015:
                boost += other["priority"] * 0.5
        c["score"] = boost

        if leg is not None:
            in_ote = leg["ote_low"] <= c["entry_price"] <= leg["ote_high"]
            if in_ote:
                d_705 = abs(c["entry_price"] - leg["ote_705"]) / max(leg["ote_705"], 1e-9)
                c["score"] += OTE_BOOST * max(0.0, 1.0 - d_705 / 0.02)
                c["ote_aligned"] = True
                c["ote_705"] = leg["ote_705"]
                c["ote_band"] = (leg["ote_low"], leg["ote_high"])
            else:
                c["score"] *= OTE_PENALTY
                c["ote_aligned"] = False
                c["ote_705"] = leg["ote_705"]
                c["ote_band"] = (leg["ote_low"], leg["ote_high"])
        else:
            c["ote_aligned"] = False
            c["ote_705"] = np.nan
            c["ote_band"] = (np.nan, np.nan)

        session = _session_from_hour(c["hour_utc"])
        sw = SESSION_WEIGHTS.get(index_name, {}).get(session, 1.0)
        c["score"] *= sw
        kz = _in_kill_zone(c["hour_utc"])
        if kz is not None:
            c["score"] += KILL_ZONE_BOOST
        c["kill_zone"] = kz
        c["session_weight"] = sw

    candidates.sort(key=lambda c: (-c["score"], c["distance_pct"]))
    best = candidates[0]

    if side == "LONG":
        stop_candidates = []
        recent_lows = h_swings[h_swings["is_swing_low"]].tail(3)
        if len(recent_lows):
            stop_candidates.append(float(recent_lows["low"].min()))
        stop_candidates.append(best["bottom"])
        if leg is not None:
            stop_candidates.append(leg["fib_bottom"])
        stop_ref = min(stop_candidates) * (1 - HOURLY_STOP_BUFFER_PCT)
    else:
        stop_candidates = []
        recent_highs = h_swings[h_swings["is_swing_high"]].tail(3)
        if len(recent_highs):
            stop_candidates.append(float(recent_highs["high"].max()))
        stop_candidates.append(best["top"])
        if leg is not None:
            stop_candidates.append(leg["fib_top"])
        stop_ref = max(stop_candidates) * (1 + HOURLY_STOP_BUFFER_PCT)

    session = _session_from_hour(best["hour_utc"])
    label_map = {
        "fvg": "FVG", "ob": "OB", "ifvg": "IFVG", "breaker": "Breaker",
        "sweep": "Liquidity Sweep", "eqhl": "Equal High/Low", "imbalance": "Imbalance",
    }
    kind_label = f"{want_dir.upper()} {label_map.get(best['kind'], best['kind'])}"

    ote_note = ""
    if best.get("ote_aligned"):
        ote_note = (f", OTE-aligned (0.705 = {best['ote_705']:.2f})")
    elif leg is not None:
        ote_note = ", outside OTE band (penalised)"

    kz_note = f", kill zone: {best['kill_zone']}" if best.get("kill_zone") else ""
    struct_note = {1: ", H1 structure bullish", -1: ", H1 structure bearish", 0: ""}.get(h1_trend, "")

    return {
        "kind": best["kind"],
        "kind_label": kind_label,
        "entry_price": round(best["entry_price"], 4),
        "zone_top": round(best["top"], 4),
        "zone_bottom": round(best["bottom"], 4),
        "hour_utc": best["hour_utc"],
        "session": session,
        "bars_since_zone": best["bars_since"],
        "distance_pct": round(best["distance_pct"] * 100, 3),
        "stop_reference": round(stop_ref, 4),
        "priority": round(best["priority"], 2),
        "score": round(best["score"], 2),
        "ote_aligned": bool(best.get("ote_aligned", False)),
        "ote_705": round(best["ote_705"], 4) if pd.notna(best.get("ote_705", np.nan)) else None,
        "ote_band_low":  round(best["ote_band"][0], 4) if pd.notna(best["ote_band"][0]) else None,
        "ote_band_high": round(best["ote_band"][1], 4) if pd.notna(best["ote_band"][1]) else None,
        "kill_zone": best.get("kill_zone"),
        "session_weight": best.get("session_weight", 1.0),
        "reason": (f"{label_map.get(best['kind'], best['kind']).upper()} zone on hourly chart, "
                   f"{best['hour_utc']:02d}:00 UTC ({session}), "
                   f"{best['bars_since']} bars ago, "
                   f"{best['distance_pct']*100:.2f}% from current price "
                   f"(confluence score {best['score']:.2f}{ote_note}{kz_note}{struct_note})."),
    }


_best_hourly_entry_rules = best_hourly_entry


def best_hourly_entry(hourly, side, daily_close, index_name="",
                      lookback=HOURLY_STRUCT_LOOKBACK, daily_target=None):
    return zl.best_entry(hourly, side, daily_close, index_name=index_name, lookback=lookback,
                         fallback=_best_hourly_entry_rules, daily_target=daily_target)


# ============================================
# TRADE JOURNAL — STORAGE
# ============================================

JOURNAL_COLUMNS = [
    'signal_date', 'target_date', 'source',
    'predicted_state', 'prob_up', 'prob_down', 'prob_cons', 'confidence', 'move_threshold',
    'action', 'entry_type', 'entry_style', 'entry', 'stop_loss', 'take_profit',
    'stop_source', 'target_source', 'close_t', 'atr', 'is_market', 'd_s', 'd_t', 'risk_reward',
    'zone_pos', 'vol20', 'rsi', 'market_structure', 'analog_updown',
    'zone_strength', 'dow', 'entry_depth_atr',
    'adapt_applied', 'adapt_shift', 'adapt_extra', 'adapt_tscale',
    'adapt_entry', 'adapt_stop', 'adapt_target', 'journal_win_prob', 'meta_filtered',
    'hourly_hour_utc', 'hourly_session', 'hourly_zone_kind',
    'hourly_entry_price', 'hourly_distance_pct', 'hourly_bars_since_zone',
    'hourly_stop_reference', 'hourly_applied', 'hourly_risk_reward',
    'daily_struct_entry', 'daily_struct_rr',
    'ote_aligned', 'ote_705', 'ote_band_low', 'ote_band_high',
    'dealing_pos_at_signal', 'dist_to_pdh', 'dist_to_pdl',
    'bias_gate_passed', 'kill_zone', 'session_weight',
    'dol_target', 'dol_source', 'dol_rank',
    'resolved', 'win_open', 'win_high', 'win_low', 'win_close', 'next_close',
    'actual_return', 'actual_state', 'dir_correct', 'right_direction',
    'filled', 'outcome', 'pnl_r', 'mae_atr', 'mfe_atr', 'stopped_but_right',
    'adapt_pnl_r', 'adapt_outcome',
]
_DATE_COLS = ['signal_date', 'target_date']
_BOOL_COLS = ['is_market', 'adapt_applied', 'meta_filtered', 'resolved', 'filled',
              'dir_correct', 'right_direction', 'stopped_but_right', 'hourly_applied',
              'ote_aligned', 'bias_gate_passed']
_TEXT_COLS = ['source', 'predicted_state', 'action', 'entry_type', 'entry_style',
              'stop_source', 'target_source', 'actual_state', 'outcome', 'adapt_outcome',
              'hourly_session', 'hourly_zone_kind', 'kill_zone', 'dol_source']
_NUM_COLS = [c for c in JOURNAL_COLUMNS
             if c not in _DATE_COLS and c not in _BOOL_COLS and c not in _TEXT_COLS]


def _journal_path(index_name: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9]+', '_', index_name).strip('_')
    return os.path.join(JOURNAL_DIR, f"{safe}_trade_journal.csv")


def _coerce_journal(j: pd.DataFrame) -> pd.DataFrame:
    j = j.copy()
    for c in JOURNAL_COLUMNS:
        if c not in j.columns:
            j[c] = np.nan
    for c in _DATE_COLS:
        j[c] = pd.to_datetime(j[c], errors='coerce')
    for c in _BOOL_COLS:
        j[c] = j[c].astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
    for c in _NUM_COLS:
        j[c] = pd.to_numeric(j[c], errors='coerce')
    for c in _TEXT_COLS:
        j[c] = j[c].fillna('').astype(str)
        j.loc[j[c].str.lower() == 'nan', c] = ''
    return j[JOURNAL_COLUMNS].reset_index(drop=True)


def _load_journal(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            j = pd.read_csv(path)
            j = _coerce_journal(j)
            j = j.dropna(subset=['signal_date']).drop_duplicates('signal_date', keep='last')
            j = j.sort_values('signal_date').reset_index(drop=True)
            print(f"\n📓 Loaded existing journal: {path} ({len(j)} rows)")
            return j
        except Exception as e:
            bak = path + ".corrupt.bak"
            print(f"\n⚠️  Could not read journal {path} ({e}) — moved to {bak}, starting a new one")
            try:
                os.replace(path, bak)
            except OSError:
                pass
    else:
        print(f"\n📓 No journal found — a new one will be created at {path}")
    return _coerce_journal(pd.DataFrame(columns=JOURNAL_COLUMNS))


def _save_journal(j: pd.DataFrame, path: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    out = j.sort_values('signal_date')[JOURNAL_COLUMNS]
    try:
        out.to_csv(path, index=False)
        print(f"💾 Journal saved: {path} ({len(out)} rows)")
    except OSError as e:
        alt = path.replace('.csv', f"_{int(time.time())}.csv")
        out.to_csv(alt, index=False)
        print(f"⚠️  Could not write {path} ({e}) — is it open in Excel? Saved to {alt} instead")


def _journal_record(row: pd.DataFrame, probs: dict, trade: dict, next_date, source: str) -> dict:
    r = row.iloc[0]
    close_t = float(r['Close'])
    atr = r['ATR_14']
    atr = float(atr) if pd.notna(atr) and atr > 0 else 0.01 * close_t
    dominant = max(probs, key=probs.get)
    rec = {c: np.nan for c in JOURNAL_COLUMNS}
    for c in _TEXT_COLS:
        rec[c] = ''
    for c in _BOOL_COLS:
        rec[c] = False
    rec.update({
        'signal_date': r['Date'], 'target_date': pd.to_datetime(next_date), 'source': source,
        'predicted_state': dominant,
        'prob_up': probs.get('UP', np.nan), 'prob_down': probs.get('DOWN', np.nan),
        'prob_cons': probs.get('CONSOLIDATION', np.nan), 'confidence': probs[dominant],
        'move_threshold': float(r['Move_Threshold']), 'action': trade['action'],
        'close_t': close_t, 'atr': atr,
        'zone_pos': r['Premium_Discount'], 'vol20': r['Volatility_20'], 'rsi': r['RSI'],
        'market_structure': r['Macro_Structure'], 'analog_updown': r['Analog_Up_Minus_Down'],
        'dow': r['DayOfWeek'],
        'dealing_pos_at_signal': r.get('Dealing_Pos', np.nan),
        'dist_to_pdh': r.get('Dist_To_PDH', np.nan),
        'dist_to_pdl': r.get('Dist_To_PDL', np.nan),
    })
    if trade['action'] in ('LONG', 'SHORT'):
        entry = float(trade['entry'])
        rec.update({
            'entry_type': trade['entry_type'], 'entry_style': trade.get('entry_style', ''),
            'entry': entry, 'stop_loss': float(trade['stop_loss']),
            'take_profit': float(trade['take_profit']),
            'stop_source': trade.get('stop_source', ''), 'target_source': trade.get('target_source', ''),
            'is_market': str(trade['entry_type']).startswith('market'),
            'd_s': abs(entry - float(trade['stop_loss'])),
            'd_t': abs(float(trade['take_profit']) - entry),
            'risk_reward': trade.get('risk_reward') if trade.get('risk_reward') is not None else np.nan,
            'zone_strength': trade.get('zone_strength', np.nan),
            'entry_depth_atr': abs(close_t - entry) / atr,
            'daily_struct_entry': trade.get('daily_entry', entry),
            'daily_struct_rr': trade.get('daily_rr', np.nan),
            'bias_gate_passed': bool(trade.get('bias_gate_passed', False)),
            'dol_target': trade.get('dol_target', np.nan),
            'dol_source': trade.get('dol_source', ''),
            'dol_rank': trade.get('dol_rank', np.nan),
        })
        hm = trade.get('hourly_meta')
        if hm is not None:
            rec.update({
                'hourly_hour_utc':        hm.get('hour_utc', np.nan),
                'hourly_session':         hm.get('session', ''),
                'hourly_zone_kind':       hm.get('kind', ''),
                'hourly_entry_price':     hm.get('entry_price', np.nan),
                'hourly_distance_pct':    hm.get('distance_pct', np.nan),
                'hourly_bars_since_zone': hm.get('bars_since_zone', np.nan),
                'hourly_stop_reference':  hm.get('stop_reference', np.nan),
                'hourly_applied':         hm.get('applied', False),
                'hourly_risk_reward':     hm.get('hourly_rr', np.nan),
                'ote_aligned':            bool(hm.get('ote_aligned', False)),
                'ote_705':                hm.get('ote_705', np.nan) if hm.get('ote_705') is not None else np.nan,
                'ote_band_low':           hm.get('ote_band_low', np.nan) if hm.get('ote_band_low') is not None else np.nan,
                'ote_band_high':          hm.get('ote_band_high', np.nan) if hm.get('ote_band_high') is not None else np.nan,
                'kill_zone':              hm.get('kill_zone') or '',
                'session_weight':         hm.get('session_weight', np.nan),
            })
    return rec


# ============================================
# TRADE JOURNAL — REPLAY ENGINE
# ============================================
def _arrays(sub: pd.DataFrame) -> dict:
    return {
        'sign':   np.where(sub['action'].values == 'LONG', 1.0, -1.0),
        'e_ref':  sub['entry'].values.astype(float),
        'is_mkt': sub['is_market'].values.astype(bool),
        'd_s':    sub['d_s'].values.astype(float),
        'd_t':    sub['d_t'].values.astype(float),
        'atr':    sub['atr'].values.astype(float),
        'op':     sub['win_open'].values.astype(float),
        'hi':     sub['win_high'].values.astype(float),
        'lo':     sub['win_low'].values.astype(float),
        'cl':     sub['win_close'].values.astype(float),
        'right':  sub['right_direction'].values.astype(bool),
    }


def _slice(a: dict, s) -> dict:
    return {k: v[s] for k, v in a.items()}


def _replay(a: dict, shift: float = 0.0, extra: float = 0.0, tscale: float = 1.0) -> dict:
    sign, atr = a['sign'], a['atr']
    hi_l = np.where(sign > 0, a['hi'], -a['lo'])
    lo_l = np.where(sign > 0, a['lo'], -a['hi'])
    cl_l = sign * a['cl']
    op_l = sign * a['op']
    e_l = sign * a['e_ref']
    e_lim = e_l - shift * atr
    mkt_fill = a['is_mkt'] if shift == 0 else np.zeros(len(sign), dtype=bool)
    entry = np.where(mkt_fill, op_l, e_lim)
    filled = mkt_fill | (lo_l <= e_lim)
    ds = a['d_s'] + extra * atr
    dt = a['d_t'] * tscale
    stop = entry - ds
    tgt = entry + dt
    stop_hit = filled & (lo_l <= stop)
    tgt_hit = filled & (~stop_hit) & (hi_l >= tgt)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_exit = (cl_l - entry) / ds
        r = np.where(stop_hit, -1.0, np.where(tgt_hit, dt / ds, r_exit))
        mae = np.maximum(entry - lo_l, 0) / atr
        mfe = np.maximum(hi_l - entry, 0) / atr
    return {
        'r': np.where(filled, r, np.nan),
        'filled': filled, 'stop_hit': stop_hit, 'tgt_hit': tgt_hit,
        'mae': np.where(filled, mae, np.nan), 'mfe': np.where(filled, mfe, np.nan),
    }


def _outcome_labels(res: dict):
    return np.where(~res['filled'], 'no_fill',
                    np.where(res['stop_hit'], 'stopped',
                             np.where(res['tgt_hit'], 'target', 'expired')))


def _expectancy(a: dict, shift, extra, tscale):
    res = _replay(a, shift, extra, tscale)
    n = max(len(res['r']), 1)
    return float(np.nansum(res['r']) / n), res


def _resolve_journal(journal: pd.DataFrame, df: pd.DataFrame, hold: int = JOURNAL_HOLD_BARS) -> int:
    pending_idx = journal.index[~journal['resolved']]
    if len(pending_idx) == 0:
        return 0
    win = pd.DataFrame({
        'w_open':  df['Open'].shift(-1).values,
        'w_high':  df['High'].rolling(hold).max().shift(-hold).values,
        'w_low':   df['Low'].rolling(hold).min().shift(-hold).values,
        'w_close': df['Close'].shift(-hold).values,
        'n_close': df['Close'].shift(-1).values,
    }, index=pd.DatetimeIndex(df['Date'].values))
    win = win[~win.index.duplicated(keep='last')]
    w = win.reindex(pd.DatetimeIndex(journal.loc[pending_idx, 'signal_date']))
    ok = w.notna().all(axis=1).values
    idx = pending_idx[ok]
    if len(idx) == 0:
        return 0
    w = w[ok]
    journal.loc[idx, 'win_open'] = w['w_open'].values
    journal.loc[idx, 'win_high'] = w['w_high'].values
    journal.loc[idx, 'win_low'] = w['w_low'].values
    journal.loc[idx, 'win_close'] = w['w_close'].values
    journal.loc[idx, 'next_close'] = w['n_close'].values
    close_t = journal.loc[idx, 'close_t'].values.astype(float)
    thr = journal.loc[idx, 'move_threshold'].values.astype(float)
    actual_ret = w['n_close'].values / close_t - 1
    actual_state = np.where(actual_ret > thr, 'UP',
                            np.where(actual_ret < -thr, 'DOWN', 'CONSOLIDATION'))
    journal.loc[idx, 'actual_return'] = actual_ret
    journal.loc[idx, 'actual_state'] = actual_state
    journal.loc[idx, 'dir_correct'] = (journal.loc[idx, 'predicted_state'].values == actual_state)
    journal.loc[idx, 'resolved'] = True
    is_trade = journal.loc[idx, 'action'].isin(['LONG', 'SHORT']).values
    journal.loc[idx[~is_trade], 'outcome'] = 'no_trade'
    sub = journal.loc[idx[is_trade]]
    good = (sub['d_s'] > 0) & (sub['atr'] > 0) & sub['entry'].notna()
    tidx = sub.index[good.values]
    if len(tidx):
        sgn = np.where(journal.loc[tidx, 'action'].values == 'LONG', 1.0, -1.0)
        nxt = journal.loc[tidx, 'next_close'].values.astype(float)
        ct = journal.loc[tidx, 'close_t'].values.astype(float)
        journal.loc[tidx, 'right_direction'] = (sgn * (nxt - ct) > 0)
        a = _arrays(journal.loc[tidx])
        res = _replay(a)
        journal.loc[tidx, 'filled'] = res['filled']
        journal.loc[tidx, 'outcome'] = _outcome_labels(res)
        journal.loc[tidx, 'pnl_r'] = res['r']
        journal.loc[tidx, 'mae_atr'] = res['mae']
        journal.loc[tidx, 'mfe_atr'] = res['mfe']
        journal.loc[tidx, 'stopped_but_right'] = res['stop_hit'] & a['right']
        sub2 = journal.loc[tidx]
        sub2 = sub2[sub2['adapt_applied']]
        if len(sub2):
            for (s, e, t), g in sub2.groupby(['adapt_shift', 'adapt_extra', 'adapt_tscale']):
                ag = _arrays(journal.loc[g.index])
                rr = _replay(ag, float(s), float(e), float(t))
                journal.loc[g.index, 'adapt_pnl_r'] = rr['r']
                journal.loc[g.index, 'adapt_outcome'] = _outcome_labels(rr)
    return len(idx)


def _print_journal_summary(journal: pd.DataFrame, index_name: str):
    res = journal[journal['resolved']]
    print(f"\n📓 {index_name} JOURNAL: {len(journal)} rows | resolved {len(res)} | "
          f"pending {int((~journal['resolved']).sum())}")
    if res.empty:
        return
    live = res[res['source'] == 'live']
    print(f"   Direction accuracy (all resolved): {res['dir_correct'].mean():.1%}  "
          f"| live-only: {live['dir_correct'].mean():.1%} (n={len(live)})" if len(live)
          else f"   Direction accuracy (all resolved): {res['dir_correct'].mean():.1%}")
    tr = res[res['filled']]
    if len(tr):
        rd = tr[tr['right_direction']]
        sbr = (rd['outcome'] == 'stopped').mean() if len(rd) else float('nan')
        print(f"   Baseline trades: {len(tr)} filled | win {((tr['pnl_r'] > 0).mean()):.1%} "
              f"| stopped {((tr['outcome'] == 'stopped').mean()):.1%} "
              f"| target {((tr['outcome'] == 'target').mean()):.1%} "
              f"| avg {tr['pnl_r'].mean():+.3f}R")
        print(f"   ↳ Of trades where the DIRECTION was right ({len(rd)}), "
              f"{sbr:.1%} were stopped out first")
        if 'hourly_applied' in tr.columns and tr['hourly_applied'].any():
            ha = tr[tr['hourly_applied'] == True]
            hn = tr[tr['hourly_applied'] == False]
            if len(ha):
                print(f"   ↳ Hourly-applied trades: {len(ha)} | avg {ha['pnl_r'].mean():+.3f}R")
            if len(hn):
                print(f"   ↳ Daily-only trades:     {len(hn)} | avg {hn['pnl_r'].mean():+.3f}R")
        if 'ote_aligned' in tr.columns and tr['ote_aligned'].any():
            ote = tr[tr['ote_aligned'] == True]
            non = tr[tr['ote_aligned'] == False]
            if len(ote):
                print(f"   ↳ OTE-aligned trades:   {len(ote):>4} | avg {ote['pnl_r'].mean():+.3f}R")
            if len(non):
                print(f"   ↳ Non-OTE trades:       {len(non):>4} | avg {non['pnl_r'].mean():+.3f}R")
        if 'kill_zone' in tr.columns:
            kz_vals = [k for k in tr['kill_zone'].dropna().unique() if isinstance(k, str) and k]
            for kz in kz_vals:
                sub = tr[tr['kill_zone'] == kz]
                if len(sub):
                    print(f"   ↳ {kz:<14}: {len(sub):>4} trades | avg {sub['pnl_r'].mean():+.3f}R")
        if 'bias_gate_passed' in tr.columns and tr['bias_gate_passed'].any():
            gp = tr[tr['bias_gate_passed'] == True]
            if len(gp):
                print(f"   ↳ Bias-gate-passed:     {len(gp):>4} trades | avg {gp['pnl_r'].mean():+.3f}R")


# ============================================
# TRADE JOURNAL — LEARNING
# ============================================
def _trades_for_learning(journal: pd.DataFrame) -> pd.DataFrame:
    m = (journal['resolved'] & journal['action'].isin(['LONG', 'SHORT'])
         & journal['win_close'].notna() & (journal['d_s'] > 0) & (journal['atr'] > 0)
         & journal['entry'].notna())
    return journal[m].sort_values('signal_date').reset_index(drop=True)


def _grid_search(a: dict) -> dict:
    best = None
    for s in CAL_SHIFT_GRID:
        for e in CAL_EXTRA_STOP_GRID:
            for t in CAL_TARGET_SCALE_GRID:
                exp, _ = _expectancy(a, s, e, t)
                score = exp - CAL_COMPLEXITY_PENALTY * (s + e)
                if best is None or score > best['score']:
                    best = {'score': score, 'exp': exp, 'params': (s, e, t)}
    return best


def _fit_calibration_subset(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n < CAL_MIN_TRADES:
        return {'accepted': False, 'params': None, 'n': n,
                'reason': f'only {n} resolved trades (need {CAL_MIN_TRADES})'}
    a = _arrays(sub)
    cut = int(n * CAL_TRAIN_FRACTION)
    a_tr, a_ho = _slice(a, slice(0, cut)), _slice(a, slice(cut, n))
    best_tr = _grid_search(a_tr)
    base_ho, _ = _expectancy(a_ho, 0.0, 0.0, 1.0)
    learn_ho, _ = _expectancy(a_ho, *best_tr['params'])
    base_all, _ = _expectancy(a, 0.0, 0.0, 1.0)
    out = {'n': n, 'hold_base': base_ho, 'hold_learn': learn_ho, 'base_all': base_all,
           'accepted': False, 'params': None}
    improved = (learn_ho - base_ho) >= CAL_MIN_IMPROVEMENT and best_tr['params'] != (0.0, 0.0, 1.0)
    if not improved:
        out['reason'] = 'no adjustment beat the baseline on the validation slice'
        return out
    final = _grid_search(a)
    out.update({'accepted': True, 'params': final['params'], 'learn_all': final['exp']})
    return out


def _fit_calibrations(trades: pd.DataFrame) -> dict:
    out = {'ALL': _fit_calibration_subset(trades)}
    for side in ('LONG', 'SHORT'):
        out[side] = _fit_calibration_subset(trades[trades['action'] == side])
    return out


def _pick_calibration(cals, side):
    if not cals:
        return None
    c = cals.get(side)
    if c and c.get('n', 0) >= CAL_MIN_TRADES:
        return c['params'] if c.get('accepted') else None
    c = cals.get('ALL')
    if c and c.get('accepted'):
        return c['params']
    return None


def _apply_calibration(trade: dict, close: float, atr: float, params) -> dict:
    shift, extra, ts = params
    sgn = 1.0 if trade['action'] == 'LONG' else -1.0
    e0 = float(trade['entry'])
    d_s = abs(e0 - float(trade['stop_loss']))
    d_t = abs(float(trade['take_profit']) - e0)
    is_mkt = str(trade['entry_type']).startswith('market')
    if is_mkt and shift == 0:
        e1 = close
    elif is_mkt:
        e1 = close - sgn * shift * atr
    else:
        e1 = e0 - sgn * shift * atr
    s1 = e1 - sgn * (d_s + extra * atr)
    t1 = e1 + sgn * d_t * ts
    risk, reward = abs(e1 - s1), abs(t1 - e1)
    new = dict(trade)
    new['entry'] = round(e1, 2)
    new['stop_loss'] = round(s1, 2)
    new['take_profit'] = round(t1, 2)
    new['risk_reward'] = round(reward / risk, 2) if risk > 0 else None
    new['stop_source'] = f"{trade.get('stop_source', '')} +{extra:.2f} ATR (journal)"
    new['target_source'] = f"{trade.get('target_source', '')} x{ts:.2f} (journal)"
    if is_mkt and shift > 0:
        new['entry_type'] = f"limit (journal-learned: {shift:.2f} ATR from close)"
    elif shift > 0:
        new['entry_type'] = f"{trade['entry_type']} + {shift:.2f} ATR deeper (journal)"
    return new


META_FEATURES = [
    'prob_up', 'prob_down', 'prob_cons', 'confidence', 'side_sign',
    'zone_pos', 'vol20', 'rsi', 'market_structure', 'analog_updown',
    'atr_pct', 'd_s_atr', 'd_t_atr', 'risk_reward', 'entry_depth_atr',
    'is_market', 'dow', 'zone_strength',
]


def _meta_matrix(j: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=j.index)
    atr = pd.to_numeric(j['atr'], errors='coerce').replace(0, np.nan)
    close_t = pd.to_numeric(j['close_t'], errors='coerce')
    for c in ['prob_up', 'prob_down', 'prob_cons', 'confidence', 'zone_pos', 'vol20', 'rsi',
              'market_structure', 'analog_updown', 'risk_reward', 'entry_depth_atr', 'dow',
              'zone_strength']:
        X[c] = pd.to_numeric(j[c], errors='coerce')
    X['side_sign'] = np.where(j['action'].values == 'LONG', 1.0, -1.0)
    X['atr_pct'] = atr / close_t
    X['d_s_atr'] = pd.to_numeric(j['d_s'], errors='coerce') / atr
    X['d_t_atr'] = pd.to_numeric(j['d_t'], errors='coerce') / atr
    X['is_market'] = j['is_market'].astype(float)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)[META_FEATURES]


def _meta_model():
    return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=15,
                                  class_weight='balanced', random_state=42, n_jobs=1)


def _fit_meta(trades: pd.DataFrame):
    t = trades[trades['filled'] & trades['pnl_r'].notna()].reset_index(drop=True)
    n = len(t)
    if n < META_MIN_TRADES:
        return {'accepted': False, 'n': n, 'reason': f'only {n} filled trades (need {META_MIN_TRADES})'}
    X = _meta_matrix(t)
    y = (t['pnl_r'] > 0).astype(int).values
    cut = int(n * CAL_TRAIN_FRACTION)
    if len(np.unique(y[:cut])) < 2 or len(np.unique(y[cut:])) < 2:
        return {'accepted': False, 'n': n, 'reason': 'one outcome class missing in a split'}
    m = _meta_model().fit(X.iloc[:cut], y[:cut])
    auc = roc_auc_score(y[cut:], m.predict_proba(X.iloc[cut:])[:, 1])
    out = {'n': n, 'auc': float(auc), 'base_win_rate': float(y.mean()), 'accepted': False}
    if auc < META_MIN_AUC:
        out['reason'] = f'hold-out AUC {auc:.3f} < {META_MIN_AUC} (not informative yet)'
        return out
    out['model'] = _meta_model().fit(X, y)
    out['accepted'] = True
    return out


def _meta_predict(meta: dict, rec: dict) -> float:
    X = _meta_matrix(pd.DataFrame([rec]))
    return float(meta['model'].predict_proba(X)[0][1])


def _summ(label, r, filled, stop, right):
    n = len(r)
    nf = int(filled.sum())
    exp = float(np.nansum(r) / n) if n else float('nan')
    win = float((r[filled] > 0).mean()) if nf else float('nan')
    stp = float(stop[filled].mean()) if nf else float('nan')
    rf = filled & right
    sbr = float((stop & right).sum() / rf.sum()) if rf.sum() else float('nan')
    print(f"   {label:<10}{n:>8}{nf:>8}{win:>9.1%}{stp:>10.1%}{sbr:>15.1%}{exp:>14.3f}")


def _walkforward_report(journal: pd.DataFrame, index_name: str):
    t = _trades_for_learning(journal)
    n = len(t)
    if n < CAL_MIN_TRADES * 2:
        print(f"\n🧪 {index_name} walk-forward check skipped ({n} resolved trades, "
              f"need {CAL_MIN_TRADES * 2})")
        return
    a_all = _arrays(t)
    start = max(CAL_MIN_TRADES, int(n * 0.5))
    step = max(25, (n - start) // 40)
    store = {k: {'r': [], 'filled': [], 'stop': [], 'right': []} for k in ('base', 'adapt')}
    for i in range(start, n, step):
        j2 = min(i + step, n)
        cals = _fit_calibrations(t.iloc[:i])
        chunk = _slice(a_all, slice(i, j2))
        for side, sgn in (('LONG', 1.0), ('SHORT', -1.0)):
            m = chunk['sign'] == sgn
            if not m.any():
                continue
            c = {k: v[m] for k, v in chunk.items()}
            params = _pick_calibration(cals, side) or (0.0, 0.0, 1.0)
            for key, p in (('base', (0.0, 0.0, 1.0)), ('adapt', params)):
                res = _replay(c, *p)
                store[key]['r'].append(res['r'])
                store[key]['filled'].append(res['filled'])
                store[key]['stop'].append(res['stop_hit'])
                store[key]['right'].append(c['right'])
    print(f"\n🧪 {index_name} WALK-FORWARD CHECK — journal adjustments learned only from earlier trades")
    print(f"   {'':<10}{'signals':>8}{'filled':>8}{'win%':>9}{'stopped%':>10}"
          f"{'stopped|right':>15}{'avg R/signal':>14}")
    for label, key in (('baseline', 'base'), ('adjusted', 'adapt')):
        d = store[key]
        if not d['r']:
            continue
        _summ(label, np.concatenate(d['r']), np.concatenate(d['filled']),
              np.concatenate(d['stop']), np.concatenate(d['right']))


# ============================================
# JOURNAL PROGRESS GRAPH
# ============================================
def save_journal_progress(journal: pd.DataFrame, index_name: str, graph_dir: str):
    if journal is None or len(journal) == 0:
        print(f"⏭️  Skipping journal_progress.png for {index_name} — journal empty")
        return
    j = journal.copy()
    j['signal_date'] = pd.to_datetime(j['signal_date'], errors='coerce')
    j = j.dropna(subset=['signal_date']).sort_values('signal_date').reset_index(drop=True)
    res = j[j['resolved']].copy()
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle(f"{index_name} — Journal Progress  |  "
                 f"{len(j)} rows logged, {len(res)} resolved, "
                 f"{int((~j['resolved']).sum())} pending",
                 fontsize=14, fontweight='bold')
    ax = axes[0, 0]
    counts = np.arange(1, len(j) + 1)
    ax.plot(j['signal_date'], counts, color='steelblue', linewidth=1.8)
    if len(res):
        ax.axvline(res['signal_date'].iloc[-1], color='green', linestyle='--',
                   alpha=0.5, label=f"Last resolved ({res['signal_date'].iloc[-1].date()})")
        ax.legend(loc='upper left', fontsize=8)
    ax.set_title(f"Cumulative predictions logged ({len(j)} total)")
    ax.set_ylabel("Rows in journal")
    ax.grid(alpha=0.3); ax.tick_params(axis='x', rotation=30)
    ax = axes[0, 1]
    if len(res) >= 5:
        dc = res['dir_correct'].astype(float).values
        window = min(20, max(5, len(dc) // 5))
        rolling = pd.Series(dc).rolling(window, min_periods=max(3, window // 2)).mean()
        overall = float(np.nanmean(dc))
        ax.plot(res['signal_date'], rolling * 100, color='darkorange',
                linewidth=1.8, label=f'Rolling ({window}-trade)')
        ax.axhline(overall * 100, color='black', linestyle='--',
                   linewidth=1.2, label=f'Overall {overall:.1%}')
        ax.axhline(100/3, color='red', linestyle=':', linewidth=1.0,
                   label='Random baseline (33.3%)')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Direction accuracy (%)")
        ax.legend(loc='lower left', fontsize=8)
    else:
        ax.text(0.5, 0.5, f"Need ≥5 resolved trades\n(have {len(res)})",
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_title("Direction accuracy (all 3 classes)")
    ax.grid(alpha=0.3); ax.tick_params(axis='x', rotation=30)
    ax = axes[1, 0]
    traded = res[res['action'].isin(['LONG', 'SHORT'])].copy()
    outcome_map = {'target': '#2ca02c', 'stopped': '#d62728',
                   'expired': '#7f7f7f', 'no_fill': '#bcbd22', 'stop_first': '#8c564b'}
    if len(traded):
        traded['outcome'] = traded['outcome'].fillna('unknown')
        order = ['target', 'expired', 'stopped', 'no_fill', 'stop_first', 'unknown']
        colors = [outcome_map.get(o, '#cccccc') for o in order]
        pivot = (traded.groupby(['signal_date', 'outcome']).size()
                 .unstack(fill_value=0).reindex(columns=order, fill_value=0))
        pivot = pivot.reindex(traded['signal_date'].unique()).fillna(0)
        bottom = np.zeros(len(pivot))
        for col, color in zip(order, colors):
            if col in pivot.columns and pivot[col].sum() > 0:
                ax.bar(pivot.index, pivot[col].values, bottom=bottom,
                       color=color, label=col, width=1.0)
                bottom += pivot[col].values
        ax.legend(loc='upper left', fontsize=8, ncol=3)
    else:
        ax.text(0.5, 0.5, f"No filled trades yet\n({len(res)} resolved, none traded)",
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_title(f"Trade outcomes over time ({len(traded)} filled trades)")
    ax.set_ylabel("Trades per day")
    ax.grid(alpha=0.3, axis='y'); ax.tick_params(axis='x', rotation=30)
    ax = axes[1, 1]
    r_series = res[res['filled']]['pnl_r'].dropna()
    if len(r_series) >= 2:
        cum_r = r_series.cumsum().values
        dates = res.loc[r_series.index, 'signal_date']
        ax.plot(dates, cum_r, color='purple', linewidth=2.0)
        ax.fill_between(dates, 0, cum_r, where=(cum_r >= 0), color='green', alpha=0.15)
        ax.fill_between(dates, 0, cum_r, where=(cum_r < 0), color='red', alpha=0.15)
        ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
        final_r = cum_r[-1]
        ax.annotate(f"{final_r:+.2f}R total\n({len(r_series)} trades, "
                    f"avg {final_r/len(r_series):+.3f}R)",
                    xy=(dates.iloc[-1], final_r),
                    xytext=(-10, 10 if final_r >= 0 else -25),
                    textcoords='offset points', fontsize=9, fontweight='bold', ha='right')
    elif len(r_series) == 1:
        cum_r = r_series.cumsum().values
        dates = res.loc[r_series.index, 'signal_date']
        ax.scatter(dates, cum_r, color='purple', s=80, zorder=3)
        ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
        ax.annotate(f"{cum_r[-1]:+.2f}R (1 trade so far)",
                    xy=(dates.iloc[-1], cum_r[-1]),
                    xytext=(-10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold', ha='right')
    else:
        ax.text(0.5, 0.5, f"Need ≥1 filled trade\n(have {len(r_series)})",
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_title("Cumulative R (baseline levels)")
    ax.set_ylabel("Cumulative R")
    ax.grid(alpha=0.3); ax.tick_params(axis='x', rotation=30)
    ax = axes[2, 0]
    correct = res[(res['filled'] == True) & (res['right_direction'] == True) &
                  res['mae_atr'].notna() & res['mfe_atr'].notna()].copy()
    if len(correct):
        colors = correct['outcome'].map(outcome_map).fillna('#cccccc')
        ax.scatter(correct['mae_atr'], correct['mfe_atr'],
                   c=colors, s=35, alpha=0.75, edgecolors='black', linewidth=0.4)
        max_val = max(float(correct['mae_atr'].max()), float(correct['mfe_atr'].max()), 1.0)
        ax.plot([0, max_val], [0, max_val], color='black', linestyle='--',
                linewidth=0.9, label='MAE = MFE')
        ax.set_xlabel("MAE (ATR against entry)")
        ax.set_ylabel("MFE (ATR in favour)")
        ax.set_title(f"Direction-correct trades: MAE vs MFE (n={len(correct)})")
        sbr = correct[correct['stopped_but_right'] == True]
        if len(sbr):
            ax.annotate(f"{len(sbr)} stopped\nbut right",
                        xy=(float(sbr['mae_atr'].mean()), float(sbr['mfe_atr'].mean())),
                        xytext=(15, 15), textcoords='offset points',
                        fontsize=9, color='darkred', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.6))
        ax.legend(loc='lower right', fontsize=8)
    else:
        ax.text(0.5, 0.5, f"Need direction-correct filled trades\n(have {len(correct)})",
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    ax.grid(alpha=0.3)
    ax = axes[2, 1]
    traded_res = res[(res['filled'] == True) & res['confidence'].notna() &
                     res['pnl_r'].notna()].copy()
    if len(traded_res) >= 10:
        traded_res['win'] = (traded_res['pnl_r'] > 0).astype(float)
        bins = np.arange(0.30, 0.75, 0.05)
        traded_res['bin'] = pd.cut(traded_res['confidence'], bins)
        grouped = traded_res.groupby('bin', observed=True).agg(
            mean_conf=('confidence', 'mean'), win_rate=('win', 'mean'), n=('win', 'size'),
        ).reset_index()
        grouped = grouped[grouped['n'] >= 3]
        if len(grouped):
            ax.plot([0.30, 0.75], [0.30, 0.75], color='black', linestyle='--',
                    linewidth=0.9, label='Perfect calibration')
            sizes = 30 + 20 * np.sqrt(grouped['n'].values)
            ax.scatter(grouped['mean_conf'], grouped['win_rate'],
                       s=sizes, color='steelblue', alpha=0.75,
                       edgecolors='black', linewidth=0.5, zorder=3)
            ax.plot(grouped['mean_conf'], grouped['win_rate'],
                    color='steelblue', linewidth=1.2, alpha=0.5)
            for _, r in grouped.iterrows():
                ax.annotate(f"n={int(r['n'])}",
                            xy=(float(r['mean_conf']), float(r['win_rate'])),
                            xytext=(5, -12), textcoords='offset points',
                            fontsize=7, alpha=0.7)
            ax.set_xlabel("Predicted confidence")
            ax.set_ylabel("Actual win rate")
            ax.set_xlim(0.28, 0.75); ax.set_ylim(0.15, 0.75)
            ax.legend(loc='upper left', fontsize=8)
        else:
            ax.text(0.5, 0.5, "Not enough data per bin (need ≥3)",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    else:
        ax.text(0.5, 0.5, f"Need ≥10 filled trades\n(have {len(traded_res)})",
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_title("Confidence calibration")
    ax.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = _safe_savefig(fig, graph_dir, "journal_progress.png")
    plt.close(fig)
    print(f"💾 Saved: {path}")


def save_cross_index_journal_summary(all_journals: dict, graph_dir: str = "."):
    if not all_journals:
        return
    valid = {k: v for k, v in all_journals.items()
             if v is not None and len(v) and len(v[v['filled']]) >= 1}
    if not valid:
        print(f"⏭️  Skipping journal_summary.png — no index has any filled trades yet")
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    any_plotted = False
    for i, (name, j) in enumerate(valid.items()):
        res = j[j['resolved'] & j['filled']].copy()
        res['signal_date'] = pd.to_datetime(res['signal_date'], errors='coerce')
        res = res.sort_values('signal_date')
        r = res['pnl_r'].dropna()
        if len(r) < 1:
            continue
        cum = r.cumsum().values
        color = colors[i % len(colors)]
        if len(r) >= 2:
            ax.plot(res.loc[r.index, 'signal_date'], cum,
                    color=color, linewidth=1.8,
                    label=f"{name} ({len(r)} trades, {cum[-1]:+.2f}R)")
        else:
            ax.scatter(res.loc[r.index, 'signal_date'], cum,
                       color=color, s=90, zorder=3,
                       label=f"{name} ({len(r)} trade, {cum[-1]:+.2f}R)")
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        print(f"⏭️  Skipping journal_summary.png — no plottable trades yet")
        return
    ax.axhline(0, color='black', linestyle=':', linewidth=0.9)
    ax.set_title("Journal cumulative R — cross-index comparison (baseline levels)",
                 fontsize=13, fontweight='bold')
    ax.set_ylabel("Cumulative R"); ax.set_xlabel("Signal date")
    ax.legend(loc='best', fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = _safe_savefig(fig, graph_dir, "journal_summary.png")
    plt.close(fig)
    print(f"💾 Saved: {path}")


# ============================================
# MASTER PIPELINE
# ============================================

def run_single_index(csv_path: str, index_name: str, dxy_df=None):
    print("\n\n" + "#"*80)
    print(f"#  PROCESSING: {index_name}  ({csv_path})")
    print("#"*80)

    if not os.path.exists(csv_path):
        print(f"⚠️  Skipping {index_name}: file not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df['Date'] = _parse_dates(df['Date'])
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    HAS_VOLUME = 'Volume' in df.columns and df['Volume'].notna().any()
    if HAS_VOLUME:
        df['Volume'] = pd.to_numeric(
            df['Volume'].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
    else:
        print(f"ℹ️  No Volume column detected — volume features will be disabled")
        df['Volume'] = 0.0

    print(f"\n📊 Data loaded: {len(df)} rows")
    print(f"📅 Range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"🔊 Volume available: {HAS_VOLUME}")

    hourly_csv_path = HOURLY_CONFIGS.get(index_name)
    hourly_df = load_hourly_csv(hourly_csv_path) if hourly_csv_path else None
    HAS_HOURLY = hourly_df is not None and len(hourly_df) > 500

    if HAS_HOURLY:
        print(f"\n⏰ Hourly data loaded: {len(hourly_df)} bars "
              f"({hourly_df['timestamp'].min().date()} → {hourly_df['timestamp'].max().date()})")
        print(f"   Used by Model 2 (hourly entry optimiser)")
    else:
        print(f"\nℹ️  No hourly data — model 2 will be inactive for this index")

    if HAS_HOURLY and HOURLY_OPT_ENABLED:
        zl.fit(index_name, hourly_df)

    # ---------------- 2. CLASSIC FEATURES ----------------
    print("\n🔧 Building classic technical features...")
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    df['Price_vs_MA5'] = df['Close'] / df['MA_5'] - 1
    df['Price_vs_MA20'] = df['Close'] / df['MA_20'] - 1
    df['Price_vs_MA50'] = df['Close'] / df['MA_50'] - 1
    df['MA5_vs_MA20'] = df['MA_5'] / df['MA_20'] - 1
    df['MA50_vs_MA200'] = df['MA_50'] / df['MA_200'] - 1
    df['Volatility_5'] = df['Return'].rolling(5).std()
    df['Volatility_20'] = df['Return'].rolling(20).std()
    df['Volatility_50'] = df['Return'].rolling(50).std()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    if HAS_VOLUME:
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_vs_MA20'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    else:
        df['Volume_Change'] = 0.0
        df['Volume_vs_MA20'] = 1.0
        df['Volume_Zscore'] = 0.0
    df['Body'] = (df['Close'] - df['Open']) / df['Open']
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag{lag}'] = df['Return'].shift(lag)

    print("🔧 Building orthogonal microstructure features...")
    if HAS_VOLUME:
        df['Volume_Price_Divergence'] = (
            df['Return'].rolling(10).corr(df['Volume'].pct_change()))
    else:
        df['Volume_Price_Divergence'] = 0.0
    df['Vol_ROC_5'] = df['Volatility_20'].pct_change(5)
    df['Vol_ROC_10'] = df['Volatility_20'].pct_change(10)

    def _consecutive_up(returns):
        up = (returns > 0).astype(int)
        group = (returns <= 0).cumsum()
        return up.groupby(group).cumsum()
    def _consecutive_down(returns):
        down = (returns < 0).astype(int)
        group = (returns >= 0).cumsum()
        return down.groupby(group).cumsum()
    df['Consec_Up'] = _consecutive_up(df['Return'])
    df['Consec_Down'] = _consecutive_down(df['Return'])
    df['Overnight_Return'] = df['Open'] / df['Close'].shift(1) - 1
    df['Intraday_Return'] = df['Close'] / df['Open'] - 1
    df['Overnight_vs_Intraday'] = df['Overnight_Return'] - df['Intraday_Return']
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    HAS_DXY = False
    dxy_features = []
    if dxy_df is not None and len(dxy_df) > 0:
        print("🔧 Merging DXY cross-asset features...")
        _dxy_cols = ['Date', 'DXY_Close', 'DXY_Return', 'DXY_vs_MA20', 'DXY_RSI',
                     'DXY_Return_Lag1', 'DXY_Return_Lag5']
        df = df.merge(dxy_df[_dxy_cols], on='Date', how='left')
        df['Corr_20d_DXY'] = df['Return'].rolling(20).corr(df['DXY_Return'])
        df['Corr_Change_20d'] = df['Corr_20d_DXY'] - df['Corr_20d_DXY'].shift(5)
        df['Relative_Strength_vs_DXY'] = df['Return'] - df['DXY_Return']
        dxy_neutral_fill = {
            'DXY_Return': 0.0, 'DXY_vs_MA20': 0.0, 'DXY_RSI': 50.0,
            'DXY_Return_Lag1': 0.0, 'DXY_Return_Lag5': 0.0,
            'Corr_20d_DXY': 0.0, 'Corr_Change_20d': 0.0,
            'Relative_Strength_vs_DXY': 0.0,
        }
        for col, val in dxy_neutral_fill.items():
            df[col] = df[col].fillna(val)
        dxy_features = ['DXY_Return', 'DXY_Return_Lag1', 'DXY_vs_MA20', 'DXY_RSI',
                        'DXY_Return_Lag5', 'Corr_20d_DXY', 'Corr_Change_20d',
                        'Relative_Strength_vs_DXY']
        HAS_DXY = True
        n_dxy_rows = df['DXY_Close'].notna().sum()
        print(f"   ✅ DXY merged — {n_dxy_rows} rows with DXY data")
    else:
        print("ℹ️  DXY unavailable — skipping cross-asset features")

    print("🔧 Building ICT/SMC features (multi-timeframe)...")
    def _find_swings(frame, left, right):
        highs = frame['High'].values
        lows = frame['Low'].values
        n = len(frame)
        swing_high = np.zeros(n, dtype=bool)
        swing_low = np.zeros(n, dtype=bool)
        for i in range(left, n - right):
            window_h = highs[i - left:i + right + 1]
            window_l = lows[i - left:i + right + 1]
            if highs[i] == window_h.max():
                swing_high[i] = True
            if lows[i] == window_l.min():
                swing_low[i] = True
        return swing_high, swing_low

    def _build_structure_state(frame, swing_left, swing_right, label):
        swing_high_raw, swing_low_raw = _find_swings(frame, swing_left, swing_right)
        raw_high_val = pd.Series(np.where(swing_high_raw, frame['High'].values, np.nan))
        raw_low_val = pd.Series(np.where(swing_low_raw, frame['Low'].values, np.nan))
        last_swing_high = raw_high_val.shift(swing_right).ffill()
        last_swing_low = raw_low_val.shift(swing_right).ffill()
        bos_bull = (frame['Close'] > last_swing_high).astype(int)
        bos_bear = (frame['Close'] < last_swing_low).astype(int)
        trend = np.zeros(len(frame))
        cur = 0
        b = bos_bull.values
        s = bos_bear.values
        for i in range(len(frame)):
            if b[i]:
                cur = 1
            elif s[i]:
                cur = -1
            trend[i] = cur
        structure = pd.Series(trend, index=frame.index)
        choch = ((structure != structure.shift(1)) & (structure != 0)).astype(int)
        return {
            f'{label}_Swing_High': last_swing_high,
            f'{label}_Swing_Low': last_swing_low,
            f'{label}_BOS_Bull': bos_bull,
            f'{label}_BOS_Bear': bos_bear,
            f'{label}_Structure': structure,
            f'{label}_CHoCH': choch,
        }

    macro = _build_structure_state(df, swing_left=10, swing_right=10, label='Macro')
    for k, v in macro.items():
        df[k] = v
    micro = _build_structure_state(df, swing_left=2, swing_right=2, label='Micro')
    for k, v in micro.items():
        df[k] = v
    df['Last_Swing_High'] = df['Macro_Swing_High']
    df['Last_Swing_Low'] = df['Macro_Swing_Low']
    df['BOS_Bull'] = df['Macro_BOS_Bull']
    df['BOS_Bear'] = df['Macro_BOS_Bear']
    df['Market_Structure'] = df['Macro_Structure']
    df['CHoCH'] = df['Macro_CHoCH']

    def _consecutive_counter(macro_struct, micro_struct):
        n = len(macro_struct)
        counter = np.zeros(n)
        cur_count = 0
        for i in range(n):
            if macro_struct[i] != 0 and micro_struct[i] != 0 and macro_struct[i] != micro_struct[i]:
                cur_count += 1
            else:
                cur_count = 0
            counter[i] = cur_count
        return counter
    df['Counter_Trend_Bars'] = _consecutive_counter(
        df['Macro_Structure'].values, df['Micro_Structure'].values)
    df['Pullback_In_Uptrend_MTF'] = (
        (df['Macro_Structure'] == 1) & (df['Micro_Structure'] == -1) &
        (df['Counter_Trend_Bars'] >= 1) & (df['Counter_Trend_Bars'] <= 6)
    ).astype(int)
    df['Bounce_In_Downtrend_MTF'] = (
        (df['Macro_Structure'] == -1) & (df['Micro_Structure'] == 1) &
        (df['Counter_Trend_Bars'] >= 1) & (df['Counter_Trend_Bars'] <= 6)
    ).astype(int)

    _fvg_bull_top, _fvg_bull_bottom = [], []
    _fvg_bear_top, _fvg_bear_bottom = [], []
    _last_bull_top = _last_bull_bottom = np.nan
    _last_bear_top = _last_bear_bottom = np.nan
    _high2 = df['High'].shift(2).values
    _low2  = df['Low'].shift(2).values
    _high  = df['High'].values
    _low   = df['Low'].values
    _fvg_bull_flag = np.zeros(len(df), dtype=int)
    _fvg_bear_flag = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if i >= 2 and not np.isnan(_high2[i]) and _low[i] > _high2[i]:
            _fvg_bull_flag[i] = 1
            _last_bull_bottom = _high2[i]
            _last_bull_top = _low[i]
        if i >= 2 and not np.isnan(_low2[i]) and _high[i] < _low2[i]:
            _fvg_bear_flag[i] = 1
            _last_bear_top = _low2[i]
            _last_bear_bottom = _high[i]
        _fvg_bull_top.append(_last_bull_top)
        _fvg_bull_bottom.append(_last_bull_bottom)
        _fvg_bear_top.append(_last_bear_top)
        _fvg_bear_bottom.append(_last_bear_bottom)
    df['FVG_Bull'] = _fvg_bull_flag
    df['FVG_Bear'] = _fvg_bear_flag
    df['FVG_Bull_Top']    = _fvg_bull_top
    df['FVG_Bull_Bottom'] = _fvg_bull_bottom
    df['FVG_Bear_Top']    = _fvg_bear_top
    df['FVG_Bear_Bottom'] = _fvg_bear_bottom
    df['FVG_Bull_Mid'] = (df['FVG_Bull_Top'] + df['FVG_Bull_Bottom']) / 2
    df['FVG_Bear_Mid'] = (df['FVG_Bear_Top'] + df['FVG_Bear_Bottom']) / 2
    df['FVG_Bull_Size'] = np.where(df['FVG_Bull'] == 1,
                                    (df['FVG_Bull_Top'] - df['FVG_Bull_Bottom']) / df['Close'], 0.0)
    df['FVG_Bear_Size'] = np.where(df['FVG_Bear'] == 1,
                                    (df['FVG_Bear_Top'] - df['FVG_Bear_Bottom']) / df['Close'], 0.0)
    df['Dist_To_Last_Bull_FVG'] = (df['Close'] - df['FVG_Bull_Bottom']) / df['Close']
    df['Dist_To_Last_Bear_FVG'] = (df['FVG_Bear_Top'] - df['Close']) / df['Close']
    df['Bull_IFVG'] = ((df['FVG_Bull_Bottom'].notna()) & (df['Close'] < df['FVG_Bull_Bottom'])).astype(int)
    df['Bear_IFVG'] = ((df['FVG_Bear_Top'].notna()) & (df['Close'] > df['FVG_Bear_Top'])).astype(int)
    df['Dist_To_Bull_IFVG'] = np.where(df['FVG_Bull_Bottom'].notna(),
                                        (df['Close'] - df['FVG_Bull_Bottom']) / df['Close'], np.nan)
    df['Dist_To_Bear_IFVG'] = np.where(df['FVG_Bear_Top'].notna(),
                                        (df['FVG_Bear_Top'] - df['Close']) / df['Close'], np.nan)

    _body = df['Close'] - df['Open']
    _avg_body = _body.abs().rolling(20).mean()
    _displacement_up = _body > 1.5 * _avg_body
    _displacement_down = (-_body) > 1.5 * _avg_body
    df['Bull_OB'] = ((_body.shift(1) < 0) & _displacement_up).astype(int)
    df['Bear_OB'] = ((_body.shift(1) > 0) & _displacement_down).astype(int)
    _open_s1  = df['Open'].shift(1)
    _close_s1 = df['Close'].shift(1)
    _high_s1  = df['High'].shift(1)
    _low_s1   = df['Low'].shift(1)
    _body_top_s1 = pd.concat([_open_s1, _close_s1], axis=1).max(axis=1)
    _body_bottom_s1 = pd.concat([_open_s1, _close_s1], axis=1).min(axis=1)
    df['Bull_OB_Body_Top']    = pd.Series(np.where(df['Bull_OB'] == 1, _body_top_s1, np.nan)).ffill()
    df['Bull_OB_Body_Bottom'] = pd.Series(np.where(df['Bull_OB'] == 1, _body_bottom_s1, np.nan)).ffill()
    df['Bull_OB_Wick_Bottom'] = pd.Series(np.where(df['Bull_OB'] == 1, _low_s1, np.nan)).ffill()
    df['Bear_OB_Body_Top']    = pd.Series(np.where(df['Bear_OB'] == 1, _body_top_s1, np.nan)).ffill()
    df['Bear_OB_Body_Bottom'] = pd.Series(np.where(df['Bear_OB'] == 1, _body_bottom_s1, np.nan)).ffill()
    df['Bear_OB_Wick_Top']    = pd.Series(np.where(df['Bear_OB'] == 1, _high_s1, np.nan)).ffill()
    df['Bull_OB_Entry_Aggressive']   = df['Bull_OB_Body_Top']
    df['Bull_OB_Entry_Moderate']     = (df['Bull_OB_Body_Bottom'] + df['Bull_OB_Wick_Bottom']) / 2
    df['Bull_OB_Entry_Conservative'] = df['Bull_OB_Wick_Bottom']
    df['Bear_OB_Entry_Aggressive']   = df['Bear_OB_Body_Bottom']
    df['Bear_OB_Entry_Moderate']     = (df['Bear_OB_Body_Top'] + df['Bear_OB_Wick_Top']) / 2
    df['Bear_OB_Entry_Conservative'] = df['Bear_OB_Wick_Top']
    df['Bull_OB_Level'] = df['Bull_OB_Body_Top']
    df['Bear_OB_Level'] = df['Bear_OB_Body_Bottom']
    df['Dist_To_Bull_OB'] = (df['Close'] - df['Bull_OB_Level']) / df['Close']
    df['Dist_To_Bear_OB'] = (df['Bear_OB_Level'] - df['Close']) / df['Close']
    _ob_candle_range = (_high_s1 - _low_s1).replace(0, np.nan)
    df['Bull_OB_Body_Ratio'] = pd.Series(np.where(
        df['Bull_OB'] == 1, (_body_top_s1 - _body_bottom_s1).abs() / _ob_candle_range, np.nan)).ffill()
    df['Bear_OB_Body_Ratio'] = pd.Series(np.where(
        df['Bear_OB'] == 1, (_body_top_s1 - _body_bottom_s1).abs() / _ob_candle_range, np.nan)).ffill()
    df['Bull_BB'] = ((df['Bull_OB_Level'].notna()) & (df['Close'] < df['Bull_OB_Level'])).astype(int)
    df['Bear_BB'] = ((df['Bear_OB_Level'].notna()) & (df['Close'] > df['Bear_OB_Level'])).astype(int)
    df['Dist_To_Bull_BB'] = np.where(df['Bull_OB_Level'].notna(),
                                      (df['Close'] - df['Bull_OB_Level']) / df['Close'], np.nan)
    df['Dist_To_Bear_BB'] = np.where(df['Bear_OB_Level'].notna(),
                                      (df['Bear_OB_Level'] - df['Close']) / df['Close'], np.nan)
    EQ_TOL = 0.0015
    df['Equal_High'] = (abs(df['High'] - df['High'].shift(1)) / df['High'] < EQ_TOL).astype(int)
    df['Equal_Low'] = (abs(df['Low'] - df['Low'].shift(1)) / df['Low'] < EQ_TOL).astype(int)
    df['Liquidity_Sweep_High'] = ((df['High'] > df['Last_Swing_High']) &
                                   (df['Close'] < df['Last_Swing_High'])).astype(int)
    df['Liquidity_Sweep_Low'] = ((df['Low'] < df['Last_Swing_Low']) &
                                  (df['Close'] > df['Last_Swing_Low'])).astype(int)
    _range_high = df['High'].rolling(20).max()
    _range_low = df['Low'].rolling(20).min()
    df['Premium_Discount'] = (df['Close'] - _range_low) / (_range_high - _range_low)
    df['Displacement'] = (_body.abs() > 1.5 * _avg_body).astype(int)

    def _compute_atr(frame, period=14):
        prev_close = frame['Close'].shift(1)
        tr = pd.concat([
            frame['High'] - frame['Low'],
            (frame['High'] - prev_close).abs(),
            (frame['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    df['ATR_14'] = _compute_atr(df)

    print("🔧 Building dealing-range + PDH/PDL anchors...")
    _dr_high = df['High'].rolling(50, min_periods=10).max()
    _dr_low  = df['Low'].rolling(50, min_periods=10).min()
    df['Dealing_High'] = _dr_high
    df['Dealing_Low']  = _dr_low
    df['Dealing_EQ']   = (df['Dealing_High'] + df['Dealing_Low']) / 2
    df['Dealing_Pos']  = (df['Close'] - df['Dealing_Low']) / (df['Dealing_High'] - df['Dealing_Low'])
    df['In_Discount']  = (df['Dealing_Pos'] < 0.50).astype(int)
    df['In_Premium']   = (df['Dealing_Pos'] > 0.50).astype(int)
    df['Dist_To_PDH']  = (df['High'].shift(1) - df['Close']) / df['Close']
    df['Dist_To_PDL']  = (df['Close'] - df['Low'].shift(1)) / df['Close']

    print("🔧 Building structural swing/liquidity ladder...")
    LADDER_SIZE = 5
    def _build_swing_ladder(swing_flag, level_values, n_keep=LADDER_SIZE):
        n = len(swing_flag)
        ladder = []
        current = []
        for i in range(n):
            if swing_flag[i]:
                lvl = float(level_values[i])
                if not np.isnan(lvl):
                    current.append(lvl)
                    if len(current) > n_keep:
                        current.pop(0)
            ladder.append(list(current))
        return ladder
    _swing_high_raw, _swing_low_raw = _find_swings(df, 10, 10)
    df['Swing_High_Ladder'] = _build_swing_ladder(_swing_high_raw, df['High'].values)
    df['Swing_Low_Ladder'] = _build_swing_ladder(_swing_low_raw, df['Low'].values)
    print(f"✅ Swing ladder built (keep last {LADDER_SIZE} highs + {LADDER_SIZE} lows)")

    print("🔧 Computing OB/FVG strength and wick penetration metrics...")
    ZONE_WIDTH_PCT = 0.005
    DISPLACEMENT_WINDOW = 5
    def _count_ob_touches(ob_flag, ob_level, highs, lows, zone_width=ZONE_WIDTH_PCT):
        n = len(ob_flag)
        touches = np.zeros(n, dtype=float)
        last_level = np.nan
        last_idx = -1
        count = 0
        for i in range(n):
            if ob_flag[i] == 1:
                last_level = ob_level[i]
                last_idx = i
                count = 0
            if not np.isnan(last_level) and 0 <= last_idx < i:
                z_top = last_level * (1 + zone_width)
                z_bot = last_level * (1 - zone_width)
                if lows[i] <= z_top and highs[i] >= z_bot:
                    count += 1
            touches[i] = count
        return touches
    def _measure_ob_displacement(ob_flag, ob_level, closes, window=DISPLACEMENT_WINDOW):
        n = len(closes)
        disp = np.full(n, np.nan)
        last_level = np.nan
        last_idx = -1
        for i in range(n):
            if ob_flag[i] == 1:
                last_level = ob_level[i]
                last_idx = i
            if last_idx >= 0 and (i - last_idx) <= window:
                max_close = np.max(closes[last_idx:i+1])
                disp[i] = max((max_close - last_level) / last_level, 0)
        return disp
    def _ob_age(ob_flag):
        n = len(ob_flag)
        ages = np.full(n, np.nan)
        last_idx = -1
        for i in range(n):
            if ob_flag[i] == 1:
                last_idx = i
            if last_idx >= 0:
                ages[i] = i - last_idx
        return ages
    def _ob_volume_z_at_formation(ob_flag, vol_z, lag=2):
        n = len(vol_z)
        vz = np.full(n, np.nan)
        for i in range(n):
            if ob_flag[i] == 1:
                vz[i] = vol_z[max(0, i - lag)]
        return pd.Series(vz).ffill().bfill().fillna(0.0).values
    def _wick_penetration(ob_flag, body_bottom, wick_bottom, lows, window=10):
        n = len(lows)
        penetration = np.full(n, np.nan)
        last_body = np.nan
        last_wick = np.nan
        last_idx = -1
        for i in range(n):
            if ob_flag[i] == 1:
                last_body = body_bottom[i]
                last_wick = wick_bottom[i]
                last_idx = i
            if last_idx >= 0 and (i - last_idx) <= window:
                wick_range = last_body - last_wick
                if wick_range > 0:
                    deepest = last_body - np.min(lows[last_idx:i+1])
                    penetration[i] = max(0.0, min(deepest / wick_range, 1.0))
        return penetration
    _bull_ob_flag = df['Bull_OB'].values
    _bull_ob_level = df['Bull_OB_Level'].values
    _highs = df['High'].values
    _lows = df['Low'].values
    _closes = df['Close'].values
    _vol_z = df['Volume_Zscore'].fillna(0).values
    df['Bull_OB_Touches']      = _count_ob_touches(_bull_ob_flag, _bull_ob_level, _highs, _lows)
    df['Bull_OB_Displacement'] = _measure_ob_displacement(_bull_ob_flag, _bull_ob_level, _closes)
    df['Bull_OB_Age']          = _ob_age(_bull_ob_flag)
    df['Bull_OB_HighVol']      = _ob_volume_z_at_formation(_bull_ob_flag, _vol_z)
    df['Bull_OB_Wick_Penetration'] = _wick_penetration(
        _bull_ob_flag, df['Bull_OB_Body_Bottom'].fillna(0).values,
        df['Bull_OB_Wick_Bottom'].fillna(0).values, _lows)
    _bear_ob_flag = df['Bear_OB'].values
    _bear_ob_level = df['Bear_OB_Level'].values
    df['Bear_OB_Touches']      = _count_ob_touches(_bear_ob_flag, _bear_ob_level, _highs, _lows)
    df['Bear_OB_Displacement'] = _measure_ob_displacement(_bear_ob_flag, _bear_ob_level, _closes)
    df['Bear_OB_Age']          = _ob_age(_bear_ob_flag)
    df['Bear_OB_HighVol']      = _ob_volume_z_at_formation(_bear_ob_flag, _vol_z)
    df['Bear_OB_Wick_Penetration'] = _wick_penetration(
        _bear_ob_flag, -df['Bear_OB_Body_Top'].fillna(0).values,
        -df['Bear_OB_Wick_Top'].fillna(0).values, -_highs)
    def _fvg_tracking(fvg_flag, fvg_top, fvg_bottom, highs, lows, closes, vol_z):
        n = len(fvg_flag)
        fill_pct = np.full(n, np.nan)
        touches = np.zeros(n, dtype=float)
        ages = np.full(n, np.nan)
        displacement = np.full(n, np.nan)
        vol_form = np.full(n, np.nan)
        last_top = np.nan
        last_bottom = np.nan
        last_idx = -1
        last_mid_close = np.nan
        touch_count = 0
        for i in range(n):
            if fvg_flag[i] == 1:
                last_top = fvg_top[i]
                last_bottom = fvg_bottom[i]
                last_idx = i
                last_mid_close = closes[i]
                touch_count = 0
                vol_form[i] = vol_z[max(0, i - 1)]
            if 0 <= last_idx < i:
                top, bottom = last_top, last_bottom
                gap_size = top - bottom
                if gap_size > 0:
                    lo_visit = np.min(lows[last_idx:i+1])
                    hi_visit = np.max(highs[last_idx:i+1])
                    if lo_visit <= bottom and hi_visit >= top:
                        fill_pct[i] = 1.0
                    elif lo_visit > top or hi_visit < bottom:
                        fill_pct[i] = 0.0
                    elif lo_visit <= top and hi_visit >= top:
                        deepest = top - max(min(lo_visit, top), bottom)
                        fill_pct[i] = min(deepest / gap_size, 1.0)
                    else:
                        deepest = min(max(hi_visit, bottom), top) - bottom
                        fill_pct[i] = min(deepest / gap_size, 1.0)
                    if lows[i] <= top and highs[i] >= bottom:
                        touch_count += 1
                ages[i] = i - last_idx
                mid = (last_top + last_bottom) / 2
                if last_mid_close > mid:
                    max_close = np.max(closes[last_idx:i+1])
                    displacement[i] = max((max_close - mid) / mid, 0)
                else:
                    min_close = np.min(closes[last_idx:i+1])
                    displacement[i] = max((mid - min_close) / mid, 0)
            touches[i] = touch_count
        return fill_pct, touches, ages, displacement, vol_form
    _bull_fvg_track = _fvg_tracking(
        _fvg_bull_flag, df['FVG_Bull_Top'].values, df['FVG_Bull_Bottom'].values,
        _highs, _lows, _closes, _vol_z)
    df['Bull_FVG_Fill_Pct']     = _bull_fvg_track[0]
    df['Bull_FVG_Touches']      = _bull_fvg_track[1]
    df['Bull_FVG_Age']          = _bull_fvg_track[2]
    df['Bull_FVG_Displacement'] = _bull_fvg_track[3]
    df['Bull_FVG_VolAtForm']    = pd.Series(_bull_fvg_track[4]).ffill().bfill().fillna(0.0).values
    _bear_fvg_track = _fvg_tracking(
        _fvg_bear_flag, df['FVG_Bear_Top'].values, df['FVG_Bear_Bottom'].values,
        _highs, _lows, _closes, _vol_z)
    df['Bear_FVG_Fill_Pct']     = _bear_fvg_track[0]
    df['Bear_FVG_Touches']      = _bear_fvg_track[1]
    df['Bear_FVG_Age']          = _bear_fvg_track[2]
    df['Bear_FVG_Displacement'] = _bear_fvg_track[3]
    df['Bear_FVG_VolAtForm']    = pd.Series(_bear_fvg_track[4]).ffill().bfill().fillna(0.0).values
    def _safe(col):
        return col.fillna(0.0)
    df['Bull_OB_Strength'] = (-_safe(df['Bull_OB_Touches']) * 0.5
                               + _safe(df['Bull_OB_Displacement']) * 8.0
                               + _safe(df['Bull_OB_HighVol']) * 0.3
                               - _safe(df['Bull_OB_Age']) / 100.0)
    df['Bear_OB_Strength'] = (-_safe(df['Bear_OB_Touches']) * 0.5
                               + _safe(df['Bear_OB_Displacement']) * 8.0
                               + _safe(df['Bear_OB_HighVol']) * 0.3
                               - _safe(df['Bear_OB_Age']) / 100.0)
    df['Bull_FVG_Strength'] = (-_safe(df['Bull_FVG_Touches']) * 0.5
                                + _safe(df['Bull_FVG_Displacement']) * 8.0
                                - _safe(df['Bull_FVG_Fill_Pct']) * 1.5
                                - _safe(df['Bull_FVG_Age']) / 100.0)
    df['Bear_FVG_Strength'] = (-_safe(df['Bear_FVG_Touches']) * 0.5
                                + _safe(df['Bear_FVG_Displacement']) * 8.0
                                - _safe(df['Bear_FVG_Fill_Pct']) * 1.5
                                - _safe(df['Bear_FVG_Age']) / 100.0)
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"✅ Infinities replaced with NaN")
    zone_strength_fill_zero = [
        'Bull_OB_Touches', 'Bull_OB_Displacement', 'Bull_OB_Age', 'Bull_OB_HighVol',
        'Bear_OB_Touches', 'Bear_OB_Displacement', 'Bear_OB_Age', 'Bear_OB_HighVol',
        'Bull_FVG_Fill_Pct', 'Bull_FVG_Touches', 'Bull_FVG_Age',
        'Bull_FVG_Displacement', 'Bull_FVG_VolAtForm',
        'Bear_FVG_Fill_Pct', 'Bear_FVG_Touches', 'Bear_FVG_Age',
        'Bear_FVG_Displacement', 'Bear_FVG_VolAtForm',
        'Bull_OB_Strength', 'Bear_OB_Strength',
        'Bull_FVG_Strength', 'Bear_FVG_Strength',
        'Bull_OB_Wick_Penetration', 'Bear_OB_Wick_Penetration',
        'Bull_OB_Body_Ratio', 'Bear_OB_Body_Ratio',
    ]
    for col in zone_strength_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    print(f"✅ Zone strength features: NaNs filled with 0")

    classic_features = [
        'Return', 'Log_Return', 'Price_vs_MA5', 'Price_vs_MA20', 'Price_vs_MA50',
        'MA5_vs_MA20', 'MA50_vs_MA200', 'Volatility_5', 'Volatility_20', 'Volatility_50',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Position',
        'Volume_Change', 'Volume_vs_MA20', 'Body', 'Upper_Shadow', 'Lower_Shadow', 'Gap',
        'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag5', 'Return_Lag10',
        'Volume_Price_Divergence', 'Vol_ROC_5', 'Vol_ROC_10', 'Consec_Up', 'Consec_Down',
        'Overnight_Return', 'Intraday_Return', 'Overnight_vs_Intraday', 'DayOfWeek',
    ]
    ict_smc_features = [
        'BOS_Bull', 'BOS_Bear', 'Market_Structure', 'CHoCH',
        'FVG_Bull', 'FVG_Bear', 'FVG_Bull_Size', 'FVG_Bear_Size',
        'Dist_To_Last_Bull_FVG', 'Dist_To_Last_Bear_FVG',
        'Bull_IFVG', 'Bear_IFVG', 'Dist_To_Bull_IFVG', 'Dist_To_Bear_IFVG',
        'Bull_OB', 'Bear_OB', 'Dist_To_Bull_OB', 'Dist_To_Bear_OB',
        'Bull_BB', 'Bear_BB', 'Dist_To_Bull_BB', 'Dist_To_Bear_BB',
        'Equal_High', 'Equal_Low', 'Liquidity_Sweep_High', 'Liquidity_Sweep_Low',
        'Premium_Discount', 'Displacement', 'Counter_Trend_Bars',
        'Pullback_In_Uptrend_MTF', 'Bounce_In_Downtrend_MTF',
        'Dealing_Pos', 'In_Discount', 'In_Premium',
        'Dist_To_PDH', 'Dist_To_PDL',
    ]
    zone_strength_features = [
        'Bull_OB_Touches', 'Bull_OB_Displacement', 'Bull_OB_Age', 'Bull_OB_HighVol',
        'Bear_OB_Touches', 'Bear_OB_Displacement', 'Bear_OB_Age', 'Bear_OB_HighVol',
        'Bull_FVG_Fill_Pct', 'Bull_FVG_Touches', 'Bull_FVG_Age',
        'Bull_FVG_Displacement', 'Bull_FVG_VolAtForm',
        'Bear_FVG_Fill_Pct', 'Bear_FVG_Touches', 'Bear_FVG_Age',
        'Bear_FVG_Displacement', 'Bear_FVG_VolAtForm',
        'Bull_OB_Strength', 'Bear_OB_Strength',
        'Bull_FVG_Strength', 'Bear_FVG_Strength',
        'Bull_OB_Wick_Penetration', 'Bear_OB_Wick_Penetration',
        'Bull_OB_Body_Ratio', 'Bear_OB_Body_Ratio',
    ]
    feature_columns = classic_features + ict_smc_features + zone_strength_features
    if HAS_DXY:
        feature_columns = feature_columns + dxy_features

    df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Next_Date'] = df['Date'].shift(-1)
    VOL_MULT = 0.5
    MIN_THRESHOLD = 0.001
    df['Move_Threshold'] = (df['Volatility_20'] * VOL_MULT).clip(lower=MIN_THRESHOLD)
    CLASS_NAMES = {0: 'DOWN', 1: 'CONSOLIDATION', 2: 'UP'}
    CLASS_ORDER = ['DOWN', 'CONSOLIDATION', 'UP']
    _target = np.select(
        [df['Future_Return'] > df['Move_Threshold'],
         df['Future_Return'] < -df['Move_Threshold']],
        [2, 0], default=1).astype(float)
    _target[df['Future_Return'].isna()] = np.nan
    df['Target'] = _target

    print("🔧 Building historical analog pattern-matching features...")
    ANALOG_K = 25
    ANALOG_MIN_HISTORY = 500
    ANALOG_RECOMPUTE_EVERY = 20
    SIGNATURE_FEATURES = [
        'Return', 'Price_vs_MA5', 'Price_vs_MA20', 'MA5_vs_MA20',
        'MA50_vs_MA200', 'Volatility_20', 'RSI', 'BB_Position',
        'Volume_vs_MA20', 'Body', 'MACD_Hist', 'Premium_Discount',
    ]
    _sig_df = df[SIGNATURE_FEATURES].copy()
    _sig_norm = (_sig_df - _sig_df.mean()) / (_sig_df.std().replace(0, np.nan))
    _sig_matrix = _sig_norm.fillna(0).values
    n = len(df)
    df['Analog_Up_Rate'] = np.nan
    df['Analog_Down_Rate'] = np.nan
    df['Analog_Flat_Rate'] = np.nan
    df['Analog_Mean_Return'] = np.nan
    df['Analog_Std_Return'] = np.nan
    df['Analog_Avg_Distance'] = np.nan
    _target_arr = df['Target'].values
    _future_ret_arr = df['Future_Return'].values
    def _analog_stats(idx, history_end):
        if history_end < ANALOG_MIN_HISTORY:
            return (np.nan,) * 6
        sig_current = _sig_matrix[idx]
        sig_past = _sig_matrix[:history_end]
        diffs = sig_past - sig_current
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        if len(dists) <= ANALOG_K:
            nearest_idx = np.arange(len(dists))
        else:
            nearest_idx = np.argpartition(dists, ANALOG_K)[:ANALOG_K]
        tgt = _target_arr[nearest_idx]
        ret = _future_ret_arr[nearest_idx]
        valid = ~np.isnan(tgt) & ~np.isnan(ret)
        if not valid.any():
            return (np.nan,) * 6
        tgt = tgt[valid].astype(int)
        ret = ret[valid]
        d = dists[nearest_idx][valid]
        return (float((tgt == 2).mean()), float((tgt == 0).mean()),
                float((tgt == 1).mean()), float(np.mean(ret)),
                float(np.std(ret)), float(np.mean(d)))
    print(f"   Computing analogs for {n} bars (K={ANALOG_K}, min_history={ANALOG_MIN_HISTORY})...")
    analog_up = np.full(n, np.nan)
    analog_down = np.full(n, np.nan)
    analog_flat = np.full(n, np.nan)
    analog_mean = np.full(n, np.nan)
    analog_std = np.full(n, np.nan)
    analog_dst = np.full(n, np.nan)
    for chunk_start in range(0, n, ANALOG_RECOMPUTE_EVERY):
        chunk_end = min(chunk_start + ANALOG_RECOMPUTE_EVERY, n)
        history_end = chunk_start
        if history_end < ANALOG_MIN_HISTORY:
            continue
        for j in range(chunk_start, chunk_end):
            up, dn, fl, mu, sd, dst = _analog_stats(j, history_end)
            analog_up[j], analog_down[j], analog_flat[j] = up, dn, fl
            analog_mean[j], analog_std[j], analog_dst[j] = mu, sd, dst
    df['Analog_Up_Rate']     = analog_up
    df['Analog_Down_Rate']   = analog_down
    df['Analog_Flat_Rate']   = analog_flat
    df['Analog_Mean_Return'] = analog_mean
    df['Analog_Std_Return']  = analog_std
    df['Analog_Avg_Distance'] = analog_dst
    df['Analog_Up_Minus_Down'] = df['Analog_Up_Rate'] - df['Analog_Down_Rate']
    for col, fill_val in {
        'Analog_Up_Rate': 0.5, 'Analog_Down_Rate': 0.5, 'Analog_Flat_Rate': 0.0,
        'Analog_Mean_Return': 0.0, 'Analog_Std_Return': 0.0,
        'Analog_Avg_Distance': np.nan, 'Analog_Up_Minus_Down': 0.0,
    }.items():
        df[col] = df[col].fillna(fill_val)
    print(f"✅ Analog features computed")
    pattern_features = [
        'Analog_Up_Rate', 'Analog_Down_Rate', 'Analog_Flat_Rate',
        'Analog_Mean_Return', 'Analog_Std_Return', 'Analog_Up_Minus_Down',
    ]
    feature_columns = feature_columns + pattern_features

    all_nan_cols = [c for c in feature_columns if df[c].isna().all()]
    if all_nan_cols:
        print(f"\n⚠️  Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
        feature_columns = [c for c in feature_columns if c not in all_nan_cols]

    df_all = df.dropna(subset=feature_columns).reset_index(drop=True)
    df_model = df_all.dropna(subset=['Target']).reset_index(drop=True)
    df_model['Target'] = df_model['Target'].astype(int)
    df_live = df_all[df_all['Target'].isna()].reset_index(drop=True)

    print(f"\n✅ Features created ({len(feature_columns)} total)")
    print(f"   Classic: {len(classic_features)}  |  ICT/SMC: {len(ict_smc_features)}  |  "
          f"Zone Strength: {len(zone_strength_features)}  |  "
          f"DXY: {len(dxy_features) if HAS_DXY else 0}  |  Pattern: {len(pattern_features)}")
    _dist = df_model['Target'].value_counts().sort_index()
    _total = len(df_model)
    print(f"📊 Target distribution (VOL_MULT = {VOL_MULT}):")
    for cls_id, name in enumerate(CLASS_ORDER):
        n_cnt = _dist.get(cls_id, 0)
        pct = n_cnt / _total * 100 if _total else 0
        print(f"   {name:<15} ({cls_id}): {n_cnt:>5}  ({pct:>5.1f}%)")
    print(f"📊 Training rows: {len(df_model)}   |   Live rows: {len(df_live)}")
    if len(df_live):
        print(f"   ↳ Latest live row: {df_live['Date'].iloc[-1].date()}  "
              f"(close={df_live['Close'].iloc[-1]:.2f})")

    split_idx = int(len(df_model) * 0.8)
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]
    X_train = train[feature_columns]
    y_train = train['Target']
    X_test = test[feature_columns]
    y_test = test['Target']
    print(f"\n📊 Training: {len(X_train)} days "
          f"({train['Date'].min().date()} → {train['Date'].max().date()})")
    print(f"📊 Testing:  {len(X_test)} days "
          f"({test['Date'].min().date()} → {test['Date'].max().date()})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🤖 Training Random Forest (MODEL 1 — daily bias)...")
    model = RandomForestClassifier(
        n_estimators=400, max_depth=14,
        min_samples_split=50, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {accuracy:.2%}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_ORDER, labels=[0, 1, 2]))
    print("\n📊 Time-Series Cross-Validation (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
        m = clone(model)
        m.fit(X_train_scaled[tr_idx], y_train.values[tr_idx])
        fold_score = m.score(X_train_scaled[val_idx], y_train.values[val_idx])
        cv_scores.append(fold_score)
        print(f"   Fold {fold}: {fold_score:.2%}")
    print(f"   CV Mean: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")

    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(f"\n📊 Top 30 Features:")
    print(importance_df.head(30).to_string(index=False))
    n_pattern_top20 = importance_df.head(20)['Feature'].isin(pattern_features).sum()
    print(f"\n🎯 Analog pattern features in top 20: {n_pattern_top20} / {len(pattern_features)}")
    if HAS_DXY:
        n_dxy_top20 = importance_df.head(20)['Feature'].isin(dxy_features).sum()
        print(f"💵 DXY cross-asset features in top 20: {n_dxy_top20} / {len(dxy_features)}")

    journal_state = {'cals': None, 'meta': None}

    def _find_row_by_date(frame, input_date):
        input_date = pd.to_datetime(input_date)
        matches = frame[frame['Date'] == input_date]
        if len(matches) > 0:
            return matches, False
        closest_idx = (frame['Date'] - input_date).abs().idxmin()
        return frame.loc[[closest_idx]], True

    def _next_trading_date(after_date):
        future = df_all[df_all['Date'] > after_date]
        if len(future):
            return future['Date'].iloc[0]
        return after_date + pd.offsets.BDay(1)

    _DIRECTION_ICON = {'UP': '🟢 UP', 'DOWN': '🔴 DOWN', 'CONSOLIDATION': '🟡 CONSOLIDATION'}

    def _describe_strength(score):
        if pd.isna(score):
            return 'unknown'
        if score >= 0.5:
            return 'STRONG'
        if score >= 0.15:
            return 'moderate'
        if score >= -0.15:
            return 'weak'
        return 'very weak / stale'

    def _pick_entry_style(body_ratio, user_pref='auto'):
        if user_pref != 'auto':
            return user_pref
        if pd.isna(body_ratio):
            return 'moderate'
        if body_ratio > 0.7:
            return 'aggressive'
        if body_ratio < 0.4:
            return 'conservative'
        return 'moderate'

    def _get_entry_level(row, side, style):
        if side == 'LONG':
            key = f'Bull_OB_Entry_{style.capitalize()}'
        else:
            key = f'Bear_OB_Entry_{style.capitalize()}'
        return row[key].iloc[0]

    def _describe_mtf(macro, micro, pullback, bounce):
        if pullback == 1:
            return 'Bullish macro, bearish micro — PULLBACK IN UPTREND'
        if bounce == 1:
            return 'Bearish macro, bullish micro — BOUNCE IN DOWNTREND'
        if macro == 1 and micro == 1:
            return 'All timeframes bullish'
        if macro == -1 and micro == -1:
            return 'All timeframes bearish'
        return 'Mixed / unresolved'

    def _structure_stop_target(side, entry, row, atr, rr_fallback=2.0):
        buffer = max(0.1 * atr, entry * 0.0005)
        if side == 'LONG':
            stop_candidates = []
            swing_lows = row['Swing_Low_Ladder'].iloc[0] if 'Swing_Low_Ladder' in row.columns else []
            for lvl in (swing_lows or []):
                if pd.notna(lvl) and lvl < entry:
                    stop_candidates.append((lvl, f'Swing Low {lvl:.2f}'))
            wick_bottom = row['Bull_OB_Wick_Bottom'].iloc[0]
            if pd.notna(wick_bottom) and wick_bottom < entry:
                stop_candidates.append((wick_bottom, f'Bull OB wick {wick_bottom:.2f}'))
            fvg_bottom = row['FVG_Bull_Bottom'].iloc[0]
            if pd.notna(fvg_bottom) and fvg_bottom < entry:
                stop_candidates.append((fvg_bottom, f'Bull FVG bottom {fvg_bottom:.2f}'))
            atr_stop = entry - 1.5 * atr
            stop_candidates.append((atr_stop, f'ATR 1.5x {atr_stop:.2f}'))
            min_dist = 0.15 * atr
            viable = [(lvl, src) for lvl, src in stop_candidates if (entry - lvl) >= min_dist]
            if viable:
                stop_lvl, stop_src = max(viable, key=lambda x: x[0])
            else:
                stop_lvl, stop_src = atr_stop, 'ATR fallback'
            stop_loss = stop_lvl - buffer
            target_candidates = []
            swing_highs = row['Swing_High_Ladder'].iloc[0] if 'Swing_High_Ladder' in row.columns else []
            for lvl in (swing_highs or []):
                if pd.notna(lvl) and lvl > entry:
                    target_candidates.append((lvl, f'Swing High {lvl:.2f}'))
            bear_ob = row['Bear_OB_Body_Bottom'].iloc[0]
            if pd.notna(bear_ob) and bear_ob > entry:
                target_candidates.append((bear_ob, f'Bear OB {bear_ob:.2f}'))
            bear_wick = row['Bear_OB_Wick_Top'].iloc[0]
            if pd.notna(bear_wick) and bear_wick > entry:
                target_candidates.append((bear_wick, f'Bear OB wick {bear_wick:.2f}'))
            fvg_top = row['FVG_Bull_Top'].iloc[0]
            if pd.notna(fvg_top) and fvg_top > entry:
                target_candidates.append((fvg_top, f'Bull FVG top {fvg_top:.2f}'))
            rr_target = entry + rr_fallback * (entry - stop_loss)
            target_candidates.append((rr_target, f'RR {rr_fallback:.1f} fallback'))
            min_rr = 1.0
            min_target = entry + min_rr * (entry - stop_loss)
            viable_t = [(lvl, src) for lvl, src in target_candidates if lvl >= min_target]
            if viable_t:
                target_lvl, target_src = min(viable_t, key=lambda x: x[0])
            else:
                target_lvl, target_src = rr_target, 'RR fallback (no valid structure)'
            take_profit = target_lvl + buffer * 0.5
        else:
            stop_candidates = []
            swing_highs = row['Swing_High_Ladder'].iloc[0] if 'Swing_High_Ladder' in row.columns else []
            for lvl in (swing_highs or []):
                if pd.notna(lvl) and lvl > entry:
                    stop_candidates.append((lvl, f'Swing High {lvl:.2f}'))
            wick_top = row['Bear_OB_Wick_Top'].iloc[0]
            if pd.notna(wick_top) and wick_top > entry:
                stop_candidates.append((wick_top, f'Bear OB wick {wick_top:.2f}'))
            fvg_top = row['FVG_Bear_Top'].iloc[0]
            if pd.notna(fvg_top) and fvg_top > entry:
                stop_candidates.append((fvg_top, f'Bear FVG top {fvg_top:.2f}'))
            atr_stop = entry + 1.5 * atr
            stop_candidates.append((atr_stop, f'ATR 1.5x {atr_stop:.2f}'))
            min_dist = 0.15 * atr
            viable = [(lvl, src) for lvl, src in stop_candidates if (lvl - entry) >= min_dist]
            if viable:
                stop_lvl, stop_src = min(viable, key=lambda x: x[0])
            else:
                stop_lvl, stop_src = atr_stop, 'ATR fallback'
            stop_loss = stop_lvl + buffer
            target_candidates = []
            swing_lows = row['Swing_Low_Ladder'].iloc[0] if 'Swing_Low_Ladder' in row.columns else []
            for lvl in (swing_lows or []):
                if pd.notna(lvl) and lvl < entry:
                    target_candidates.append((lvl, f'Swing Low {lvl:.2f}'))
            bull_ob = row['Bull_OB_Body_Top'].iloc[0]
            if pd.notna(bull_ob) and bull_ob < entry:
                target_candidates.append((bull_ob, f'Bull OB {bull_ob:.2f}'))
            bull_wick = row['Bull_OB_Wick_Bottom'].iloc[0]
            if pd.notna(bull_wick) and bull_wick < entry:
                target_candidates.append((bull_wick, f'Bull OB wick {bull_wick:.2f}'))
            fvg_bottom = row['FVG_Bear_Bottom'].iloc[0]
            if pd.notna(fvg_bottom) and fvg_bottom < entry:
                target_candidates.append((fvg_bottom, f'Bear FVG bottom {fvg_bottom:.2f}'))
            rr_target = entry - rr_fallback * (stop_loss - entry)
            target_candidates.append((rr_target, f'RR {rr_fallback:.1f} fallback'))
            min_rr = 1.0
            min_target = entry - min_rr * (stop_loss - entry)
            viable_t = [(lvl, src) for lvl, src in target_candidates if lvl <= min_target]
            if viable_t:
                target_lvl, target_src = max(viable_t, key=lambda x: x[0])
            else:
                target_lvl, target_src = rr_target, 'RR fallback (no valid structure)'
            take_profit = target_lvl - buffer * 0.5
        return stop_loss, take_profit, stop_src, target_src

    def _dol_target(side, entry, stop_loss, row):
        candidates = []
        close = float(row['Close'].iloc[0])
        pdh = row['High'].shift(1).iloc[0] if len(row) else np.nan
        pdl = row['Low'].shift(1).iloc[0] if len(row) else np.nan

        if side == 'LONG':
            swing_highs = row['Swing_High_Ladder'].iloc[0] if 'Swing_High_Ladder' in row.columns else []
            for lvl in (swing_highs or []):
                if pd.notna(lvl) and lvl > entry:
                    candidates.append((float(lvl), f'Swing High {lvl:.2f}', 3))
            if pd.notna(pdh) and pdh > entry:
                candidates.append((float(pdh), f'PDH {pdh:.2f}', 2))
            equal = row['Equal_High'].iloc[0] if 'Equal_High' in row.columns else 0
            if equal == 1 and pdh > entry:
                candidates.append((float(pdh), f'EQH ~{pdh:.2f}', 1))
        else:
            swing_lows = row['Swing_Low_Ladder'].iloc[0] if 'Swing_Low_Ladder' in row.columns else []
            for lvl in (swing_lows or []):
                if pd.notna(lvl) and lvl < entry:
                    candidates.append((float(lvl), f'Swing Low {lvl:.2f}', 3))
            if pd.notna(pdl) and pdl < entry:
                candidates.append((float(pdl), f'PDL {pdl:.2f}', 2))
            equal = row['Equal_Low'].iloc[0] if 'Equal_Low' in row.columns else 0
            if equal == 1 and pdl < entry:
                candidates.append((float(pdl), f'EQL ~{pdl:.2f}', 1))

        if not candidates:
            return np.nan, '', np.nan

        risk = abs(entry - stop_loss)
        if risk <= 0:
            return np.nan, '', np.nan
        min_rr = 1.0
        def rr_of(lvl):
            return abs(lvl - entry) / risk
        viable = [c for c in candidates if rr_of(c[0]) >= min_rr]
        if not viable:
            return np.nan, '', np.nan
        viable.sort(key=lambda c: (c[2], abs(c[0] - entry)))
        lvl, src, rank = viable[0]
        return lvl, src, rank

    def _bias_gate(row, probs, action):
        dominant = max(probs, key=probs.get)
        if dominant == 'UP':
            side = 'LONG'
        elif dominant == 'DOWN':
            side = 'SHORT'
        else:
            side = 'LONG' if probs['UP'] >= probs['DOWN'] else 'SHORT'
        return True, side

    def recommend_trade_entry(row, probs, rr_target=2.0, atr_stop_mult=1.5,
                              entry_style='auto',
                              max_entry_dist_pct=0.010,
                              max_ob_age=60,
                              min_strength=-0.20):
        close = float(row['Close'].iloc[0])
        atr = row['ATR_14'].iloc[0]
        zone_pos = row['Premium_Discount'].iloc[0]
        bull_ob_strength = row['Bull_OB_Strength'].iloc[0]
        bear_ob_strength = row['Bear_OB_Strength'].iloc[0]
        bull_fvg_strength = row['Bull_FVG_Strength'].iloc[0]
        bear_fvg_strength = row['Bear_FVG_Strength'].iloc[0]
        bull_ob_age = row['Bull_OB_Age'].iloc[0]
        bear_ob_age = row['Bear_OB_Age'].iloc[0]
        bull_fvg_age = row['Bull_FVG_Age'].iloc[0]
        bear_fvg_age = row['Bear_FVG_Age'].iloc[0]
        bull_ob_body_ratio = row['Bull_OB_Body_Ratio'].iloc[0]
        bear_ob_body_ratio = row['Bear_OB_Body_Ratio'].iloc[0]
        if pd.isna(atr):
            atr = 0.01 * close
        if pd.isna(zone_pos):
            zone_pos = 0.5
        max_entry_dist = close * max_entry_dist_pct
        dominant = max(probs, key=probs.get)
        confidence = probs[dominant]
        if confidence >= 0.55:
            conviction = 'High'
        elif confidence >= 0.38:
            conviction = 'Medium'
        else:
            conviction = 'Low'

        def _zone_ok(level, strength, age, side, max_age):
            if pd.isna(level):
                return False
            if side == 'LONG':
                if not (level < close) or (close - level) > max_entry_dist:
                    return False
            else:
                if not (level > close) or (level - close) > max_entry_dist:
                    return False
            if pd.notna(age) and age > max_age:
                return False
            if pd.notna(strength) and strength < min_strength:
                return False
            return True

        gate_ok, side = _bias_gate(row, probs, 'LONG' if dominant == 'UP' else 'SHORT')
        rejected = []

        if side == 'LONG':
            ob_style = _pick_entry_style(bull_ob_body_ratio, entry_style)
            ob_entry_level = _get_entry_level(row, 'LONG', ob_style)
            bull_fvg_mid = row['FVG_Bull_Mid'].iloc[0] if 'FVG_Bull_Mid' in row.columns else np.nan
            candidates = []
            if _zone_ok(ob_entry_level, bull_ob_strength, bull_ob_age, 'LONG', max_ob_age):
                candidates.append((ob_entry_level, f'Bull OB ({ob_style} wick)', bull_ob_strength))
            elif pd.notna(ob_entry_level) and ob_entry_level < close:
                rejected.append(f"Bull OB rejected (style={ob_style})")
            if _zone_ok(bull_fvg_mid, bull_fvg_strength, bull_fvg_age, 'LONG', max_ob_age):
                candidates.append((bull_fvg_mid, 'Bull FVG (mid)', bull_fvg_strength))
            if candidates:
                entry, zone_kind, zone_strength = max(candidates, key=lambda x: x[2] if pd.notna(x[2]) else -np.inf)
                entry_type = f'limit (into {zone_kind}, strength: {_describe_strength(zone_strength)})'
            else:
                entry = close
                entry_type = 'market at close (no qualifying zone)'
                zone_strength = np.nan
        else:
            ob_style = _pick_entry_style(bear_ob_body_ratio, entry_style)
            ob_entry_level = _get_entry_level(row, 'SHORT', ob_style)
            bear_fvg_mid = row['FVG_Bear_Mid'].iloc[0] if 'FVG_Bear_Mid' in row.columns else np.nan
            candidates = []
            if _zone_ok(ob_entry_level, bear_ob_strength, bear_ob_age, 'SHORT', max_ob_age):
                candidates.append((ob_entry_level, f'Bear OB ({ob_style} wick)', bear_ob_strength))
            elif pd.notna(ob_entry_level) and ob_entry_level > close:
                rejected.append(f"Bear OB rejected (style={ob_style})")
            if _zone_ok(bear_fvg_mid, bear_fvg_strength, bear_fvg_age, 'SHORT', max_ob_age):
                candidates.append((bear_fvg_mid, 'Bear FVG (mid)', bear_fvg_strength))
            if candidates:
                entry, zone_kind, zone_strength = max(candidates, key=lambda x: x[2] if pd.notna(x[2]) else -np.inf)
                entry_type = f'limit (into {zone_kind}, strength: {_describe_strength(zone_strength)})'
            else:
                entry = close
                entry_type = 'market at close (no qualifying zone)'
                zone_strength = np.nan

        stop_loss, take_profit, stop_src, target_src = _structure_stop_target(
            side, entry, row, atr, rr_fallback=rr_target)
        daily_entry = entry
        daily_entry_type = entry_type
        daily_risk = abs(entry - stop_loss)
        daily_reward = abs(take_profit - entry)
        daily_rr = round(daily_reward / daily_risk, 2) if daily_risk > 0 else None

        dol_lvl, dol_src, dol_rank = _dol_target(side, entry, stop_loss, row)
        if not pd.isna(dol_lvl):
            current_rr = abs(take_profit - entry) / daily_risk if daily_risk > 0 else 0
            dol_rr = abs(dol_lvl - entry) / daily_risk if daily_risk > 0 else 0
            if dol_rr >= max(1.0, current_rr * 0.9):
                take_profit = dol_lvl
                target_src = f"{dol_src} (DOL rank {dol_rank})"
                daily_reward = abs(take_profit - entry)
                daily_rr = round(daily_reward / daily_risk, 2) if daily_risk > 0 else None
        else:
            dol_lvl, dol_src, dol_rank = np.nan, '', np.nan

        # -------- MODEL 2: hourly entry (learned zone reactions + OTE) --------
        hourly_meta = None
        if HOURLY_OPT_ENABLED and HAS_HOURLY and side in ('LONG', 'SHORT'):
            try:
                struct = best_hourly_entry(hourly_df, side, close, index_name=index_name,
                                           daily_target=take_profit)
            except Exception as e:
                print(f"   ⚠️  Hourly optimizer failed: {e}")
                struct = None

            if struct is not None:
                learned = bool(struct.get("learned", False))
                hourly_entry = float(struct["entry_price"])
                hourly_stop_candidate = float(struct["stop_reference"])
                if learned:
                    use_hourly_stop = True
                    stop_for_hourly = hourly_stop_candidate
                    tp_for_hourly = (float(struct["tp_reference"])
                                     if struct.get("tp_reference") is not None else take_profit)
                else:
                    if side == 'LONG':
                        use_hourly_stop = hourly_stop_candidate > stop_loss
                    else:
                        use_hourly_stop = hourly_stop_candidate < stop_loss
                    stop_for_hourly = hourly_stop_candidate if use_hourly_stop else stop_loss
                    tp_for_hourly = take_profit

                hourly_risk = abs(hourly_entry - stop_for_hourly)
                hourly_reward = abs(tp_for_hourly - hourly_entry)
                hourly_rr = round(hourly_reward / hourly_risk, 2) if hourly_risk > 0 else None

                applied = False
                if learned:
                    ok = (not struct.get("blocked", False)
                          and hourly_rr is not None and hourly_rr >= zl.CFG["MIN_RR"]
                          and struct["react_prob"] >= struct["react_base"])
                else:
                    ok = bool(hourly_rr is not None and daily_rr and hourly_rr >= daily_rr * 0.95)

                if ok:
                    entry = hourly_entry
                    stop_loss = stop_for_hourly
                    entry_type = (f"limit (hourly {struct['kind'].replace('_', ' ')} "
                                  f"{struct['session']} {struct['hour_utc']:02d}:00 UTC)")
                    if learned:
                        take_profit = tp_for_hourly
                        stop_src = f"hourly {struct['kind']} zone — {struct['stop_note']}"
                        if struct.get("tp_note"):
                            target_src = struct["tp_note"]
                    elif use_hourly_stop:
                        stop_src = f"hourly {struct['kind']} zone edge (tighter)"
                    applied = True

                hourly_meta = {
                    **struct,
                    "applied": applied,
                    "hourly_rr": hourly_rr,
                    "daily_entry": round(daily_entry, 2),
                    "daily_entry_type": daily_entry_type,
                    "daily_rr": daily_rr,
                }

        dominant_final = max(probs, key=probs.get)
        expected_side = {'UP': 'LONG', 'DOWN': 'SHORT'}.get(dominant_final)
        if expected_side is not None and side != expected_side:
            side = expected_side
            entry = close
            stop_loss, take_profit, stop_src, target_src = _structure_stop_target(
                side, entry, row, atr, rr_fallback=rr_target)
            entry_type = 'market at close (daily-bias alignment fallback)'
            hourly_meta = None
            dol_lvl2, dol_src2, dol_rank2 = _dol_target(side, entry, stop_loss, row)
            if not pd.isna(dol_lvl2):
                take_profit = dol_lvl2
                target_src = f"{dol_src2} (DOL rank {dol_rank2})"
                dol_lvl, dol_src, dol_rank = dol_lvl2, dol_src2, dol_rank2

        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        actual_rr = round(reward / risk, 2) if risk > 0 else None
        reason = f"{side} setup — {dominant} at {confidence:.1%} ({conviction}), {entry_type}"
        if rejected:
            reason += " | " + " | ".join(rejected[:2])

        return {
            'action': side,
            'entry_type': entry_type,
            'entry_style': ob_style,
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'stop_source': stop_src,
            'take_profit': round(take_profit, 2),
            'target_source': target_src,
            'risk_reward': actual_rr,
            'confidence': confidence,
            'conviction': conviction,
            'zone': 'Discount' if zone_pos < 0.5 else 'Premium',
            'zone_strength': zone_strength,
            'reason': reason,
            'hourly_meta': hourly_meta,
            'daily_entry': round(daily_entry, 2),
            'daily_rr': daily_rr,
            'bias_gate_passed': True,
            'dol_target': round(dol_lvl, 2) if pd.notna(dol_lvl) else np.nan,
            'dol_source': dol_src,
            'dol_rank': dol_rank,
        }

    def adapt_trade_with_journal(baseline, row, probs, next_date):
        info = {'params': None, 'meta_p': None, 'meta_exp_r': None, 'filtered': False}
        rec = _journal_record(row, probs, baseline, next_date, 'live')
        info['record'] = rec
        if baseline['action'] == 'NO TRADE' or not USE_JOURNAL:
            return baseline, info
        trade = dict(baseline)
        close = float(row['Close'].iloc[0])
        atr = rec['atr']
        params = _pick_calibration(journal_state['cals'], baseline['action'])
        if params is not None:
            trade = _apply_calibration(baseline, close, atr, params)
            info['params'] = params
        meta = journal_state['meta']
        if meta is not None and meta.get('accepted'):
            p = _meta_predict(meta, rec)
            rr = baseline.get('risk_reward') or 1.0
            exp_r = p * rr - (1 - p)
            info['meta_p'] = p
            info['meta_exp_r'] = exp_r
            if exp_r < META_MIN_EXPECTED_R:
                info['filtered'] = True
                trade = {
                    'action': 'NO TRADE',
                    'reason': (f"Journal filter: similar past setups won {p:.1%} of the time "
                               f"at R:R {rr} -> expected {exp_r:+.2f}R per trade."),
                }
        return trade, info

    def predict_for_date(input_date):
        input_date = pd.to_datetime(input_date)
        use_live = len(df_live) and (df_live['Date'] == input_date).any()
        source = df_live if use_live else df_model
        row, was_fuzzy = _find_row_by_date(source, input_date)
        if was_fuzzy:
            print(f"⚠️  Date {input_date.date()} not found. Using closest: {row['Date'].iloc[0].date()}")
        X_input = row[feature_columns]
        X_input_scaled = scaler.transform(X_input)
        prediction = int(model.predict(X_input_scaled)[0])
        probability = model.predict_proba(X_input_scaled)[0]
        probs = {CLASS_NAMES[int(c)]: float(p) for c, p in zip(model.classes_, probability)}
        predicted_state = CLASS_NAMES[prediction]
        confidence = probs[predicted_state]
        input_date_actual = row['Date'].iloc[0]
        close_t = row['Close'].iloc[0]
        next_date = _next_trading_date(input_date_actual)
        macro_s = int(row['Macro_Structure'].iloc[0])
        micro_s = int(row['Micro_Structure'].iloc[0])
        pullback_flag = int(row['Pullback_In_Uptrend_MTF'].iloc[0])
        bounce_flag = int(row['Bounce_In_Downtrend_MTF'].iloc[0])
        counter_bars = int(row['Counter_Trend_Bars'].iloc[0])
        structure_label = {1: 'Bullish', -1: 'Bearish', 0: 'Unresolved'}
        mtf_summary = _describe_mtf(macro_s, micro_s, pullback_flag, bounce_flag)
        zone = 'Premium' if row['Premium_Discount'].iloc[0] > 0.5 else 'Discount'
        dealing_pos = row['Dealing_Pos'].iloc[0] if 'Dealing_Pos' in row.columns else np.nan
        analog_up   = float(row['Analog_Up_Rate'].iloc[0])
        analog_dn   = float(row['Analog_Down_Rate'].iloc[0])
        analog_fl   = float(row['Analog_Flat_Rate'].iloc[0])
        analog_mean = float(row['Analog_Mean_Return'].iloc[0])
        analog_std  = float(row['Analog_Std_Return'].iloc[0])

        print("\n" + "="*60)
        print(f"🔮 {index_name} PREDICTION")
        print("="*60)
        print(f"📅 Data date (T):        {input_date_actual.date()}")
        print(f"💰 Close price (T):      {close_t:.2f}")
        print(f"📅 Predicting move for:  {next_date.date()}")
        print(f"\n🎯 Predicted state:      {_DIRECTION_ICON[predicted_state]}")
        print(f"📊 Confidence:           {confidence:.2%}")
        print(f"\n📈 Probabilities:")
        print(f"   UP:            {probs['UP']:.2%}")
        print(f"   DOWN:          {probs['DOWN']:.2%}")
        print(f"   CONSOLIDATION: {probs['CONSOLIDATION']:.2%}")
        print(f"\n🎯 HISTORICAL ANALOGS (K=25 nearest past setups)")
        print(f"   Analog UP rate:      {analog_up:.1%}")
        print(f"   Analog DOWN rate:    {analog_dn:.1%}")
        print(f"   Analog FLAT rate:    {analog_fl:.1%}")
        print(f"   Analog mean return:  {analog_mean*100:+.2f}%")
        print(f"   Analog std return:   {analog_std*100:.2f}%")
        if HAS_DXY:
            dxy_ret   = float(row['DXY_Return'].iloc[0])
            dxy_rs    = float(row['DXY_RSI'].iloc[0])
            dxy_vsma  = float(row['DXY_vs_MA20'].iloc[0])
            dxy_corr  = float(row['Corr_20d_DXY'].iloc[0])
            dxy_rs_x  = float(row['Relative_Strength_vs_DXY'].iloc[0])
            print(f"\n💵 DXY CROSS-ASSET CONTEXT")
            print(f"   DXY return:          {dxy_ret*100:+.3f}%")
            print(f"   DXY RSI:             {dxy_rs:.1f}")
            print(f"   DXY vs 20d MA:       {dxy_vsma*100:+.3f}%")
            print(f"   20d corr (mkt,DXY):  {dxy_corr:+.3f}")
            print(f"   Rel strength vs DXY: {dxy_rs_x*100:+.3f}%")
        print(f"\n🧭 MULTI-TIMEFRAME STRUCTURE")
        print(f"   Macro (10-bar):  {structure_label[macro_s]}")
        print(f"   Micro (2-bar):   {structure_label[micro_s]}")
        print(f"   Counter-trend bars: {counter_bars}")
        print(f"   Summary:         {mtf_summary}")
        print(f"🧭 Zone:                 {zone} ({row['Premium_Discount'].iloc[0]:.2f})")
        if pd.notna(dealing_pos):
            print(f"🧭 Dealing range:        pos={dealing_pos:.2f} "
                  f"({'premium' if dealing_pos > 0.5 else 'discount'})")
        signals = []
        if row['Bull_BB'].iloc[0] == 1: signals.append('Bull Breaker')
        if row['Bear_BB'].iloc[0] == 1: signals.append('Bear Breaker')
        if row['Bull_IFVG'].iloc[0] == 1: signals.append('Bull IFVG')
        if row['Bear_IFVG'].iloc[0] == 1: signals.append('Bear IFVG')
        if row['BOS_Bull'].iloc[0] == 1: signals.append('BOS Bull')
        if row['BOS_Bear'].iloc[0] == 1: signals.append('BOS Bear')
        if row['CHoCH'].iloc[0] == 1: signals.append('CHoCH')
        if row['Liquidity_Sweep_High'].iloc[0] == 1: signals.append('Sweep High')
        if row['Liquidity_Sweep_Low'].iloc[0] == 1: signals.append('Sweep Low')
        print(f"\n🔔 Active SMC signals:   {', '.join(signals) if signals else 'None'}")

        baseline = recommend_trade_entry(row, probs)
        trade, adapt_info = adapt_trade_with_journal(baseline, row, probs, next_date)

        print(f"\n💼 {index_name} TRADE ENTRY RECOMMENDATION  (research tool, not financial advice)")
        print("-"*60)
        if trade['action'] == 'NO TRADE':
            print(f"   Action: NO TRADE")
            print(f"   Reason: {trade['reason']}")
        else:
            hm = trade.get('hourly_meta')
            print(f"   DAILY BIAS (Model 1 — 46 years training):")
            print(f"     Action:       {trade['action']}")
            print(f"     Conviction:   {trade['conviction']}  ({trade['confidence']:.1%})")
            print(f"     Zone:         {trade['zone']}")
            print()
            print(f"   DAILY STRUCTURAL ENTRY:")
            print(f"     Entry type:   {hm['daily_entry_type'] if hm else trade['entry_type']}")
            print(f"     Entry:        {(hm['daily_entry'] if hm else trade['entry']):.2f}")
            print(f"     Stop loss:    {trade['stop_loss']:.2f}  ({trade['stop_source']})")
            print(f"     Take profit:  {trade['take_profit']:.2f}  ({trade['target_source']})")
            print(f"     R:R:          {(hm['daily_rr'] if hm else trade['risk_reward'])}")
            if 'dol_source' in trade and trade['dol_source']:
                print(f"     DOL target:   {trade['dol_target']:.2f}  "
                      f"({trade['dol_source']}, rank {trade['dol_rank']})")
            print()
            if hm is not None:
                print(f"   HOURLY STRUCTURAL ENTRY (Model 2 — learned zone reactions):")
                print(f"     Zone type:    {hm['kind'].replace('_', ' ').upper()}")
                print(f"     Session:      {hm['session']}")
                print(f"     Origin time:  {hm['hour_utc']:02d}:00 UTC")
                print(f"     Bars since:   {hm['bars_since_zone']} hourly bars")
                print(f"     Entry price:  {hm['entry_price']:.2f}  (limit)")
                print(f"     Zone range:   {hm['zone_bottom']:.2f} – {hm['zone_top']:.2f}")
                print(f"     Dist from px: {hm['distance_pct']:.2f}%")
                print(f"     Stop ref:     {hm['stop_reference']:.2f}")
                print(f"     Hourly R:R:   {hm['hourly_rr']}")
                ote_suffix = ""
                if hm.get('ote_aligned') and hm.get('ote_705') is not None:
                    ote_suffix = f" (0.705 = {hm['ote_705']:.2f})"
                print(f"     OTE aligned:  {'YES' if hm.get('ote_aligned') else 'NO'}{ote_suffix}")
                if hm.get('kill_zone'):
                    print(f"     Kill zone:    {hm['kill_zone']}")
                print(f"     Status:       {'APPLIED' if hm['applied'] else 'INFO only (worse R:R than daily)'}")
                print(f"     Reason:       {hm['reason']}")
            else:
                print(f"   HOURLY STRUCTURAL ENTRY (Model 2):")
                print(f"     ⚠️  No qualifying zone — using daily structural entry")
            print()
            print(f"   FINAL RECOMMENDATION:")
            print(f"     Entry:        {trade['entry']:.2f}  ({trade['entry_type']})")
            print(f"     Stop loss:    {trade['stop_loss']:.2f}")
            print(f"     Take profit:  {trade['take_profit']:.2f}")
            print(f"     R:R:          {trade['risk_reward']}")

        if USE_JOURNAL and baseline['action'] != 'NO TRADE':
            print(f"\n📓 JOURNAL ADJUSTMENTS")
            if adapt_info['params'] is not None:
                s_, e_, t_ = adapt_info['params']
                print(f"   Applied: entry {s_:.2f} ATR deeper | stop +{e_:.2f} ATR wider | "
                      f"target x{t_:.2f}")
            else:
                print(f"   No adjustment accepted yet (not enough resolved trades, or none beat "
                      f"the baseline on validation) — structure levels used as-is")
            if adapt_info['meta_p'] is not None:
                print(f"   Journal win-probability: {adapt_info['meta_p']:.1%}  "
                      f"(expected {adapt_info['meta_exp_r']:+.2f}R at baseline R:R)"
                      f"{'  -> FILTERED OUT' if adapt_info['filtered'] else ''}")
            else:
                print(f"   Journal trade-quality filter: not active yet")
        print("="*60)

        return predicted_state, probs, row, trade, baseline, adapt_info

    # ---------------- 7. GRAPH FUNCTIONS ----------------
    def save_price_with_prediction(row, probs, filename="prediction_price.png"):
        input_date = row['Date'].iloc[0]
        idx_matches = df_all.index[df_all['Date'] == input_date]
        idx = int(idx_matches[0]) if len(idx_matches) else len(df_all) - 1
        lookback = 120
        start = max(0, idx - lookback)
        window = df_all.iloc[start:idx + 1]
        last_date = row['Date'].iloc[0]
        last_close = row['Close'].iloc[0]
        next_date = _next_trading_date(last_date)
        pred_class = max(probs, key=probs.get)
        confidence = probs[pred_class]
        color = {'UP': 'green', 'DOWN': 'red', 'CONSOLIDATION': 'goldenrod'}[pred_class]
        marker = {'UP': '^', 'DOWN': 'v', 'CONSOLIDATION': 'D'}[pred_class]
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(window['Date'], window['Close'],
                color='steelblue', linewidth=1.8, label='Close Price', zorder=2)
        ax.axhline(last_close, color=color, linestyle='--',
                   linewidth=1.6, alpha=0.85, zorder=1,
                   label=f"Latest close (T): {last_close:,.2f}")
        ax.axvline(last_date, color='gray', linestyle='--', alpha=0.5, zorder=1)
        x_min = window['Date'].min()
        x_max = window['Date'].max()
        span_days = (x_max - x_min).days if (x_max - x_min).days > 0 else 1
        right_pad = pd.Timedelta(days=int(span_days * 0.22))
        ax.set_xlim(x_min, x_max + right_pad)
        label_x  = x_max + pd.Timedelta(days=int(span_days * 0.02))
        marker_x = x_max + pd.Timedelta(days=int(span_days * 0.15))
        ax.annotate(f"{last_close:,.2f}", xy=(label_x, last_close),
                    xytext=(0, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=11, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color, alpha=0.95),
                    annotation_clip=False, zorder=10)
        ax.scatter([marker_x], [last_close], color=color, s=280, marker=marker,
                   zorder=6, edgecolor='black', linewidth=1.4,
                   label=f"Predicted {pred_class} ({confidence:.1%})", clip_on=False)
        ax.set_title(f"{index_name} — {pred_class} predicted for {next_date.date()}  |  "
                     f"Confidence {confidence:.1%}",
                     fontsize=13, fontweight='bold')
        ax.set_ylabel("Price")
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = _safe_savefig(fig, INDEX_GRAPH_DIR, filename)
        plt.close(fig)
        print(f"💾 Saved: {path}")

    def save_test_predictions_chart():
        test_dates = test['Date'].values
        test_close = test['Close'].values
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(test_dates, test_close, color='black', linewidth=1.4, label='Close Price')
        correct_mask = (y_pred == y_test.values)
        ax.scatter(test_dates[correct_mask], test_close[correct_mask],
                   color='green', s=22, alpha=0.7, label=f'Correct ({correct_mask.sum()})')
        ax.scatter(test_dates[~correct_mask], test_close[~correct_mask],
                   color='red', s=22, alpha=0.7, label=f'Wrong ({(~correct_mask).sum()})')
        ax.set_title(f"{index_name} — Test Set Predictions  |  Accuracy: {accuracy:.2%}",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Date"); ax.set_ylabel("Price")
        ax.legend(loc='best'); ax.grid(alpha=0.3)
        plt.tight_layout()
        path = _safe_savefig(fig, INDEX_GRAPH_DIR, "test_predictions.png")
        plt.close(fig)
        print(f"💾 Saved: {path}")

    def save_confusion_matrix():
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        n_cls = len(CLASS_ORDER)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
        ax.set_xticklabels(CLASS_ORDER); ax.set_yticklabels(CLASS_ORDER)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f'{index_name} — Confusion Matrix (3-Class)', fontweight='bold')
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontsize=13)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        path = _safe_savefig(fig, INDEX_GRAPH_DIR, "confusion_matrix.png")
        plt.close(fig)
        print(f"💾 Saved: {path}")

    def save_feature_importance():
        top = importance_df.head(30).iloc[::-1]
        def color_for(f):
            if f in pattern_features:
                return '#f39c12'
            if f in dxy_features:
                return '#16a085'
            if f in ['Counter_Trend_Bars', 'Pullback_In_Uptrend_MTF', 'Bounce_In_Downtrend_MTF']:
                return '#e74c3c'
            if f in zone_strength_features:
                return '#8e44ad'
            if f in ict_smc_features:
                return 'darkorange'
            return 'teal'
        colors = [color_for(f) for f in top['Feature']]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh(top['Feature'], top['Importance'], color=colors)
        ax.set_xlabel('Importance')
        ax.set_title(f'{index_name} — Top 30 Feature Importances\n'
                     '(amber = analog pattern, green = DXY, red = MTF, '
                     'purple = zone strength, orange = ICT/SMC, teal = classic)',
                     fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        path = _safe_savefig(fig, INDEX_GRAPH_DIR, "feature_importance.png")
        plt.close(fig)
        print(f"💾 Saved: {path}")

    def save_probability_over_time():
        dates = test['Date'].values
        p_down, p_cons, p_up = y_proba[:, 0], y_proba[:, 1], y_proba[:, 2]
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.stackplot(dates, p_down, p_cons, p_up,
                     colors=['#d62728', '#bbbbbb', '#2ca02c'],
                     labels=['P(DOWN)', 'P(CONSOLIDATION)', 'P(UP)'],
                     alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title(f'{index_name} — Model State Probabilities over Test Period',
                     fontweight='bold')
        ax.set_xlabel('Date'); ax.set_ylabel('Probability')
        ax.legend(loc='upper left', ncol=3, fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = _safe_savefig(fig, INDEX_GRAPH_DIR, "state_probabilities.png")
        plt.close(fig)
        print(f"💾 Saved: {path}")

    def save_market_structure_chart():
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(df_all['Date'], df_all['Close'], color='black',
                linewidth=1.0, label='Close', zorder=2, alpha=0.6)
        macro_bull = df_all[df_all['Macro_BOS_Bull'] == 1]
        macro_bear = df_all[df_all['Macro_BOS_Bear'] == 1]
        ax.scatter(macro_bull['Date'], macro_bull['Close'], color='darkgreen',
                   marker='^', s=70, alpha=0.5, label='Macro BOS Bull', zorder=3)
        ax.scatter(macro_bear['Date'], macro_bear['Close'], color='darkred',
                   marker='v', s=70, alpha=0.5, label='Macro BOS Bear', zorder=3)
        micro_choch = df_all[df_all['Micro_CHoCH'] == 1]
        ax.scatter(micro_choch['Date'], micro_choch['Close'], color='gold',
                   marker='o', s=12, alpha=0.6, label='Micro CHoCH', zorder=5)
        bull_bb = df_all[df_all['Bull_BB'] == 1]
        bear_bb = df_all[df_all['Bear_BB'] == 1]
        ax.scatter(bull_bb['Date'], bull_bb['Close'], color='cyan',
                   marker='o', s=20, alpha=0.5, label='Bull Breaker', zorder=5)
        ax.scatter(bear_bb['Date'], bear_bb['Close'], color='orange',
                   marker='o', s=20, alpha=0.5, label='Bear Breaker', zorder=5)
        ax.set_title(f'{index_name} — Multi-Timeframe Market Structure\n'
                     '(Macro=big triangles, Micro=small dots)',
                     fontweight='bold')
        ax.set_xlabel('Date'); ax.set_ylabel('Price')
        ax.legend(loc='best', fontsize=8, ncol=2); ax.grid(alpha=0.3)
        plt.tight_layout()
        path = _safe_savefig(fig, INDEX_GRAPH_DIR, "market_structure.png")
        plt.close(fig)
        print(f"💾 Saved: {path}")

    def save_trade_recommendation_chart(row, trade, filename="trade_recommendation.png"):
        if trade['action'] == 'NO TRADE':
            print(f"⏭️  Skipping trade chart — no trade recommended")
            return
        input_date = row['Date'].iloc[0]
        idx_matches = df_all.index[df_all['Date'] == input_date]
        idx = int(idx_matches[0]) if len(idx_matches) else len(df_all) - 1
        base_window = df_all.iloc[max(0, idx - 60):idx + 1].copy()

        # Make sure the plotted series always ends on the row we're acting on,
        # so the yellow "last close" dot can't float off the end of the line.
        last_date = row['Date'].iloc[0]
        close_t = float(row['Close'].iloc[0])
        if len(base_window) and base_window['Date'].iloc[-1] == last_date:
            window = base_window
        else:
            extra_row = df_all[df_all['Date'] == last_date]
            if len(extra_row):
                window = pd.concat([base_window, extra_row], ignore_index=True).drop_duplicates('Date').sort_values('Date')
            else:
                window = pd.concat([base_window, row[['Date', 'Close']]], ignore_index=True) \
                           .drop_duplicates('Date').sort_values('Date')
            window = window.iloc[-61:]

        predicted_date = _next_trading_date(last_date)
        macro_s = int(row['Macro_Structure'].iloc[0])
        micro_s = int(row['Micro_Structure'].iloc[0])
        structure_str = {1: 'Bull', -1: 'Bear', 0: '??'}
        trend_label = f"Macro {structure_str[macro_s]} · Micro {structure_str[micro_s]}"

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(window['Date'], window['Close'],
                color='steelblue', linewidth=1.8, label='Close Price', zorder=3)

        hm = trade.get('hourly_meta')
        drawn_ob_levels = []  # only levels we actually draw go into the y-range

        # Draw the OB body/wick only when the trade actually went into that OB.
        # Otherwise (e.g. "market at close (no qualifying zone)") we skip it,
        # which is what was stretching the GOLD chart's y-axis.
        entry_type_str = (hm['daily_entry_type'] if hm is not None else trade['entry_type']) or ''
        used_bull_ob = ('Bull OB' in entry_type_str)
        used_bear_ob = ('Bear OB' in entry_type_str)

        if trade['action'] == 'LONG' and used_bull_ob:
            body_top    = row['Bull_OB_Body_Top'].iloc[0]
            body_bottom = row['Bull_OB_Body_Bottom'].iloc[0]
            wick_bottom = row['Bull_OB_Wick_Bottom'].iloc[0]
            if pd.notna(body_top) and pd.notna(wick_bottom):
                ax.axhspan(body_bottom, body_top, color='dodgerblue', alpha=0.18, zorder=1,
                           label=f"OB body: {body_bottom:.2f}–{body_top:.2f}")
                ax.axhspan(wick_bottom, body_bottom, color='cyan', alpha=0.12, zorder=1,
                           label=f"OB wick: {wick_bottom:.2f}–{body_bottom:.2f}")
                drawn_ob_levels += [body_top, body_bottom, wick_bottom]
        elif trade['action'] == 'SHORT' and used_bear_ob:
            body_top    = row['Bear_OB_Body_Top'].iloc[0]
            body_bottom = row['Bear_OB_Body_Bottom'].iloc[0]
            wick_top    = row['Bear_OB_Wick_Top'].iloc[0]
            if pd.notna(body_bottom) and pd.notna(wick_top):
                ax.axhspan(body_bottom, body_top, color='tomato', alpha=0.18, zorder=1,
                           label=f"OB body: {body_bottom:.2f}–{body_top:.2f}")
                ax.axhspan(body_top, wick_top, color='lightcoral', alpha=0.12, zorder=1,
                           label=f"OB wick: {body_top:.2f}–{wick_top:.2f}")
                drawn_ob_levels += [body_top, body_bottom, wick_top]

        if hm is not None:
            ax.axhline(hm['daily_entry'], color='grey', linestyle=':', linewidth=1.5,
                       zorder=4, label=f"Daily entry: {hm['daily_entry']:.2f}")
            ax.axhspan(hm['zone_bottom'], hm['zone_top'], color='purple', alpha=0.10,
                       zorder=1, label=f"Hourly {hm['kind'].replace('_',' ')} "
                                       f"{hm['zone_bottom']:.2f}–{hm['zone_top']:.2f}")
            if hm.get('ote_band_low') is not None and hm.get('ote_band_high') is not None:
                ax.axhspan(hm['ote_band_low'], hm['ote_band_high'],
                           color='gold', alpha=0.12, zorder=1,
                           label=f"OTE 0.62–0.79: {hm['ote_band_low']:.2f}–{hm['ote_band_high']:.2f}")
                if hm.get('ote_705') is not None:
                    ax.axhline(hm['ote_705'], color='gold', linestyle='-.', linewidth=1.2,
                               zorder=4, label=f"OTE 0.705: {hm['ote_705']:.2f}")

        entry_style_name = trade.get('entry_style', 'moderate')
        ax.axhline(trade['entry'], color='black', linestyle='-', linewidth=1.8,
                   zorder=5, label=f"Final entry ({entry_style_name}): {trade['entry']:.2f}")
        ax.axhline(trade['stop_loss'], color='red', linestyle='--', linewidth=1.5, zorder=4,
                   label=f"Stop: {trade['stop_loss']:.2f}  [{trade.get('stop_source','?')}]")
        ax.axhline(trade['take_profit'], color='green', linestyle='--', linewidth=1.5, zorder=4,
                   label=f"Target: {trade['take_profit']:.2f}  [{trade.get('target_source','?')}]")
        ax.axhspan(trade['stop_loss'], trade['entry'], color='red', alpha=0.06, zorder=0)
        ax.axhspan(trade['entry'], trade['take_profit'], color='green', alpha=0.06, zorder=0)

        # Yellow dot at the DAILY last close, on the same x as the last plotted candle.
        ax.scatter([last_date], [close_t], color='gold', s=140, marker='o', zorder=8,
                   edgecolor='black', linewidth=1.4, label=f"Close T: {close_t:,.2f}")

        x_min = window['Date'].min()
        x_max = window['Date'].max()
        span_days = (x_max - x_min).days if (x_max - x_min).days > 0 else 1
        right_pad_days = max(int(span_days * 0.22), (predicted_date - x_max).days + 2)
        ax.set_xlim(x_min, x_max + pd.Timedelta(days=right_pad_days))
        ax.axvline(predicted_date, color='#8e44ad', linestyle='--',
                   linewidth=2.2, alpha=0.95, zorder=6,
                   label=f"Trade window: {predicted_date.date()}")
        ax.annotate(f"▶ {predicted_date.date()}",
                    xy=(predicted_date, close_t),
                    xytext=(4, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=9, fontweight='bold',
                    color='#8e44ad',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='#8e44ad', alpha=0.95),
                    annotation_clip=False, zorder=11)

        signals_to_plot = []
        bull_ob_level   = row['Bull_OB_Level'].iloc[0]
        bear_ob_level   = row['Bear_OB_Level'].iloc[0]
        bull_fvg_bottom = row['FVG_Bull_Bottom'].iloc[0]
        bear_fvg_top    = row['FVG_Bear_Top'].iloc[0]
        last_swing_high = row['Last_Swing_High'].iloc[0]
        last_swing_low  = row['Last_Swing_Low'].iloc[0]
        if row['Bull_BB'].iloc[0] == 1 and pd.notna(bull_ob_level):
            signals_to_plot.append(('Bull Breaker', bull_ob_level, 'cyan', 'o'))
        if row['Bear_BB'].iloc[0] == 1 and pd.notna(bear_ob_level):
            signals_to_plot.append(('Bear Breaker', bear_ob_level, 'orange', 'o'))
        if row['Bull_IFVG'].iloc[0] == 1 and pd.notna(bull_fvg_bottom):
            signals_to_plot.append(('Bull IFVG', bull_fvg_bottom, 'lime', 's'))
        if row['Bear_IFVG'].iloc[0] == 1 and pd.notna(bear_fvg_top):
            signals_to_plot.append(('Bear IFVG', bear_fvg_top, 'magenta', 's'))
        if row['BOS_Bull'].iloc[0] == 1 and pd.notna(last_swing_high):
            signals_to_plot.append(('BOS Bull', last_swing_high, 'green', '^'))
        if row['BOS_Bear'].iloc[0] == 1 and pd.notna(last_swing_low):
            signals_to_plot.append(('BOS Bear', last_swing_low, 'red', 'v'))
        x_marker = window['Date'].iloc[-1] + pd.Timedelta(days=2)
        for name, level, color, marker in signals_to_plot:
            if abs(level - close_t) / close_t > 0.08:
                continue
            ax.plot([window['Date'].iloc[-1], x_marker], [level, level],
                    color=color, linewidth=2.5, alpha=0.9, zorder=6)
            ax.scatter([x_marker], [level], color=color, s=90, marker=marker,
                       zorder=7, edgecolor='black', linewidth=1.2)
            ax.annotate(f"{name}\n@ {level:.2f}", xy=(x_marker, level),
                        xytext=(8, 0), textcoords='offset points',
                        va='center', ha='left', fontsize=8, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                  edgecolor=color, alpha=0.9),
                        annotation_clip=False, zorder=10)

        signal_names = [s[0] for s in signals_to_plot]
        header_line = (f"{index_name} — {trade['action']} Setup ({trade.get('entry_style', 'moderate')} wick)  |  "
                       f"R:R {trade['risk_reward']}  |  {trade['conviction']} ({trade['confidence']:.1%})")
        trend_line = f"MTF: {trend_label}"
        date_line = f"{last_date.date()}  ->  Trade window: {predicted_date.date()}"
        title_lines = [header_line, f"{trend_line}    |    {date_line}"]
        if signal_names:
            title_lines.append(f"Signals: {' / '.join(signal_names)}")
        if hm is not None:
            status = "APPLIED" if hm.get('applied') else "INFO only"
            ote_tag = "OTE-aligned" if hm.get('ote_aligned') else "outside OTE"
            title_lines.append(
                f"Hourly entry: {hm['kind'].replace('_',' ')} "
                f"{hm['session']} {hm['hour_utc']:02d}:00 UTC "
                f"@ {hm['entry_price']:.2f}  [{status}, {ote_tag}]"
            )
        ax.set_title("\n".join(title_lines), fontsize=11, fontweight='bold')
        ax.set_ylabel("Price"); ax.set_xlabel("Date")
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9, edgecolor='gray')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = _safe_savefig(fig, INDEX_GRAPH_DIR, filename)
        plt.close(fig)
        print(f"💾 Saved: {path}")

    def save_trade_recommendation_hourly_chart(row, trade, filename="trade_recommendation.png"):
        if trade['action'] == 'NO TRADE':
            print(f"⏭️  Skipping hourly chart — no trade recommended")
            return
        if not HAS_HOURLY:
            print(f"⏭️  Skipping hourly chart — no hourly data")
            return
        hm = trade.get('hourly_meta')
        if hm is None:
            print(f"⏭️  Skipping hourly chart — no hourly recommendation")
            return

        n_bars = 160
        h_tail = hourly_df.tail(n_bars).reset_index(drop=True)
        if len(h_tail) < 20:
            print(f"⏭️  Skipping hourly chart — not enough hourly bars")
            return

        entry_price    = trade['entry']
        stop_price     = trade['stop_loss']
        target_price   = trade['take_profit']
        zone_top       = hm['zone_top']
        zone_bottom    = hm['zone_bottom']
        side           = trade['action']
        session        = hm['session']
        hour_utc       = hm['hour_utc']
        bars_since     = int(hm.get('bars_since_zone', 0))

        target_trade_date = _next_trading_date(row['Date'].iloc[0])
        origin_ts = h_tail.loc[max(0, len(h_tail) - 1 - bars_since), 'timestamp']
        origin_date = origin_ts.strftime('%d %b %Y')

        all_fvgs = _find_hourly_fvgs(hourly_df, lookback=n_bars)
        all_obs  = _find_hourly_order_blocks(hourly_df, lookback=n_bars)
        target_zone = None
        best_dist = 0.005 * target_price
        for z in all_fvgs + all_obs:
            z_top = z.get('top', z.get('body_top'))
            z_bot = z.get('bottom', z.get('body_bottom'))
            if z_top is None or z_bot is None:
                continue
            z_mid = (z_top + z_bot) / 2
            d = abs(z_mid - target_price)
            if d < best_dist:
                best_dist = d
                kind_suffix = 'fvg' if z in all_fvgs else 'ob'
                target_zone = {
                    'top': z_top, 'bottom': z_bot, 'mid': z_mid,
                    'kind': z.get('direction', '') + '_' + kind_suffix,
                }

        price_points = [entry_price, stop_price, target_price, zone_top, zone_bottom]
        if target_zone:
            price_points += [target_zone['top'], target_zone['bottom']]
        window_lo = min(h_tail['low'].min(), min(price_points))
        window_hi = max(h_tail['high'].max(), max(price_points))
        pad = (window_hi - window_lo) * 0.05
        y_lo = window_lo - pad
        y_hi = window_hi + pad

        h_tail = h_tail.reset_index(drop=True)
        h_tail['x'] = np.arange(len(h_tail))
        n = len(h_tail)
        future_x = n + 2

        fig, ax = plt.subplots(figsize=(15, 8))

        session_colors = {
            'Asia':        '#f0f4f8',
            'London':      '#f8f4f0',
            'New York':    '#f0f8f0',
            'Asia (late)': '#f4f0f8',
        }
        h_tail['session'] = h_tail['hour_utc'].apply(_session_from_hour)
        blocks = []
        current_session = None
        block_start = 0
        for i in range(n):
            s = h_tail.loc[i, 'session']
            if s != current_session:
                if current_session is not None:
                    blocks.append((block_start, i - 1, current_session))
                current_session = s
                block_start = i
        if current_session is not None:
            blocks.append((block_start, n - 1, current_session))
        for start, end, sess in blocks:
            ax.axvspan(start - 0.5, end + 0.5,
                       color=session_colors.get(sess, '#ffffff'),
                       alpha=0.6, zorder=0)

        body_width = 0.7
        for _, r in h_tail.iterrows():
            x = r['x']
            o, h_, l_, c = r['open'], r['high'], r['low'], r['close']
            ax.plot([x, x], [l_, h_], color='black', linewidth=0.9, zorder=3,
                    solid_capstyle='round')
            body_top_c = max(o, c)
            body_bot_c = min(o, c)
            body_height = body_top_c - body_bot_c
            face = 'white' if c >= o else 'black'
            rect = plt.Rectangle(
                (x - body_width / 2, body_bot_c),
                body_width, max(body_height, 1e-9),
                facecolor=face, edgecolor='black',
                linewidth=0.7, zorder=4,
            )
            ax.add_patch(rect)

        ax.axhspan(zone_bottom, zone_top,
                   color='#8e44ad', alpha=0.15, zorder=1,
                   label=f"{hm['kind'].replace('_', ' ').title()} zone")
        ax.axhline(entry_price, color='#8e44ad', linestyle='-',
                   linewidth=1.8, zorder=5,
                   label=f"Entry: {entry_price:.2f}")
        ax.axhline(stop_price, color='#c0392b', linestyle='--',
                   linewidth=1.5, zorder=5,
                   label=f"Stop: {stop_price:.2f}")
        ax.axhline(target_price, color='#27ae60', linestyle='--',
                   linewidth=1.5, zorder=5,
                   label=f"Target: {target_price:.2f}")
        if target_zone is not None:
            ax.axhspan(target_zone['bottom'], target_zone['top'],
                       color='#27ae60', alpha=0.12, zorder=1,
                       label=f"{target_zone['kind'].replace('_', ' ').title()} target zone")
        if hm.get('ote_band_low') is not None and hm.get('ote_band_high') is not None:
            ax.axhspan(hm['ote_band_low'], hm['ote_band_high'],
                       color='gold', alpha=0.12, zorder=1,
                       label=f"OTE 0.62–0.79")
            if hm.get('ote_705') is not None:
                ax.axhline(hm['ote_705'], color='gold', linestyle='-.',
                           linewidth=1.2, zorder=5, label=f"OTE 0.705: {hm['ote_705']:.2f}")
        ax.axhspan(stop_price, entry_price, color='#e74c3c', alpha=0.05, zorder=0)
        ax.axhspan(entry_price, target_price, color='#2ecc71', alpha=0.05, zorder=0)

        last_x = h_tail['x'].iloc[-1]
        last_close = h_tail['close'].iloc[-1]
        ax.scatter([last_x], [last_close], color='gold', s=120,
                   marker='o', zorder=8, edgecolor='black', linewidth=1.2)

        tick_positions = list(range(0, n, max(1, n // 8)))
        if tick_positions[-1] != n - 1:
            tick_positions.append(n - 1)
        tick_labels = [
            h_tail.loc[p, 'timestamp'].strftime('%d %b %H:%M')
            for p in tick_positions
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=0)

        origin_x = max(0, n - 1 - bars_since)
        ax.axvline(origin_x, color='#7f8c8d', linestyle=':', linewidth=1.4,
                   alpha=0.9, zorder=2,
                   label=f"Zone origin: {origin_date} {hour_utc:02d}:00 UTC")

        ax.axvline(future_x, color='#8e44ad', linestyle='--',
                   linewidth=2.2, alpha=0.95, zorder=6,
                   label=f"Trade window: {target_trade_date.date()} "
                         f"{session} {hour_utc:02d}:00 UTC")
        ax.annotate(f"▶ {target_trade_date.date()}\n{session} {hour_utc:02d}:00 UTC",
                    xy=(future_x, entry_price),
                    xytext=(6, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=8, fontweight='bold',
                    color='#8e44ad',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#8e44ad', alpha=0.95),
                    annotation_clip=False, zorder=11)

        ax.set_xlim(-1, future_x + 1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("Price", fontsize=10)
        ax.set_xlabel("")
        ax.grid(alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#cccccc')
            ax.spines[spine].set_linewidth(0.8)

        rr_str = f"{trade['risk_reward']}" if trade['risk_reward'] else "—"
        applied_str = "APPLIED" if hm['applied'] else "INFO ONLY"
        ote_str = " · OTE-aligned" if hm.get('ote_aligned') else ""
        title = (f"{index_name}  ·  {side}  ·  {applied_str}{ote_str}  ·  "
                 f"Trade window: {target_trade_date.date()}\n"
                 f"Entry {entry_price:.2f}  ·  Stop {stop_price:.2f}  ·  "
                 f"Target {target_price:.2f}  ·  R:R {rr_str}  ·  "
                 f"{session} {hour_utc:02d}:00 UTC")
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=12)

        ax.legend(loc='upper left', fontsize=8, framealpha=0.9,
                  edgecolor='#cccccc', ncol=2)

        plt.tight_layout()
        path = _safe_savefig(fig, INDEX_GRAPH_DIR, filename)
        plt.close(fig)
        print(f"💾 Saved: {path}")

    # ---------------- 8. RUN PIPELINE ----------------
    INDEX_GRAPH_DIR = _normalize_path(f"{index_name} graphs")
    os.makedirs(INDEX_GRAPH_DIR, exist_ok=True)
    LATEST_DATE = df['Date'].max().date()
    print(f"\n🔎 Latest date in CSV: {LATEST_DATE}")

    journal = None
    journal_file = _journal_path(index_name)
    if USE_JOURNAL:
        print("\n" + "-"*60)
        print(f"📓 TRADE JOURNAL — {index_name}")
        print("-"*60)
        journal = _load_journal(journal_file)
        n_resolved = _resolve_journal(journal, df)
        print(f"✅ Resolved {n_resolved} pending signal(s) against actual price action")
        _print_journal_summary(journal, index_name)
        learn_trades = _trades_for_learning(journal)
        journal_state['cals'] = _fit_calibrations(learn_trades)
        journal_state['meta'] = _fit_meta(learn_trades)
        print(f"\n🧠 LEARNING FROM JOURNAL ({len(learn_trades)} resolved trades)")
        for key in ('ALL', 'LONG', 'SHORT'):
            c = journal_state['cals'][key]
            if c.get('accepted'):
                s_, e_, t_ = c['params']
                print(f"   Calibration [{key:<5}] ACCEPTED  n={c['n']}  entry +{s_:.2f} ATR deeper, "
                      f"stop +{e_:.2f} ATR wider, target x{t_:.2f}  "
                      f"(validation: {c['hold_base']:+.3f}R -> {c['hold_learn']:+.3f}R per signal)")
            elif 'reason' in c:
                print(f"   Calibration [{key:<5}] not applied — {c['reason']}")
        meta = journal_state['meta']
        if meta.get('accepted'):
            print(f"   Trade-quality filter  ACTIVE  n={meta['n']}  hold-out AUC {meta['auc']:.3f}  "
                  f"(base win rate {meta['base_win_rate']:.1%})")
        else:
            print(f"   Trade-quality filter  not active — {meta.get('reason', 'n/a')}")
        _walkforward_report(journal, index_name)

    predicted_state, probs, row, trade, baseline_trade, adapt_info = predict_for_date(str(LATEST_DATE))

    if USE_JOURNAL:
        rec = adapt_info['record']
        rec['adapt_applied'] = bool(adapt_info['params'] is not None and trade['action'] != 'NO TRADE')
        if rec['adapt_applied']:
            rec['adapt_shift'], rec['adapt_extra'], rec['adapt_tscale'] = adapt_info['params']
            rec['adapt_entry'] = trade['entry']
            rec['adapt_stop'] = trade['stop_loss']
            rec['adapt_target'] = trade['take_profit']
        rec['journal_win_prob'] = adapt_info['meta_p'] if adapt_info['meta_p'] is not None else np.nan
        rec['meta_filtered'] = bool(adapt_info['filtered'])
        if trade.get('hourly_meta') is not None:
            hm = trade['hourly_meta']
            rec['hourly_hour_utc']        = hm.get('hour_utc', np.nan)
            rec['hourly_session']         = hm.get('session', '')
            rec['hourly_zone_kind']       = hm.get('kind', '')
            rec['hourly_entry_price']     = hm.get('entry_price', np.nan)
            rec['hourly_distance_pct']    = hm.get('distance_pct', np.nan)
            rec['hourly_bars_since_zone'] = hm.get('bars_since_zone', np.nan)
            rec['hourly_stop_reference']  = hm.get('stop_reference', np.nan)
            rec['hourly_applied']         = hm.get('applied', False)
            rec['hourly_risk_reward']     = hm.get('hourly_rr', np.nan)
            rec['ote_aligned']            = bool(hm.get('ote_aligned', False))
            rec['ote_705']                = hm.get('ote_705', np.nan) if hm.get('ote_705') is not None else np.nan
            rec['ote_band_low']           = hm.get('ote_band_low', np.nan) if hm.get('ote_band_low') is not None else np.nan
            rec['ote_band_high']          = hm.get('ote_band_high', np.nan) if hm.get('ote_band_high') is not None else np.nan
            rec['kill_zone']              = hm.get('kill_zone') or ''
            rec['session_weight']         = hm.get('session_weight', np.nan)
        keep = ~((journal['signal_date'] == rec['signal_date']) & (~journal['resolved']))
        journal = journal[keep]
        journal = pd.concat([journal, pd.DataFrame([rec])], ignore_index=True) if len(journal) \
            else pd.DataFrame([rec])
        journal = _coerce_journal(journal)
        _save_journal(journal, journal_file)
        save_journal_progress(journal, index_name, INDEX_GRAPH_DIR)

    print("\n🎨 Generating graphs...")
    save_price_with_prediction(row, probs)
    save_test_predictions_chart()
    save_confusion_matrix()
    save_feature_importance()
    save_probability_over_time()
    save_market_structure_chart()
    save_trade_recommendation_chart(row, trade)
    save_trade_recommendation_hourly_chart(row, trade)

    return {
        'index_name': index_name,
        'csv_path': csv_path,
        'accuracy': accuracy,
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)),
        'predicted_state': predicted_state,
        'probs': probs,
        'close': float(row['Close'].iloc[0]),
        'latest_date': str(LATEST_DATE),
        'trade': trade,
        'journal_adjusted': bool(adapt_info['params'] is not None),
        'journal': journal if USE_JOURNAL else None,
    }


# ============================================
# RUN ALL INDEXES
# ============================================

print("\n" + "="*80)
print("🚀 MULTI-INDEX PIPELINE STARTING")
print("="*80)
print(f"\nIndexes to process: {len(INDEX_CONFIGS)}")
for csv_p, name in INDEX_CONFIGS:
    print(f"   • {name:<12} → {csv_p}")

dxy_df = load_dxy_series(DXY_CSV_PATH)
results = []
for csv_path, index_name in INDEX_CONFIGS:
    try:
        res = run_single_index(csv_path, index_name, dxy_df=dxy_df)
        if res is not None:
            results.append(res)
    except Exception as e:
        print(f"\n❌ Error processing {index_name}: {e}")
        import traceback
        traceback.print_exc()


# ============================================
# FINAL CROSS-INDEX SUMMARY
# ============================================

print("\n\n" + "="*80)
print("📊 CROSS-INDEX SUMMARY")
print("="*80)
if not results:
    print("\n⚠️  No indexes were processed successfully.")
else:
    print(f"\n{'Index':<12}{'Latest Date':<14}{'Close':>10}{'Prediction':>14}{'Conf':>8}{'CV Mean':>10}")
    print("-"*80)
    for r in results:
        pred_short = {'UP': '🟢 UP', 'DOWN': '🔴 DOWN', 'CONSOLIDATION': '🟡 FLAT'}[r['predicted_state']]
        conf = max(r['probs'].values())
        print(f"{r['index_name']:<12}{r['latest_date']:<14}{r['close']:>10.2f}"
              f"{pred_short:>14}{conf:>7.1%}{r['cv_mean']:>10.2%}")

    print("\n" + "-"*80)
    print("Per-index trade recommendations:")
    for r in results:
        print(f"\n  {r['index_name']} ({r['latest_date']}):")
        trade = r['trade']
        if trade['action'] == 'NO TRADE':
            print(f"    → NO TRADE ({trade['reason']})")
        else:
            tag = "  [journal-adjusted]" if r.get('journal_adjusted') else ""
            hm = trade.get('hourly_meta')
            if hm is not None and hm.get('applied'):
                ote_tag = ", OTE-aligned" if hm.get('ote_aligned') else ""
                hourly_tag = (f"  [hourly {hm['kind'].replace('_',' ')} "
                              f"{hm['session']} {hm['hour_utc']:02d}:00 UTC @ {hm['entry_price']:.2f}"
                              f"{ote_tag}]")
            elif hm is not None:
                hourly_tag = (f"  [hourly info: {hm['kind'].replace('_',' ')} "
                              f"{hm['session']} {hm['hour_utc']:02d}:00 UTC @ {hm['entry_price']:.2f}]")
            else:
                hourly_tag = "  [daily structural entry — no H1 alignment]"
            print(f"    → {trade['action']}  entry {trade['entry']:.2f}  "
                  f"stop {trade['stop_loss']:.2f}  target {trade['take_profit']:.2f}  "
                  f"R:R {trade['risk_reward']}{tag}{hourly_tag}")

    print("\n🎨 Generating cross-index journal summary...")
    save_cross_index_journal_summary(
        {r['index_name']: r['journal'] for r in results if r.get('journal') is not None}
    )

    print("\n" + "="*80)
    print("✅ ALL INDEXES PROCESSED")
    print("="*80)

print("\n⚠️  Disclaimer: this is a research/backtesting tool. Predictions and trade")
print("   levels are model output based on historical patterns, not financial advice.")

