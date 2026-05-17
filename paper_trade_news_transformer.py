# paper_trade_news_transformer.py
# 今日ニュース取得 → Transformer予測 → ペーパー取引予定保存 → 後日答え合わせ → 勝率集計
#
# 個別株:
#   BUY予測のみペーパーBUY
#   弱気予測はNO_TRADE
#
# 指数:
#   1321/1570 強気 → 1321 or 1570 BUY
#   1321/1570 弱気 → 1357 BUY
#
# 出力:
#   data/paper_trade/paper_signals.csv
#   data/paper_trade/paper_results.csv
#   data/paper_trade/paper_summary.csv
#   data/paper_trade/today_candidates.csv

from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote
import hashlib
import time
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "paper_trade"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_EVENTS_CSV = DATA_DIR / "jquants_grid" / "language_stack_ab" / "language_events_collected.csv"

PAPER_SIGNALS_CSV = OUT_DIR / "paper_signals.csv"
PAPER_RESULTS_CSV = OUT_DIR / "paper_results.csv"
PAPER_SUMMARY_CSV = OUT_DIR / "paper_summary.csv"
TODAY_CANDIDATES_CSV = OUT_DIR / "today_candidates.csv"

JST_NOW = datetime.now()
RUN_DATE = JST_NOW.strftime("%Y-%m-%d")

CODES = {
    "72030": {
        "code4": "7203",
        "yf": "7203.T",
        "name": "トヨタ",
        "queries": ["トヨタ","トヨタ自動車","Toyota"],
    },

    "83060": {
        "code4": "8306",
        "yf": "8306.T",
        "name": "三菱UFJ",
        "queries": ["三菱UFJ","MUFG"],
    },

    "99840": {
        "code4": "9984",
        "yf": "9984.T",
        "name": "ソフトバンクG",
        "queries": ["ソフトバンクグループ","SoftBank Group","Arm"],
    },

    "13210": {
        "code4": "1321",
        "yf": "1321.T",
        "name": "日経225ETF",
        "queries": ["日経平均","日経225"],
    },

    "15700": {
        "code4": "1570",
        "yf": "1570.T",
        "name": "日経レバETF",
        "queries": ["日経レバ"],
    },

    "13570": {
        "code4": "1357",
        "yf": "1357.T",
        "name": "ダブルインバース",
        "queries": ["ダブルインバース"],
    },
}
HOLD_DAYS_LIST = [1, 5, 10]
LOOKBACK_DAYS = 10
EMB_DIM = 32

EPOCHS = 20
PATIENCE = 4
BATCH_SIZE = 32
LR = 1e-3

BUY_THR_GRID = np.arange(0.50, 0.81, 0.05)
BEAR_THR = 0.40  # prob_up <= 0.40 を弱気扱い


def normalize_code(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 4:
        return s + "0"
    return s


def make_id(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:20]


def next_business_day(d):
    d = pd.to_datetime(d).date() + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return pd.Timestamp(d)


def business_day_after(d, hold_days):
    d = pd.to_datetime(d)
    cur = d
    count = 0
    while count < hold_days:
        cur = cur + timedelta(days=1)
        if cur.weekday() < 5:
            count += 1
    return pd.Timestamp(cur.date())


def clean_text(s):
    return str(s or "").replace("\n", " ").replace("\r", " ").strip()


def download_yf_prices():
    rows = []

    for code, meta in CODES.items():
        print(f"[PRICE] {code} {meta['yf']}")
        try:
            df = yf.download(
                meta["yf"],
                period="2y",
                interval="1d",
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                continue

            df = df.reset_index()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            for _, r in df.iterrows():
                rows.append({
                    "date": pd.to_datetime(r["Date"]).strftime("%Y-%m-%d"),
                    "code": code,
                    "open": float(r["Open"]),
                    "high": float(r["High"]),
                    "low": float(r["Low"]),
                    "close": float(r["Close"]),
                    "volume": float(r["Volume"]) if "Volume" in r else 0,
                })

            time.sleep(0.3)

        except Exception as e:
            print(f"  [WARN] price failed {code}: {e}")

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    out.to_csv(OUT_DIR / "yf_prices.csv", index=False, encoding="utf-8-sig")
    return out


def load_historical_events():
    if not HIST_EVENTS_CSV.exists():
        print(f"[WARN] historical events not found: {HIST_EVENTS_CSV}")
        return pd.DataFrame(columns=["event_id", "datetime", "date", "code", "title", "body", "source", "url", "full_text"])

    ev = pd.read_csv(HIST_EVENTS_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev["code"] = ev["code"].apply(normalize_code)

    if "full_text" not in ev.columns:
        ev["full_text"] = ev["title"].astype(str) + " " + ev["body"].astype(str)

    ev = ev.dropna(subset=["date"])
    ev = ev[ev["code"].isin(CODES.keys())].copy()
    return ev


def gdelt_search(query, days_back=7, maxrecords=50):
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={quote(query)}"
        f"&mode=artlist"
        f"&format=json"
        f"&maxrecords={maxrecords}"
        f"&sort=hybridrel"
        f"&timespan={days_back}d"
    )

    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        return r.json().get("articles", []) or []
    except Exception:
        return []


def collect_today_news():
    rows = []
    seen = set()

    print("\n=== COLLECT TODAY NEWS ===")

    for code, meta in CODES.items():
        # yfinance
        print(f"[YF NEWS] {code}")
        try:
            ticker = yf.Ticker(meta["yf"])
            news = ticker.get_news(count=30, tab="news")
        except Exception as e:
            print(f"  [WARN] yfinance news failed: {e}")
            news = []

        for item in news:
            title = clean_text(item.get("title", ""))
            body = clean_text(item.get("summary", "") or item.get("description", "") or "")
            publisher = clean_text(item.get("publisher", ""))
            link = clean_text(item.get("link", ""))

            text = f"{title} {body} {publisher}"
            if not title:
                continue

            if not any(q.lower() in text.lower() for q in meta["queries"] + [meta["name"]]):
                continue

            ts = item.get("providerPublishTime")
            if ts:
                dt = datetime.fromtimestamp(int(ts))
            else:
                dt = JST_NOW

            event_id = make_id("yf", code, title, link)
            if event_id in seen:
                continue
            seen.add(event_id)

            rows.append({
                "event_id": event_id,
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "date": pd.to_datetime(dt.strftime("%Y-%m-%d")),
                "code": code,
                "title": title,
                "body": body,
                "source": f"yfinance:{publisher}",
                "url": link,
                "full_text": f"{title} {body}",
            })

        # GDELT
        for q in meta["queries"]:
            print(f"[GDELT] {code} {q}")
            articles = gdelt_search(q, days_back=7, maxrecords=50)

            for a in articles:
                title = clean_text(a.get("title", ""))
                domain = clean_text(a.get("domain", ""))
                url = clean_text(a.get("url", ""))
                if not title:
                    continue

                text = f"{title} {domain}"
                if not any(q2.lower() in text.lower() for q2 in meta["queries"] + [meta["name"]]):
                    continue

                try:
                    dt = pd.to_datetime(a.get("seendate"), utc=True).tz_convert("Asia/Tokyo").tz_localize(None)
                except Exception:
                    dt = pd.Timestamp(JST_NOW)

                event_id = make_id("gdelt", code, title, url)
                if event_id in seen:
                    continue
                seen.add(event_id)

                rows.append({
                    "event_id": event_id,
                    "datetime": pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M:%S"),
                    "date": pd.to_datetime(dt).normalize(),
                    "code": code,
                    "title": title,
                    "body": domain,
                    "source": f"GDELT:{domain}",
                    "url": url,
                    "full_text": title,
                })

            time.sleep(0.5)

    today = pd.DataFrame(rows)

    if today.empty:
        print("[WARN] no fresh news found. Using historical latest context only.")
        today = pd.DataFrame(columns=["event_id", "datetime", "date", "code", "title", "body", "source", "url", "full_text"])

    today.to_csv(OUT_DIR / "today_news.csv", index=False, encoding="utf-8-sig")
    print(f"\n[TODAY NEWS] rows={len(today)}")
    if not today.empty:
        print(today[["date", "code", "title", "source"]].head(30).to_string(index=False))

    return today


def build_daily_text(events):
    if events.empty:
        return pd.DataFrame(columns=["date", "code", "event_count", "text"])

    rows = []
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()

    for (date, code), g in events.groupby(["date", "code"]):
        rows.append({
            "date": date,
            "code": code,
            "event_count": len(g),
            "text": "。".join(g["full_text"].astype(str).tolist()),
        })

    return pd.DataFrame(rows)


def build_embeddings(daily_text):
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    if daily_text.empty:
        raise RuntimeError("daily_text is empty")

    print("\n=== EMBEDDING ===")
    texts = daily_text["text"].astype(str).tolist()

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

    n_comp = min(EMB_DIM, emb.shape[0] - 1, emb.shape[1])
    if n_comp < 2:
        raise RuntimeError("not enough embedding rows")

    pca = PCA(n_components=n_comp, random_state=42)
    emb_red = pca.fit_transform(emb)

    out = daily_text[["date", "code", "event_count"]].copy()
    for j in range(n_comp):
        out[f"emb_{j}"] = emb_red[:, j]

    out.to_csv(OUT_DIR / "daily_embeddings.csv", index=False, encoding="utf-8-sig")
    return out

def build_embedding_dynamics(emb):

    emb = emb.sort_values(
        ["code","date"]
    ).copy()

    emb_cols = [
        c for c in emb.columns
        if c.startswith("emb_")
    ]

    out=[]

    for code,g in emb.groupby("code"):

        E = g[emb_cols].values

        vel = np.diff(
            E,
            axis=0,
            prepend=E[[0]]
        )

        acc = np.diff(
            vel,
            axis=0,
            prepend=vel[[0]]
        )

        jerk = np.diff(
            acc,
            axis=0,
            prepend=acc[[0]]
        )

        tmp = g[
            ["date","code"]
        ].copy()

        tmp["emb_energy"] = (
            vel**2
        ).sum(axis=1)

        tmp["emb_curvature"] = (
            np.linalg.norm(acc,axis=1)
            /
            (
                np.linalg.norm(
                    vel,
                    axis=1
                )+1e-6
            )
        )

        for j in range(8):

            tmp[f"vel_{j}"]=vel[:,j]
            tmp[f"acc_{j}"]=acc[:,j]
            tmp[f"jerk_{j}"]=jerk[:,j]

        out.append(tmp)

    return pd.concat(out)


def add_labels(prices):
    df = prices.sort_values(["code", "date"]).copy()

    for h in HOLD_DAYS_LIST:
        df[f"future_close_{h}d"] = df.groupby("code")["close"].shift(-h)
        df[f"ret_{h}d"] = (df[f"future_close_{h}d"] - df["close"]) / df["close"]
        df[f"target_up_{h}d"] = (df[f"ret_{h}d"] > 0).astype(int)

    return df


def build_panel(prices, emb):
    prices = add_labels(prices)

    emb = emb.copy()
    emb["date"] = pd.to_datetime(emb["date"])
    emb["code"] = emb["code"].apply(normalize_code)

    panel = prices.merge(emb, on=["date", "code"], how="left")

    emb_cols = [c for c in panel.columns if c.startswith("emb_")]
    for c in emb_cols:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0)

    panel["event_count"] = pd.to_numeric(panel.get("event_count", 0), errors="coerce").fillna(0)

    panel["ret_1"] = panel.groupby("code")["close"].pct_change(1).fillna(0)
    panel["ret_5"] = panel.groupby("code")["close"].pct_change(5).fillna(0)
    panel["vol_chg_5"] = panel.groupby("code")["volume"].pct_change(5).replace([np.inf, -np.inf], 0).fillna(0)

    dyn_cols = [

    c for c in panel.columns

    if

    c.startswith("vel_")

    or

    c.startswith("acc_")

    or

    c.startswith("jerk_")

    or

    c in [
    "emb_energy",
    "emb_curvature"
    ]

    ]

    feature_cols = (

    emb_cols

    +

    dyn_cols

    +

    [
    "event_count",

    "ret_1",

    "ret_5",

    "vol_chg_5"
    ]

    )

    panel.to_csv(OUT_DIR / "paper_panel.csv", index=False, encoding="utf-8-sig")
    return panel, feature_cols


def make_sequences(code_df, feature_cols, hold_days, include_unlabeled=False):
    X, y, ret, dates, closes = [], [], [], [], []

    target_col = f"target_up_{hold_days}d"
    ret_col = f"ret_{hold_days}d"

    arr = code_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)

    for i in range(LOOKBACK_DAYS - 1, len(code_df)):
        r = code_df.iloc[i]

        if not include_unlabeled and pd.isna(r[ret_col]):
            continue

        X.append(arr[i - LOOKBACK_DAYS + 1:i + 1])
        y.append(0 if pd.isna(r[target_col]) else int(r[target_col]))
        ret.append(np.nan if pd.isna(r[ret_col]) else float(r[ret_col]))
        dates.append(pd.to_datetime(r["date"]))
        closes.append(float(r["close"]))

    if not X:
        return None

    return {
        "X": np.stack(X),
        "y": np.array(y, dtype=np.int64),
        "ret": np.array(ret, dtype=np.float32),
        "date": np.array(dates),
        "close": np.array(closes, dtype=np.float32),
    }


def split_sequences(seq):
    dates = pd.to_datetime(seq["date"])
    unique_dates = sorted(pd.Series(dates).unique())

    n = len(unique_dates)
    train_end = unique_dates[int(n * 0.60)]
    valid_end = unique_dates[int(n * 0.80)]

    train_idx = dates < train_end
    valid_idx = (dates >= train_end) & (dates < valid_end)
    test_idx = dates >= valid_end

    def sub(mask):
        return {k: v[mask] for k, v in seq.items()}

    return sub(train_idx), sub(valid_idx), sub(test_idx)


def profit_factor(pnls):
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    if not losses:
        return 999.0 if wins else 0.0
    return float(sum(wins) / abs(sum(losses)))


def evaluate_probs(probs, rets, thr):
    mask = probs >= thr
    if mask.sum() == 0:
        return {"trades": 0, "winrate": 0, "sum_pnl": 0, "avg_pnl": 0, "pf": 0}

    pnls = rets[mask]
    pnls = pnls[~np.isnan(pnls)]
    if len(pnls) == 0:
        return {"trades": 0, "winrate": 0, "sum_pnl": 0, "avg_pnl": 0, "pf": 0}

    return {
        "trades": int(len(pnls)),
        "winrate": float((pnls > 0).mean()),
        "sum_pnl": float(pnls.sum()),
        "avg_pnl": float(pnls.mean()),
        "pf": profit_factor(pnls.tolist()),
    }


def train_transformer(train, valid, input_dim):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TinyNewsTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(input_dim, 64)
            layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                dropout=0.15,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(
                nn.LayerNorm(64),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            z = self.proj(x)
            z = self.encoder(z)
            return self.head(z[:, -1, :])

    model = TinyNewsTransformer().to(device)

    X_train = torch.tensor(train["X"], dtype=torch.float32)
    y_train = torch.tensor(train["y"], dtype=torch.long)

    X_valid = torch.tensor(valid["X"], dtype=torch.float32).to(device)
    y_valid = torch.tensor(valid["y"], dtype=torch.long).to(device)

    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = 999
    best_state = None
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(model(X_valid), y_valid).item()

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= PATIENCE:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, device


def predict_probs(model, device, X):
    import torch

    model.eval()
    out = []

    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.tensor(X[i:i + BATCH_SIZE], dtype=torch.float32).to(device)
            p = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            out.append(p)

    return np.concatenate(out)


def train_models_and_predict(panel, feature_cols):
    candidates = []
    all_grid = []

    for code in CODES.keys():
        code_df = panel[panel["code"] == code].copy().sort_values("date")

        if len(code_df) < 150:
            continue

        for hold_days in HOLD_DAYS_LIST:
            seq = make_sequences(code_df, feature_cols, hold_days, include_unlabeled=False)
            if seq is None or len(seq["X"]) < 100:
                continue

            train, valid, test = split_sequences(seq)

            if len(train["X"]) < 50 or len(valid["X"]) < 20:
                continue

            model, device = train_transformer(train, valid, train["X"].shape[-1])
            valid_probs = predict_probs(model, device, valid["X"])

            best = None
            for thr in BUY_THR_GRID:
                m = evaluate_probs(valid_probs, valid["ret"], thr)
                if m["trades"] < 3:
                    continue

                score = m["sum_pnl"] + m["pf"] * 0.01 + m["winrate"] * 0.01
                if best is None or score > best["score"]:
                    best = {"threshold": float(thr), "score": score, **m}

            if best is None:
                continue

            # 最新行を予測
            full_seq = make_sequences(code_df, feature_cols, hold_days, include_unlabeled=True)
            latest_X = full_seq["X"][-1:]
            latest_date = pd.to_datetime(full_seq["date"][-1])
            latest_close = float(full_seq["close"][-1])

            prob_up = float(predict_probs(model, device, latest_X)[0])

            action = "NO_TRADE"
            trade_code = code
            direction = "NEUTRAL"

            if prob_up >= best["threshold"]:
                action = "BUY"
                direction = "UP"
                trade_code = code
            elif prob_up <= BEAR_THR:
                direction = "DOWN"
                if code in ["13210", "15700"]:
                    action = "BUY"
                    trade_code = "13570"
                else:
                    action = "NO_TRADE"

            entry_date = next_business_day(latest_date)
            exit_date = business_day_after(entry_date, hold_days)

            row = {
                "run_date": RUN_DATE,
                "signal_date": latest_date.strftime("%Y-%m-%d"),
                "planned_entry_date": entry_date.strftime("%Y-%m-%d"),
                "planned_exit_date": exit_date.strftime("%Y-%m-%d"),
                "source_code": code,
                "trade_code": trade_code,
                "name": CODES.get(trade_code, CODES[code])["name"],
                "hold_days": hold_days,
                "prob_up": round(prob_up, 6),
                "threshold": round(best["threshold"], 4),
                "direction": direction,
                "action": action,
                "latest_close": latest_close,
                "valid_trades": best["trades"],
                "valid_winrate": best["winrate"],
                "valid_sum_pnl": best["sum_pnl"],
                "valid_pf": best["pf"],
                "status": "PENDING" if action == "BUY" else "NO_TRADE",
            }

            candidates.append(row)
            all_grid.append(row)

            print(
                f"[PRED] {code} hold={hold_days} prob_up={prob_up:.3f} "
                f"thr={best['threshold']:.2f} direction={direction} action={action} trade={trade_code}"
            )

    cand = pd.DataFrame(candidates)
    cand.to_csv(TODAY_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    return cand


def append_new_signals(candidates):
    if candidates.empty:
        return

    buys = candidates[candidates["action"] == "BUY"].copy()
    if buys.empty:
        print("[SIGNAL] no BUY signals today")
        return

    # 同一銘柄は valid_sum と prob で一番良いものだけ
    buys["rank_score"] = buys["valid_sum_pnl"].astype(float) + buys["prob_up"].astype(float) * 0.1
    buys = buys.sort_values("rank_score", ascending=False).groupby("trade_code").head(1)

    if PAPER_SIGNALS_CSV.exists():
        hist = pd.read_csv(PAPER_SIGNALS_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    else:
        hist = pd.DataFrame()

    existing_ids = set(hist["signal_id"]) if not hist.empty and "signal_id" in hist.columns else set()

    rows = []
    for _, r in buys.iterrows():
        sid = make_id(r["run_date"], r["trade_code"], r["planned_entry_date"], r["planned_exit_date"], r["hold_days"])
        if sid in existing_ids:
            continue

        d = r.to_dict()
        d["signal_id"] = sid
        d["entry_open"] = ""
        d["exit_close"] = ""
        d["pnl_pct"] = ""
        d["win"] = ""
        rows.append(d)

    if not rows:
        print("[SIGNAL] no new signals")
        return

    out = pd.DataFrame(rows)

    if hist.empty:
        new_hist = out
    else:
        new_hist = pd.concat([hist, out], ignore_index=True)

    new_hist.to_csv(PAPER_SIGNALS_CSV, index=False, encoding="utf-8-sig")

    print("\n=== NEW PAPER SIGNALS ===")
    print(out[["planned_entry_date", "planned_exit_date", "trade_code", "name", "hold_days", "direction", "prob_up", "status"]].to_string(index=False))

def apply_early_exit_if_signal_disappeared(candidates):
    """
    既存のPENDING/OPENポジションについて、
    今日の再推論でBUYシグナルが消えた場合、
    手じまい予定に変更する。

    paper_signals.csv上では、
    status = EARLY_EXIT
    planned_exit_date = 次営業日
    exit_reason = SIGNAL_DISAPPEARED
    にする。
    """

    if not PAPER_SIGNALS_CSV.exists():
        return

    sig = pd.read_csv(
        PAPER_SIGNALS_CSV,
        dtype=str,
        encoding="utf-8-sig"
    ).fillna("")

    if sig.empty:
        return

    if candidates is None or candidates.empty:
        active_buy_keys = set()
    else:
        buys = candidates[
            candidates["action"].astype(str) == "BUY"
        ].copy()

        active_buy_keys = set(
            buys["trade_code"].astype(str)
        )

    today = pd.to_datetime(RUN_DATE)
    exit_date = next_business_day(today)

    changed = 0

    for idx, r in sig.iterrows():

        status = str(r.get("status", ""))

        if status not in ["PENDING", "OPEN"]:
            continue

        trade_code = str(r.get("trade_code", ""))

        if not trade_code:
            continue

        # 今日の推論で同じtrade_codeのBUYが残っていれば継続
        if trade_code in active_buy_keys:
            continue

        # BUYシグナルが消えたので早期手じまい予定
        sig.loc[idx, "status"] = "EARLY_EXIT"
        sig.loc[idx, "planned_exit_date"] = exit_date.strftime("%Y-%m-%d")
        sig.loc[idx, "exit_reason"] = "SIGNAL_DISAPPEARED"
        sig.loc[idx, "early_exit_checked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        changed += 1

    if changed > 0:
        sig.to_csv(
            PAPER_SIGNALS_CSV,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\n[EARLY EXIT] signal disappeared: {changed} position(s)")
    else:
        print("\n[EARLY EXIT] no disappeared signals")


def update_paper_results(prices):
    if not PAPER_SIGNALS_CSV.exists():
        print("[RESULT] no paper_signals.csv yet")
        return

    sig = pd.read_csv(PAPER_SIGNALS_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    if sig.empty:
        return

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["code"] = prices["code"].astype(str)

    for idx, r in sig.iterrows():
        if r.get("status") not in ["PENDING", "OPEN", "EARLY_EXIT"]:
            continue

        code = str(r["trade_code"])
        entry_date = pd.to_datetime(r["planned_entry_date"])
        exit_date = pd.to_datetime(r["planned_exit_date"])

        # ===================================
        # DEBUG:
        # exit日未到達でも現在価格で仮採点
        # ===================================

        today_max = prices["date"].max()

        if exit_date > today_max:

            exit_date = today_max


        px = prices[prices["code"] == code].copy()

        entry = px[px["date"] == entry_date]
        exit_ = px[px["date"] == exit_date]

        if entry.empty or exit_.empty:
            continue

        entry_open = float(entry.iloc[0]["open"])
        exit_close = float(exit_.iloc[0]["close"])

        pnl = (exit_close - entry_open) / entry_open

        sig.loc[idx, "entry_open"] = round(entry_open, 4)
        sig.loc[idx, "exit_close"] = round(exit_close, 4)
        sig.loc[idx, "pnl_pct"] = round(pnl * 100, 4)
        sig.loc[idx, "win"] = 1 if pnl > 0 else 0
        if r.get("status") == "EARLY_EXIT":
            sig.loc[idx, "status"] = "CLOSED_EARLY"
        else:
            sig.loc[idx, "status"] = "CLOSED"

    sig.to_csv(PAPER_SIGNALS_CSV, index=False, encoding="utf-8-sig")
    sig.to_csv(PAPER_RESULTS_CSV, index=False, encoding="utf-8-sig")

    closed = sig[sig["status"] == "CLOSED"].copy()
    if closed.empty:
        print("[RESULT] no closed trades yet")
        return

    closed["pnl_pct_num"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed["win_num"] = pd.to_numeric(closed["win"], errors="coerce")

    summary = (
        closed.groupby("trade_code")
        .agg(
            trades=("signal_id", "count"),
            winrate=("win_num", "mean"),
            sum_pnl_pct=("pnl_pct_num", "sum"),
            avg_pnl_pct=("pnl_pct_num", "mean"),
        )
        .reset_index()
    )

    total = pd.DataFrame([{
        "trade_code": "TOTAL",
        "trades": len(closed),
        "winrate": closed["win_num"].mean(),
        "sum_pnl_pct": closed["pnl_pct_num"].sum(),
        "avg_pnl_pct": closed["pnl_pct_num"].mean(),
    }])

    summary = pd.concat([summary, total], ignore_index=True)
    summary.to_csv(PAPER_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n=== PAPER SUMMARY ===")
    print(summary.to_string(index=False))

def build_forward_report():
    """
    paper_signals.csv からフォワード検証結果を整理する。

    出力:
      paper_forward_trades.csv
      paper_forward_summary_total.csv
      paper_forward_summary_by_code.csv
      paper_forward_summary_by_hold.csv
      paper_forward_summary_by_status.csv
      paper_forward_summary_by_month.csv
      paper_forward_recent.csv
    """

    if not PAPER_SIGNALS_CSV.exists():
        print("[FORWARD] no paper_signals.csv yet")
        return

    sig = pd.read_csv(
        PAPER_SIGNALS_CSV,
        dtype=str,
        encoding="utf-8-sig"
    ).fillna("")

    if sig.empty:
        print("[FORWARD] paper_signals.csv is empty")
        return

    closed_statuses = [
        "CLOSED",
        "CLOSED_EARLY",
    ]

    trades = sig[
        sig["status"].astype(str).isin(closed_statuses)
    ].copy()

    if trades.empty:
        print("[FORWARD] no closed trades yet")
        return

    # =========================
    # 型変換
    # =========================
    for c in [
        "pnl_pct",
        "win",
        "prob_up",
        "threshold",
        "valid_winrate",
        "valid_sum_pnl",
        "valid_pf",
        "hold_days",
    ]:
        if c in trades.columns:
            trades[c] = pd.to_numeric(
                trades[c],
                errors="coerce"
            )

    for c in [
        "run_date",
        "signal_date",
        "planned_entry_date",
        "planned_exit_date",
    ]:
        if c in trades.columns:
            trades[c] = pd.to_datetime(
                trades[c],
                errors="coerce"
            )

    if "pnl_pct" not in trades.columns:
        print("[FORWARD] pnl_pct column missing")
        return

    trades["pnl_pct"] = trades["pnl_pct"].fillna(0)
    trades["win"] = trades["win"].fillna(
        (trades["pnl_pct"] > 0).astype(int)
    )

    trades["entry_month"] = trades["planned_entry_date"].dt.strftime("%Y-%m")
    trades["exit_month"] = trades["planned_exit_date"].dt.strftime("%Y-%m")

    # =========================
    # 累積損益
    # =========================
    trades = trades.sort_values(
        ["planned_exit_date", "trade_code", "hold_days"]
    ).reset_index(drop=True)

    trades["cum_pnl_pct"] = trades["pnl_pct"].cumsum()
    trades["trade_no"] = np.arange(1, len(trades) + 1)

    # ドローダウン
    trades["equity_curve"] = 1.0 + trades["cum_pnl_pct"] / 100.0
    trades["equity_peak"] = trades["equity_curve"].cummax()
    trades["drawdown_pct"] = (
        (trades["equity_curve"] / trades["equity_peak"] - 1.0)
        * 100.0
    )

    # =========================
    # 集計関数
    # =========================
    def summarize(df):
        if df.empty:
            return pd.Series({
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "winrate": 0.0,
                "sum_pnl_pct": 0.0,
                "avg_pnl_pct": 0.0,
                "median_pnl_pct": 0.0,
                "best_pnl_pct": 0.0,
                "worst_pnl_pct": 0.0,
                "profit_factor_like": 0.0,
                "max_drawdown_pct": 0.0,
            })

        wins = df[df["pnl_pct"] > 0]["pnl_pct"]
        losses = df[df["pnl_pct"] < 0]["pnl_pct"]

        if len(losses) == 0:
            pf = 999.0 if len(wins) > 0 else 0.0
        else:
            pf = wins.sum() / abs(losses.sum())

        return pd.Series({
            "trades": len(df),
            "wins": int((df["pnl_pct"] > 0).sum()),
            "losses": int((df["pnl_pct"] <= 0).sum()),
            "winrate": float((df["pnl_pct"] > 0).mean()),
            "sum_pnl_pct": float(df["pnl_pct"].sum()),
            "avg_pnl_pct": float(df["pnl_pct"].mean()),
            "median_pnl_pct": float(df["pnl_pct"].median()),
            "best_pnl_pct": float(df["pnl_pct"].max()),
            "worst_pnl_pct": float(df["pnl_pct"].min()),
            "profit_factor_like": float(pf),
            "max_drawdown_pct": float(df["drawdown_pct"].min()),
        })

    # =========================
    # 出力
    # =========================
    trades.to_csv(
        OUT_DIR / "paper_forward_trades.csv",
        index=False,
        encoding="utf-8-sig"
    )

    total = summarize(trades).to_frame().T
    total.insert(0, "group", "TOTAL")
    total.to_csv(
        OUT_DIR / "paper_forward_summary_total.csv",
        index=False,
        encoding="utf-8-sig"
    )

    by_code = (
        trades
        .groupby(["trade_code", "name"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values("sum_pnl_pct", ascending=False)
    )
    by_code.to_csv(
        OUT_DIR / "paper_forward_summary_by_code.csv",
        index=False,
        encoding="utf-8-sig"
    )

    by_hold = (
        trades
        .groupby(["hold_days"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values("sum_pnl_pct", ascending=False)
    )
    by_hold.to_csv(
        OUT_DIR / "paper_forward_summary_by_hold.csv",
        index=False,
        encoding="utf-8-sig"
    )

    by_status = (
        trades
        .groupby(["status"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values("sum_pnl_pct", ascending=False)
    )
    by_status.to_csv(
        OUT_DIR / "paper_forward_summary_by_status.csv",
        index=False,
        encoding="utf-8-sig"
    )

    by_month = (
        trades
        .groupby(["exit_month"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values("exit_month")
    )
    by_month.to_csv(
        OUT_DIR / "paper_forward_summary_by_month.csv",
        index=False,
        encoding="utf-8-sig"
    )

    recent = trades.tail(30).copy()
    recent.to_csv(
        OUT_DIR / "paper_forward_recent.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # =========================
    # コンソール表示
    # =========================
    print("\n=== FORWARD TOTAL ===")
    print(total.to_string(index=False))

    print("\n=== FORWARD BY CODE ===")
    print(
        by_code[
            [
                "trade_code",
                "name",
                "trades",
                "winrate",
                "sum_pnl_pct",
                "avg_pnl_pct",
                "profit_factor_like",
                "max_drawdown_pct",
            ]
        ].to_string(index=False)
    )

    print("\n=== FORWARD BY HOLD DAYS ===")
    print(
        by_hold[
            [
                "hold_days",
                "trades",
                "winrate",
                "sum_pnl_pct",
                "avg_pnl_pct",
                "profit_factor_like",
                "max_drawdown_pct",
            ]
        ].to_string(index=False)
    )

    print("\n=== FORWARD RECENT 10 ===")
    show_cols = [
        "planned_entry_date",
        "planned_exit_date",
        "trade_code",
        "name",
        "hold_days",
        "status",
        "prob_up",
        "pnl_pct",
        "win",
        "cum_pnl_pct",
        "drawdown_pct",
    ]

    show_cols = [
        c for c in show_cols
        if c in recent.columns
    ]

    print(
        recent[show_cols]
        .tail(10)
        .to_string(index=False)
    )

def main():
    print("=== JStockLLM Paper Trading ===")

    prices = download_yf_prices()

    hist_events = load_historical_events()
    today_events = collect_today_news()

    all_events = pd.concat([hist_events, today_events], ignore_index=True)
    all_events["date"] = pd.to_datetime(all_events["date"])
    all_events["code"] = all_events["code"].apply(normalize_code)
    all_events["full_text"] = all_events["full_text"].astype(str)
    all_events = all_events.drop_duplicates(subset=["event_id"], keep="last")

    all_events.to_csv(OUT_DIR / "all_events_for_training.csv", index=False, encoding="utf-8-sig")

    daily_text = build_daily_text(all_events)
    emb = build_embeddings(
        daily_text
    )

    dyn = build_embedding_dynamics(
        emb
    )

    panel, feature_cols = build_panel(
        prices,
        emb
    )

    panel = panel.merge(
        dyn,
        on=["date","code"],
        how="left"
    )
    print("\n=== DATA ===")
    print(f"prices      : {prices.shape}")
    print(f"hist_events : {hist_events.shape}")
    print(f"today_news  : {today_events.shape}")
    print(f"all_events  : {all_events.shape}")
    print(f"daily_text  : {daily_text.shape}")
    print(f"features    : {len(feature_cols)}")

    update_paper_results(prices)

    candidates = train_models_and_predict(
        panel,
        feature_cols
    )

    apply_early_exit_if_signal_disappeared(
        candidates
    )

    append_new_signals(
        candidates
    )
    build_forward_report()

    print("\n=== OUTPUTS ===")
    print(PAPER_SIGNALS_CSV)
    print(PAPER_RESULTS_CSV)
    print(PAPER_SUMMARY_CSV)
    print(TODAY_CANDIDATES_CSV)


if __name__ == "__main__":
    main()