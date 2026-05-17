# context_time_rotation_ab.py
# A: LLM context only
# B: LLM context + time-rotated context vectors
#
# run:
#   cd C:\Tools\JStockLLM
#   python context_time_rotation_ab.py

from pathlib import Path
import math
import warnings
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
JQ_DIR = DATA_DIR / "jquants_grid"
LANG_DIR = JQ_DIR / "language_stack_ab"
OUT_DIR = JQ_DIR / "context_time_rotation_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_CSV = JQ_DIR / "prices_clean.csv"
EVENTS_CSV = LANG_DIR / "language_events_collected.csv"

CODES = ["72030", "83060", "67580", "99840", "13210", "15700", "13570"]

HOLD_DAYS_LIST = [1, 5, 10]
LOOKBACK_DAYS = 10

EMB_DIM = 32
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
PATIENCE = 5
THRESHOLDS = np.arange(0.50, 0.81, 0.05)


def normalize_code(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 4:
        return s + "0"
    return s


def load_prices():
    df = pd.read_csv(PRICE_CSV, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].apply(normalize_code)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["code"].isin(CODES)].copy()
    df = df.dropna(subset=["date", "code", "close"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def load_events():
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(
            f"not found: {EVENTS_CSV}\n"
            "先に language_stack_ab_test.py を実行して language_events_collected.csv を作ってください。"
        )

    ev = pd.read_csv(EVENTS_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev["code"] = ev["code"].apply(normalize_code)

    if "full_text" not in ev.columns:
        ev["full_text"] = ev.get("title", "").astype(str) + " " + ev.get("body", "").astype(str)

    ev = ev.dropna(subset=["date"])
    ev = ev[ev["code"].isin(CODES)].copy()
    ev["full_text"] = ev["full_text"].astype(str).str.replace(r"\s+", " ", regex=True)

    return ev.sort_values(["code", "date"]).reset_index(drop=True)


def build_daily_text(events):
    rows = []

    for (date, code), g in events.groupby(["date", "code"]):
        text = "。".join(g["full_text"].astype(str).tolist())
        rows.append(
            {
                "date": pd.to_datetime(date).normalize(),
                "code": code,
                "event_count": len(g),
                "text": text,
            }
        )

    daily = pd.DataFrame(rows)
    daily.to_csv(OUT_DIR / "daily_text.csv", index=False, encoding="utf-8-sig")
    return daily


def build_embeddings(daily_text):
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    print("\n=== BUILD SENTENCE EMBEDDINGS ===")

    texts = daily_text["text"].astype(str).tolist()

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    emb = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    n_comp = min(EMB_DIM, emb.shape[0] - 1, emb.shape[1])

    pca = PCA(n_components=n_comp, random_state=42)
    emb_red = pca.fit_transform(emb)

    out = daily_text[["date", "code", "event_count"]].copy()

    for j in range(n_comp):
        out[f"emb_{j}"] = emb_red[:, j]

    out.to_csv(OUT_DIR / "daily_context_embeddings.csv", index=False, encoding="utf-8-sig")

    print(f"embedding rows: {len(out)}")
    print(f"embedding dim : {n_comp}")

    return out


def rotate_vector(vec, angle):
    """
    2次元ペアごとに回転。
    emb[0],emb[1] を angle 回転
    emb[2],emb[3] を 2*angle 回転
    emb[4],emb[5] を 3*angle 回転
    という形で、時期によって文脈ベクトルの向きを変える。
    """
    v = np.array(vec, dtype=float).copy()

    for i in range(0, len(v) - 1, 2):
        freq = 1 + (i // 2) % 4
        a = angle * freq

        x = v[i]
        y = v[i + 1]

        v[i] = x * math.cos(a) - y * math.sin(a)
        v[i + 1] = x * math.sin(a) + y * math.cos(a)

    return v


def add_time_rotation(emb_df):
    """
    発言時期をEmbeddingに織り込む。
    使う時期要素:
    - 年内周期 day_of_year
    - 月周期 month
    - 銘柄ごとの直近イベントからの経過日数
    """
    df = emb_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].apply(normalize_code)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    days_since_list = []

    for code, sub in df.groupby("code"):
        last = None
        arr = []

        for d in sub["date"]:
            if last is None:
                arr.append(0)
            else:
                arr.append((d - last).days)
            last = d

        days_since_list.extend(arr)

    df["days_since_prev_event"] = days_since_list

    rotated_rows = []

    for _, r in df.iterrows():
        d = pd.to_datetime(r["date"])

        day_phase = 2 * math.pi * (d.dayofyear / 365.25)
        month_phase = 2 * math.pi * (d.month / 12.0)
        recency_phase = 2 * math.pi * (min(float(r["days_since_prev_event"]), 90.0) / 90.0)

        angle = day_phase + 0.5 * month_phase + 0.25 * recency_phase

        vec = r[emb_cols].astype(float).values
        rot = rotate_vector(vec, angle)

        out = {
            "date": r["date"],
            "code": r["code"],
            "event_count": r["event_count"],
            "days_since_prev_event": r["days_since_prev_event"],
        }

        for j, val in enumerate(rot):
            out[f"rot_emb_{j}"] = float(val)

        rotated_rows.append(out)

    rot_df = pd.DataFrame(rotated_rows)
    rot_df.to_csv(OUT_DIR / "daily_time_rotated_embeddings.csv", index=False, encoding="utf-8-sig")

    return rot_df

def build_embedding_dynamics(emb_df):

    df = emb_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    out_rows=[]

    for code, sub in df.groupby("code"):

        sub=sub.sort_values("date").reset_index(drop=True)

        E=sub[emb_cols].values.astype(float)

        vel=np.diff(E,axis=0,prepend=E[[0]])
        acc=np.diff(vel,axis=0,prepend=vel[[0]])
        jerk=np.diff(acc,axis=0,prepend=acc[[0]])

        energy=(vel**2).sum(axis=1)

        curvature=np.linalg.norm(acc,axis=1) / (
            np.linalg.norm(vel,axis=1)+1e-6
        )

        cosine_prev=[
            1.0
            if i==0 else
            cosine_similarity(
                E[i:i+1],
                E[i-1:i]
            )[0,0]
            for i in range(len(E))
        ]

        tmp=sub[["date","code","event_count"]].copy()

        tmp["emb_energy"]=energy
        tmp["emb_curvature"]=curvature
        tmp["emb_cos_prev"]=cosine_prev

        for j in range(min(8,vel.shape[1])):
            tmp[f"vel_{j}"]=vel[:,j]

        for j in range(min(8,acc.shape[1])):
            tmp[f"acc_{j}"]=acc[:,j]

        for j in range(min(8,jerk.shape[1])):
            tmp[f"jerk_{j}"]=jerk[:,j]

        out_rows.append(tmp)

    out=pd.concat(out_rows)

    out.to_csv(
        OUT_DIR/"embedding_dynamics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return out

def build_text_rhythm(daily_text):

    rows=[]

    for _,r in daily_text.iterrows():

        txt=str(r["text"])

        words=txt.split()

        rows.append({

            "date":r["date"],
            "code":r["code"],

            "char_len":
                len(txt),

            "word_count":
                len(words),

            "comma_count":
                txt.count("、"),

            "period_count":
                txt.count("。"),

            "avg_word_len":
                np.mean(
                    [len(x) for x in words]
                ) if words else 0,

        })

    out=pd.DataFrame(rows)

    out.to_csv(
        OUT_DIR/"text_rhythm.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return out


def add_labels(prices):
    df = prices.sort_values(["code", "date"]).copy()

    for h in HOLD_DAYS_LIST:
        df[f"future_close_{h}d"] = df.groupby("code")["close"].shift(-h)
        df[f"ret_{h}d"] = (df[f"future_close_{h}d"] - df["close"]) / df["close"]
        df[f"target_up_{h}d"] = (df[f"ret_{h}d"] > 0).astype(int)

    return df


def build_panel(prices, emb_df, rot_df):
    prices = add_labels(prices)

    emb_df = emb_df.copy()
    emb_df["date"] = pd.to_datetime(emb_df["date"])
    emb_df["code"] = emb_df["code"].apply(normalize_code)

    rot_df = rot_df.copy()
    rot_df["date"] = pd.to_datetime(rot_df["date"])
    rot_df["code"] = rot_df["code"].apply(normalize_code)

    panel = prices.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].apply(normalize_code)

    # =========================
    # Base embedding columns
    # =========================
    a_cols = [
        c for c in emb_df.columns
        if c.startswith("emb_")
    ]

    # =========================
    # Time-rotated embedding columns
    # =========================
    b_cols = [
        c for c in rot_df.columns
        if c.startswith("rot_emb_")
    ]

    # =========================
    # Merge context embedding
    # =========================
    emb_merge_cols = ["date", "code"] + a_cols

    if "event_count" in emb_df.columns:
        emb_merge_cols.append("event_count")

    panel = panel.merge(
        emb_df[emb_merge_cols],
        on=["date", "code"],
        how="left",
    )

    # =========================
    # Merge rotated embedding
    # =========================
    rot_merge_cols = ["date", "code"] + b_cols

    if "days_since_prev_event" in rot_df.columns:
        rot_merge_cols.append("days_since_prev_event")

    if "event_count" in rot_df.columns:
        rot_merge_cols.append("event_count")

    panel = panel.merge(
        rot_df[rot_merge_cols],
        on=["date", "code"],
        how="left",
        suffixes=("", "_rot"),
    )

    # =========================
    # Fix duplicated event_count columns
    # =========================
    event_cols = [
        c for c in panel.columns
        if c == "event_count" or c.startswith("event_count_")
    ]

    if event_cols:
        panel["event_count"] = (
            panel[event_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .max(axis=1)
        )

        drop_event_cols = [
            c for c in event_cols
            if c != "event_count"
        ]

        panel = panel.drop(
            columns=drop_event_cols,
            errors="ignore",
        )
    else:
        panel["event_count"] = 0

    # =========================
    # Fill numeric columns
    # =========================
    for c in a_cols + b_cols:
        if c in panel.columns:
            panel[c] = (
                pd.to_numeric(panel[c], errors="coerce")
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
            )

    if "days_since_prev_event" in panel.columns:
        panel["days_since_prev_event"] = (
            pd.to_numeric(panel["days_since_prev_event"], errors="coerce")
            .replace([np.inf, -np.inf], 999)
            .fillna(999)
        )
    else:
        panel["days_since_prev_event"] = 999

    panel["event_count"] = (
        pd.to_numeric(panel["event_count"], errors="coerce")
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # =========================
    # Dynamics / rhythm columns
    # NOTE:
    # dyn / rhythm は main() 側で panel に merge したあとに
    # feature_sets を作る場合もあるため、ここでは存在するものだけ拾う
    # =========================
    dyn_cols = []

    for c in panel.columns:
        if c.startswith("vel_"):
            dyn_cols.append(c)
        elif c.startswith("acc_"):
            dyn_cols.append(c)
        elif c.startswith("jerk_"):
            dyn_cols.append(c)
        elif c in [
            "emb_energy",
            "emb_curvature",
            "emb_cos_prev",
            "char_len",
            "word_count",
            "comma_count",
            "period_count",
            "avg_word_len",
        ]:
            dyn_cols.append(c)

    # 重複除去
    dyn_cols = list(dict.fromkeys(dyn_cols))

    # 存在する列だけ使う
    a_cols = [
        c for c in a_cols
        if c in panel.columns
    ]

    b_cols = [
        c for c in b_cols
        if c in panel.columns
    ]

    base_cols = [
        c for c in ["event_count"]
        if c in panel.columns
    ]

    # =========================
    # A/B feature sets
    # =========================
    feature_sets = {
        "A_CONTEXT_ONLY":
            a_cols + base_cols,

        "B_CONTEXT_TIME_ROTATION":
            b_cols + base_cols,

        "C_CONTEXT_PLUS_DYNAMICS":
            a_cols + dyn_cols + base_cols,
    }

    # 重複列削除
    for k in list(feature_sets.keys()):
        feature_sets[k] = list(dict.fromkeys(feature_sets[k]))

    panel.to_csv(
        OUT_DIR / "context_time_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== FEATURE SETS ===")
    for k, cols in feature_sets.items():
        print(f"{k}: {len(cols)} features")

    return panel, feature_sets


def make_sequences(code_df, feature_cols, hold_days):
    X, y, ret, dates, closes = [], [], [], [], []

    target_col = f"target_up_{hold_days}d"
    ret_col = f"ret_{hold_days}d"

    arr = code_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
    y_arr = code_df[target_col].values
    r_arr = code_df[ret_col].values
    d_arr = code_df["date"].values
    c_arr = code_df["close"].values

    for i in range(LOOKBACK_DAYS - 1, len(code_df)):
        if pd.isna(r_arr[i]):
            continue

        X.append(arr[i - LOOKBACK_DAYS + 1:i + 1])
        y.append(int(y_arr[i]))
        ret.append(float(r_arr[i]))
        dates.append(pd.to_datetime(d_arr[i]))
        closes.append(float(c_arr[i]))

    if not X:
        return None

    return {
        "X": np.stack(X),
        "y": np.array(y, dtype=np.int64),
        "ret": np.array(ret, dtype=np.float32),
        "date": np.array(dates),
        "close": np.array(closes, dtype=np.float32),
    }


def split_sequences(seq_data):
    dates = pd.to_datetime(seq_data["date"])
    unique_dates = sorted(pd.Series(dates).unique())

    n = len(unique_dates)
    train_end = unique_dates[int(n * 0.60)]
    valid_end = unique_dates[int(n * 0.80)]

    train_idx = dates < train_end
    valid_idx = (dates >= train_end) & (dates < valid_end)
    test_idx = dates >= valid_end

    def subset(mask):
        return {
            "X": seq_data["X"][mask],
            "y": seq_data["y"][mask],
            "ret": seq_data["ret"][mask],
            "date": seq_data["date"][mask],
            "close": seq_data["close"][mask],
        }

    return subset(train_idx), subset(valid_idx), subset(test_idx)


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
    wins = (pnls > 0).astype(int)

    return {
        "trades": int(len(pnls)),
        "winrate": float(wins.mean()),
        "sum_pnl": float(pnls.sum()),
        "avg_pnl": float(pnls.mean()),
        "pf": profit_factor(pnls.tolist()),
    }


def train_transformer(train, valid, input_dim):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TinyContextTransformer(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.15):
            super().__init__()

            self.proj = nn.Linear(input_dim, d_model)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=128,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )

            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            z = self.proj(x)
            z = self.encoder(z)
            last = z[:, -1, :]
            return self.head(last)

    model = TinyContextTransformer(input_dim=input_dim).to(device)

    X_train = torch.tensor(train["X"], dtype=torch.float32)
    y_train = torch.tensor(train["y"], dtype=torch.long)

    X_valid = torch.tensor(valid["X"], dtype=torch.float32).to(device)
    y_valid = torch.tensor(valid["y"], dtype=torch.long).to(device)

    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            valid_logits = model(X_valid)
            valid_loss = loss_fn(valid_logits, y_valid).item()

        print(
            f"  epoch={epoch:02d} "
            f"train_loss={np.mean(losses):.4f} "
            f"valid_loss={valid_loss:.4f}"
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, device


def predict_probs(model, device, X):
    import torch

    model.eval()
    probs = []

    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.tensor(X[i:i + BATCH_SIZE], dtype=torch.float32).to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(p)

    return np.concatenate(probs)


def run_ab(panel, feature_sets):
    results = []
    predictions = []

    for mode_name, features in feature_sets.items():
        print("\n==============================")
        print(f"MODE: {mode_name}")
        print(f"features: {len(features)}")
        print("==============================")

        for code in CODES:
            code_df = panel[panel["code"] == code].copy().sort_values("date")

            if len(code_df) < 150:
                continue

            for hold_days in HOLD_DAYS_LIST:
                print("\n------------------------------")
                print(f"{mode_name} code={code} hold={hold_days}")
                print("------------------------------")

                seq = make_sequences(code_df, features, hold_days)

                if seq is None or len(seq["X"]) < 100:
                    print("  skip: too few sequences")
                    continue

                train, valid, test = split_sequences(seq)

                if len(train["X"]) < 50 or len(valid["X"]) < 20 or len(test["X"]) < 20:
                    print("  skip: too few split rows")
                    continue

                input_dim = train["X"].shape[-1]

                model, device = train_transformer(train, valid, input_dim)

                valid_probs = predict_probs(model, device, valid["X"])
                test_probs = predict_probs(model, device, test["X"])

                best = None

                for thr in np.arange(0.50, 0.81, 0.05):
                    m = evaluate_probs(valid_probs, valid["ret"], thr)

                    if m["trades"] < 3:
                        continue

                    score = m["sum_pnl"] + m["pf"] * 0.01 + m["winrate"] * 0.01

                    if best is None or score > best["score"]:
                        best = {
                            "mode": mode_name,
                            "code": code,
                            "hold_days": hold_days,
                            "threshold": round(float(thr), 4),
                            "score": float(score),
                            **{f"valid_{k}": v for k, v in m.items()},
                        }

                if best is None:
                    print("  no valid threshold")
                    continue

                test_m = evaluate_probs(test_probs, test["ret"], best["threshold"])

                row = {
                    **best,
                    **{f"test_{k}": v for k, v in test_m.items()},
                    "train_rows": len(train["X"]),
                    "valid_rows": len(valid["X"]),
                    "test_rows": len(test["X"]),
                    "train_start": pd.to_datetime(train["date"][0]),
                    "train_end": pd.to_datetime(train["date"][-1]),
                    "valid_start": pd.to_datetime(valid["date"][0]),
                    "valid_end": pd.to_datetime(valid["date"][-1]),
                    "test_start": pd.to_datetime(test["date"][0]),
                    "test_end": pd.to_datetime(test["date"][-1]),
                }

                results.append(row)

                pred = pd.DataFrame(
                    {
                        "date": test["date"],
                        "mode": mode_name,
                        "code": code,
                        "hold_days": hold_days,
                        "close": test["close"],
                        "ret": test["ret"],
                        "target_up": test["y"],
                        "prob_up": test_probs,
                        "threshold": best["threshold"],
                    }
                )

                pred["signal"] = np.where(pred["prob_up"] >= best["threshold"], "BUY", "NO_TRADE")
                predictions.append(pred)

                print(
                    f"[BEST] {mode_name} {code} hold={hold_days} "
                    f"thr={best['threshold']} "
                    f"valid_trades={best['valid_trades']} "
                    f"valid_sum={best['valid_sum_pnl']:.4f} "
                    f"test_trades={test_m['trades']} "
                    f"test_sum={test_m['sum_pnl']:.4f}"
                )

    res = pd.DataFrame(results)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    res.to_csv(OUT_DIR / "context_time_rotation_results.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT_DIR / "context_time_rotation_predictions.csv", index=False, encoding="utf-8-sig")

    if not res.empty:
        summary = (
            res.groupby("mode")
            .agg(
                rows=("code", "count"),
                total_test_trades=("test_trades", "sum"),
                total_test_sum_pnl=("test_sum_pnl", "sum"),
                avg_test_winrate=("test_winrate", "mean"),
                avg_test_pf=("test_pf", "mean"),
            )
            .reset_index()
        )

        summary.to_csv(OUT_DIR / "context_time_rotation_summary.csv", index=False, encoding="utf-8-sig")

        print("\n=== A/B SUMMARY ===")
        print(summary.to_string(index=False))

    return res, pred


def write_readme_after_results(res):
    if res.empty:
        return

    summary = (
        res.groupby("mode")
        .agg(
            rows=("code", "count"),
            total_test_trades=("test_trades", "sum"),
            total_test_sum_pnl=("test_sum_pnl", "sum"),
            avg_test_winrate=("test_winrate", "mean"),
            avg_test_pf=("test_pf", "mean"),
        )
        .reset_index()
    )

    readme = "# JStockLLM: Context Time Rotation A/B Test\n\n"
    readme += "## 目的\n\n"
    readme += "LLM系Sentence Embeddingで得た文脈ベクトルに対して、発言時期をベクトル回転として織り込むと予測性能が上がるかを検証する。\n\n"

    readme += "## 比較\n\n"
    readme += "### A_CONTEXT_ONLY\n\n"
    readme += "```text\nニュース本文\n↓\nSentence Embedding\n↓\n過去10日系列\n↓\nTransformer\n```\n\n"

    readme += "### B_CONTEXT_TIME_ROTATION\n\n"
    readme += "```text\nニュース本文\n↓\nSentence Embedding\n↓\n発言時期でEmbeddingを回転\n↓\n過去10日系列\n↓\nTransformer\n```\n\n"

    readme += "## 時期回転の考え方\n\n"
    readme += "同じ文脈でも、いつ出たかによって市場での意味が変わるという仮説に基づく。\n\n"
    readme += "```text\n同じ「利上げ」でも\n決算期前\nFOMC前\n日銀会合前\nでは意味が変わる可能性がある\n```\n\n"

    readme += "## Summary\n\n"
    readme += summary.to_markdown(index=False)
    readme += "\n\n"

    readme += "## 考察テンプレ\n\n"
    readme += "- B が A を上回る場合：文脈ベクトルに時期情報を織り込む仮説は有効。\n"
    readme += "- A が B を上回る場合：回転がノイズになっている可能性。\n"
    readme += "- 両者が近い場合：ニュース件数不足、または時期情報がすでに文脈内に含まれている可能性。\n\n"

    readme += "## 次にやること\n\n"
    readme += "- ニュース件数を1000件以上に増やす\n"
    readme += "- TDnet/四季報/EDINETを追加\n"
    readme += "- 銘柄別に回転周期を最適化する\n"
    readme += "- 30日回転/90日回転/決算期回転を比較する\n"
    readme += "- 価格特徴量を足した最終モデルにする\n"

    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8-sig")


def main():
    print("=== Context Time Rotation A/B Test ===")

    prices = load_prices()
    events = load_events()
    daily_text = build_daily_text(events)
    emb = build_embeddings(daily_text)
    rot = add_time_rotation(emb)

    dyn = build_embedding_dynamics(emb)

    rhythm = build_text_rhythm(
        daily_text
    )

    panel, feature_sets = build_panel(
        prices,
        emb,
        rot
    )

    panel=panel.merge(
        dyn,
        on=["date","code"],
        how="left"
    )

    panel=panel.merge(
        rhythm,
        on=["date","code"],
        how="left"
    )
    print("\n=== DATA ===")
    print(f"prices     : {prices.shape}")
    print(f"events     : {events.shape}")
    print(f"daily_text : {daily_text.shape}")
    print(f"emb        : {emb.shape}")
    print(f"rot        : {rot.shape}")
    print(f"panel      : {panel.shape}")

    # =========================
    # FINAL FIX:
    # main側のmerge後に event_count が
    # event_count_x / event_count_y 等へ分裂した場合の修復
    # =========================
    event_cols = [
        c for c in panel.columns
        if c == "event_count" or c.startswith("event_count_")
    ]

    if event_cols:
        panel["event_count"] = (
            panel[event_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .max(axis=1)
        )

        panel = panel.drop(
            columns=[
                c for c in event_cols
                if c != "event_count"
            ],
            errors="ignore",
        )
    else:
        panel["event_count"] = 0

    # =========================
    # FINAL FEATURE SETS
    # build_panel後にdyn/rhythmをmergeした後の
    # 最新panel列から作り直す
    # =========================
    a_cols = [
        c for c in panel.columns
        if c.startswith("emb_")
    ]

    b_cols = [
        c for c in panel.columns
        if c.startswith("rot_emb_")
    ]

    dyn_cols = []

    for c in panel.columns:
        if c.startswith("vel_"):
            dyn_cols.append(c)
        elif c.startswith("acc_"):
            dyn_cols.append(c)
        elif c.startswith("jerk_"):
            dyn_cols.append(c)
        elif c in [
            "emb_energy",
            "emb_curvature",
            "emb_cos_prev",
            "char_len",
            "word_count",
            "comma_count",
            "period_count",
            "avg_word_len",
        ]:
            dyn_cols.append(c)

    dyn_cols = list(dict.fromkeys(dyn_cols))

    base_cols = ["event_count"]

    feature_sets = {
        "A_CONTEXT_ONLY":
            a_cols + base_cols,

        "B_CONTEXT_TIME_ROTATION":
            b_cols + base_cols,

        "C_CONTEXT_PLUS_DYNAMICS":
            a_cols + dyn_cols + base_cols,
    }

    for k in list(feature_sets.keys()):
        feature_sets[k] = [
            c for c in list(dict.fromkeys(feature_sets[k]))
            if c in panel.columns
        ]

    print("\n=== FINAL FEATURE SETS ===")
    for k, cols in feature_sets.items():
        print(f"{k}: {len(cols)} features")

    res, pred = run_ab(panel, feature_sets)
    write_readme_after_results(res)

    print("\n=== OUTPUTS ===")
    print(OUT_DIR / "context_time_rotation_results.csv")
    print(OUT_DIR / "context_time_rotation_predictions.csv")
    print(OUT_DIR / "context_time_rotation_summary.csv")
    print(OUT_DIR / "README.md")


if __name__ == "__main__":
    main()