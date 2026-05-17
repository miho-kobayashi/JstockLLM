# transformer_news_model.py
# News text -> sentence embedding -> sequence Transformer -> prob_up -> grid search
#
# install:
#   pip install pandas numpy torch sentence-transformers scikit-learn
#
# run:
#   cd C:\Tools\JStockLLM
#   python transformer_news_model.py

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
JQ_DIR = DATA_DIR / "jquants_grid"
LANG_DIR = JQ_DIR / "language_stack_ab"
OUT_DIR = JQ_DIR / "transformer_news"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_CSV = JQ_DIR / "prices_clean.csv"
EVENTS_CSV = LANG_DIR / "language_events_collected.csv"

CODES = ["72030", "83060", "67580", "99840", "13210", "15700", "13570"]

HOLD_DAYS_LIST = [1, 5, 10]
LOOKBACK_DAYS = 10

EMB_DIM_REDUCED = 32
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
        ev["full_text"] = ev["title"].astype(str) + " " + ev["body"].astype(str)

    ev = ev.dropna(subset=["date"])
    ev = ev[ev["code"].isin(CODES)].copy()
    ev["full_text"] = ev["full_text"].astype(str).str.replace(r"\s+", " ", regex=True)
    ev = ev.sort_values(["code", "date"]).reset_index(drop=True)
    return ev


def build_daily_text(events):
    rows = []

    for (date, code), g in events.groupby(["date", "code"]):
        text = "。".join(g["full_text"].astype(str).tolist())
        rows.append({
            "date": pd.to_datetime(date).normalize(),
            "code": code,
            "event_count": len(g),
            "text": text,
        })

    daily = pd.DataFrame(rows)
    daily.to_csv(OUT_DIR / "daily_text.csv", index=False, encoding="utf-8-sig")
    return daily


def build_embeddings(daily_text):
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    print("\n=== BUILD EMBEDDINGS ===")

    texts = daily_text["text"].astype(str).tolist()
    if len(texts) < 5:
        raise RuntimeError("ニュース日数が少なすぎます。")

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    n_comp = min(EMB_DIM_REDUCED, emb.shape[0] - 1, emb.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    emb_red = pca.fit_transform(emb)

    out = daily_text[["date", "code", "event_count"]].copy()

    for j in range(n_comp):
        out[f"emb_{j}"] = emb_red[:, j]

    out.to_csv(OUT_DIR / "daily_embeddings.csv", index=False, encoding="utf-8-sig")

    print(f"embedding rows: {len(out)}")
    print(f"embedding dim : {n_comp}")
    return out


def add_labels(prices):
    df = prices.sort_values(["code", "date"]).copy()

    for h in HOLD_DAYS_LIST:
        df[f"future_close_{h}d"] = df.groupby("code")["close"].shift(-h)
        df[f"ret_{h}d"] = (df[f"future_close_{h}d"] - df["close"]) / df["close"]
        df[f"target_up_{h}d"] = (df[f"ret_{h}d"] > 0).astype(int)

    return df


def build_panel(prices, emb_df):
    prices = add_labels(prices)

    emb_df = emb_df.copy()
    emb_df["date"] = pd.to_datetime(emb_df["date"])
    emb_df["code"] = emb_df["code"].apply(normalize_code)

    df = prices.merge(emb_df, on=["date", "code"], how="left")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    for c in emb_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["event_count"] = pd.to_numeric(df.get("event_count", 0), errors="coerce").fillna(0)

    # 価格補助特徴量も少しだけ入れる
    df["ret_1"] = df.groupby("code")["close"].pct_change(1).fillna(0)
    df["ret_5"] = df.groupby("code")["close"].pct_change(5).fillna(0)
    df["vol_chg_5"] = df.groupby("code")["volume"].pct_change(5).replace([np.inf, -np.inf], 0).fillna(0)

    feature_cols = emb_cols + ["event_count", "ret_1", "ret_5", "vol_chg_5"]

    df.to_csv(OUT_DIR / "transformer_panel.csv", index=False, encoding="utf-8-sig")
    return df, feature_cols


def make_sequences(code_df, feature_cols, hold_days):
    X = []
    y = []
    ret = []
    dates = []
    closes = []

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

        seq = arr[i - LOOKBACK_DAYS + 1:i + 1]
        X.append(seq)
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

    class TinyNewsTransformer(nn.Module):
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

    model = TinyNewsTransformer(input_dim=input_dim).to(device)

    X_train = torch.tensor(train["X"], dtype=torch.float32)
    y_train = torch.tensor(train["y"], dtype=torch.long)

    X_valid = torch.tensor(valid["X"], dtype=torch.float32).to(device)
    y_valid = torch.tensor(valid["y"], dtype=torch.long).to(device)

    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            valid_logits = model(X_valid)
            valid_loss = loss_fn(valid_logits, y_valid).item()

        print(
            f"  epoch={epoch:02d} "
            f"train_loss={np.mean(train_losses):.4f} "
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


def run_transformer_grid(panel, feature_cols):
    results = []
    predictions = []

    for code in CODES:
        code_df = panel[panel["code"] == code].copy().sort_values("date")

        if len(code_df) < 150:
            continue

        for hold_days in HOLD_DAYS_LIST:
            print("\n==============================")
            print(f"TRANSFORMER code={code} hold={hold_days}")
            print("==============================")

            seq = make_sequences(code_df, feature_cols, hold_days)
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
            for thr in THRESHOLDS:
                m = evaluate_probs(valid_probs, valid["ret"], thr)
                if m["trades"] < 3:
                    continue

                score = m["sum_pnl"] + m["pf"] * 0.01 + m["winrate"] * 0.01

                if best is None or score > best["score"]:
                    best = {
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

            pred = pd.DataFrame({
                "date": test["date"],
                "code": code,
                "hold_days": hold_days,
                "close": test["close"],
                "ret": test["ret"],
                "target_up": test["y"],
                "prob_up": test_probs,
                "threshold": best["threshold"],
            })
            pred["signal"] = np.where(pred["prob_up"] >= best["threshold"], "BUY", "NO_TRADE")
            predictions.append(pred)

            print(
                f"[BEST TRANSFORMER] {code} hold={hold_days} "
                f"thr={best['threshold']} "
                f"valid_trades={best['valid_trades']} "
                f"valid_sum={best['valid_sum_pnl']:.4f} "
                f"test_trades={test_m['trades']} "
                f"test_sum={test_m['sum_pnl']:.4f}"
            )

    res = pd.DataFrame(results)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    res.to_csv(OUT_DIR / "transformer_grid_results.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT_DIR / "transformer_predictions.csv", index=False, encoding="utf-8-sig")

    if not res.empty:
        summary = (
            res.agg(
                total_test_trades=("test_trades", "sum"),
                total_test_sum_pnl=("test_sum_pnl", "sum"),
                avg_test_winrate=("test_winrate", "mean"),
                avg_test_pf=("test_pf", "mean"),
            )
        )

        print("\n=== TRANSFORMER RESULTS ===")
        print(res.sort_values("test_sum_pnl", ascending=False).to_string(index=False))

    return res, pred


def main():
    print("=== Transformer News Model ===")

    prices = load_prices()
    events = load_events()
    daily_text = build_daily_text(events)
    emb = build_embeddings(daily_text)
    panel, feature_cols = build_panel(prices, emb)

    print("\n=== DATA ===")
    print(f"prices      : {prices.shape}")
    print(f"events      : {events.shape}")
    print(f"daily_text  : {daily_text.shape}")
    print(f"embeddings  : {emb.shape}")
    print(f"panel       : {panel.shape}")
    print(f"feature_dim : {len(feature_cols)}")

    if len(events) < 500:
        print("\n[WARN] ニュース件数が少ないです。")
        print("Transformerは動きますが、結果はまだ参考値です。1000件以上ほしいです。")

    res, pred = run_transformer_grid(panel, feature_cols)

    print("\n=== OUTPUTS ===")
    print(OUT_DIR / "daily_text.csv")
    print(OUT_DIR / "daily_embeddings.csv")
    print(OUT_DIR / "transformer_panel.csv")
    print(OUT_DIR / "transformer_grid_results.csv")
    print(OUT_DIR / "transformer_predictions.csv")


if __name__ == "__main__":
    main()