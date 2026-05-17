# language_stack_ab_test.py
# A: word pairs
# B: word pairs + timing
# C: B + Word2Vec
# D: C + Sentence Embedding
# E: D + small Transformer-like sequence features
#
# Uses:
#   data/jquants_grid/prices_clean.csv
#   data/raw_news.csv
#   data/tdnet_history.csv
#
# Also collects:
#   GDELT recent news
#   yfinance ticker news

from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import hashlib
import itertools
import re
import time
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
JQ_DIR = DATA_DIR / "jquants_grid"
OUT_DIR = JQ_DIR / "language_stack_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_CSV = JQ_DIR / "prices_clean.csv"
RAW_NEWS_CSV = DATA_DIR / "raw_news.csv"
TDNET_CSV = DATA_DIR / "tdnet_history.csv"

JST = timezone(timedelta(hours=9))

CODES = {
    "72030": {
        "code4": "7203",
        "yf": "7203.T",
        "names": ["トヨタ", "トヨタ自動車", "Toyota", "TOYOTA"],
        "queries": ["トヨタ", "トヨタ自動車", "Toyota"],
    },
    "83060": {
        "code4": "8306",
        "yf": "8306.T",
        "names": ["三菱UFJ", "三菱ＵＦＪ", "MUFG", "Mitsubishi UFJ"],
        "queries": ["三菱UFJ", "MUFG", "Mitsubishi UFJ"],
    },
    "67580": {
        "code4": "6758",
        "yf": "6758.T",
        "names": ["ソニー", "ソニーグループ", "Sony", "SONY"],
        "queries": ["ソニーグループ", "ソニー", "Sony"],
    },
    "99840": {
        "code4": "9984",
        "yf": "9984.T",
        "names": ["ソフトバンクグループ", "ソフトバンクG", "SoftBank Group", "Arm", "ARM"],
        "queries": ["ソフトバンクグループ", "SoftBank Group", "Arm"],
    },
    "13210": {
        "code4": "1321",
        "yf": "1321.T",
        "names": ["日経平均", "日経225", "Nikkei", "Nikkei 225", "NEXT FUNDS"],
        "queries": ["日経平均", "日経225", "Nikkei 225"],
    },
    "15700": {
        "code4": "1570",
        "yf": "1570.T",
        "names": ["日経レバ", "日経平均レバレッジ", "1570"],
        "queries": ["日経レバ", "日経平均レバレッジ", "1570"],
    },
    "13570": {
        "code4": "1357",
        "yf": "1357.T",
        "names": ["日経ダブルインバース", "ダブルインバース", "1357"],
        "queries": ["日経ダブルインバース", "ダブルインバース", "1357"],
    },
}

HOLD_DAYS_LIST = [1, 5, 10]

SEED_WORDS = [
    "増配", "減配", "配当", "上方修正", "下方修正", "業績予想", "自社株買い",
    "決算", "営業利益", "純利益", "売上", "赤字", "黒字", "最高益",
    "円安", "円高", "為替", "ドル円", "金利", "利上げ", "利下げ", "日銀", "FRB",
    "AI", "人工知能", "生成AI", "半導体", "NVIDIA", "Arm", "ゲーム", "PS5",
    "投資", "評価益", "IPO", "NASDAQ", "米国株",
    "関税", "中国", "米国", "欧州", "輸出", "EV", "電池",
    "リスク", "地政学", "戦争", "紛争", "災害", "不祥事", "訴訟",
    "日経平均", "TOPIX", "先物", "リスクオフ", "リスクオン",
]

STOP_WORDS = {
    "これ", "それ", "ため", "よう", "こと", "もの", "ところ",
    "について", "として", "による", "により", "です", "ます",
    "the", "and", "for", "with", "from", "this", "that", "will",
}

MAX_DYNAMIC_WORDS_PER_DAY = 10
MIN_WORD_TOTAL_COUNT = 5
MAX_PAIR_FEATURES = 300
W2V_DIM = 16
SENT_EMB_DIM_REDUCED = 16
SEQ_LOOKBACK_DAYS = 10


def normalize_code(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 4:
        for c5, meta in CODES.items():
            if meta["code4"] == s:
                return c5
        return s + "0"
    return s


def make_id(*parts):
    text = "|".join(str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def clean_text(s):
    s = str(s or "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def safe_col(s):
    s = str(s)
    s = re.sub(r"[^0-9A-Za-z一-龥ァ-ンー]", "_", s)
    return s[:60]


def read_csv_safe(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def is_related(code5, text):
    t = str(text).lower()
    return any(name.lower() in t for name in CODES[code5]["names"])


def extract_words(text):
    text = clean_text(text)
    words = []

    for w in SEED_WORDS:
        cnt = text.count(w)
        if cnt > 0:
            words.extend([w] * cnt)

    rough = re.findall(r"[A-Za-z][A-Za-z0-9\-\+]{1,}|[一-龥ァ-ンー]{2,}", text)

    for w in rough:
        w = w.strip()
        if len(w) < 2:
            continue
        if len(w) > 20:
            continue
        if w.lower() in STOP_WORDS or w in STOP_WORDS:
            continue
        words.append(w)

    return words


def load_prices():
    if not PRICE_CSV.exists():
        raise FileNotFoundError(f"not found: {PRICE_CSV}")

    df = pd.read_csv(PRICE_CSV, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].apply(normalize_code)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["code"].isin(CODES.keys())].copy()
    df = df.dropna(subset=["date", "code", "close"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def gdelt_search(query, days_back=90, maxrecords=250):
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={quote(query)}"
        f"&mode=artlist"
        f"&format=json"
        f"&maxrecords={maxrecords}"
        f"&sort=hybridrel"
        f"&timespan={days_back}d"
    )

    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"  [GDELT WARN] status={r.status_code} query={query}")
        return []

    try:
        data = r.json()
    except Exception:
        return []

    return data.get("articles", []) or []


def parse_gdelt_date(item):
    s = item.get("seendate") or item.get("datetime") or ""
    try:
        dt = pd.to_datetime(s, utc=True)
        return dt.tz_convert("Asia/Tokyo").tz_localize(None)
    except Exception:
        return pd.Timestamp(datetime.now(JST).replace(tzinfo=None))


def collect_gdelt_news():
    rows = []
    seen = set()

    print("\n=== COLLECT: GDELT ===")

    for code5, meta in CODES.items():
        for q in meta["queries"]:
            print(f"[GDELT] {code5} query={q}")

            try:
                articles = gdelt_search(q, days_back=90, maxrecords=250)
                print(f"  articles: {len(articles)}")
            except Exception as e:
                print(f"  [ERROR] {type(e).__name__}: {e}")
                continue

            for item in articles:
                title = clean_text(item.get("title", ""))
                url = clean_text(item.get("url", ""))
                domain = clean_text(item.get("domain", ""))
                text = f"{title} {domain}"

                if not title:
                    continue
                if not is_related(code5, text):
                    continue

                dt = parse_gdelt_date(item)
                nid = make_id("gdelt", code5, title, url)

                if nid in seen:
                    continue
                seen.add(nid)

                rows.append({
                    "event_id": nid,
                    "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": dt.normalize().strftime("%Y-%m-%d"),
                    "code": code5,
                    "title": title,
                    "body": text,
                    "source": f"GDELT:{domain}",
                    "url": url,
                })

            time.sleep(1.0)

    return pd.DataFrame(rows)


def collect_yfinance_news():
    rows = []
    seen = set()

    print("\n=== COLLECT: yfinance ===")

    for code5, meta in CODES.items():
        print(f"[YF] {code5} {meta['yf']}")

        try:
            ticker = yf.Ticker(meta["yf"])
            news = ticker.get_news(count=50, tab="news")
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            continue

        print(f"  raw news: {len(news)}")

        for item in news:
            title = clean_text(item.get("title", ""))
            publisher = clean_text(item.get("publisher", ""))
            link = clean_text(item.get("link", ""))
            body = clean_text(item.get("summary", "") or item.get("description", "") or title)
            text = f"{title} {body} {publisher}"

            if not title:
                continue
            if not is_related(code5, text):
                continue

            ts = item.get("providerPublishTime")
            if ts:
                dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(JST).replace(tzinfo=None)
            else:
                dt = datetime.now(JST).replace(tzinfo=None)

            nid = make_id("yfinance", code5, title, publisher, link)

            if nid in seen:
                continue
            seen.add(nid)

            rows.append({
                "event_id": nid,
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "date": pd.Timestamp(dt).normalize().strftime("%Y-%m-%d"),
                "code": code5,
                "title": title,
                "body": body,
                "source": f"yfinance:{publisher}",
                "url": link,
            })

        time.sleep(0.5)

    return pd.DataFrame(rows)


def load_local_news():
    rows = []

    print("\n=== LOAD: local CSV ===")

    raw = read_csv_safe(RAW_NEWS_CSV)
    if not raw.empty:
        print(f"[LOCAL] raw_news rows={len(raw)}")
        for _, r in raw.iterrows():
            dt = pd.to_datetime(r.get("datetime", ""), errors="coerce")
            code = normalize_code(r.get("code", ""))
            if pd.isna(dt) or code not in CODES:
                continue

            title = clean_text(r.get("title", ""))
            body = clean_text(r.get("body", ""))
            source = clean_text(r.get("source", "raw_news"))
            url = clean_text(r.get("url", ""))

            rows.append({
                "event_id": make_id("raw", code, dt, title, url),
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "date": dt.normalize().strftime("%Y-%m-%d"),
                "code": code,
                "title": title,
                "body": body,
                "source": source,
                "url": url,
            })

    td = read_csv_safe(TDNET_CSV)
    if not td.empty:
        print(f"[LOCAL] tdnet rows={len(td)}")
        for _, r in td.iterrows():
            dt = pd.to_datetime(r.get("datetime", ""), errors="coerce")
            code = normalize_code(r.get("code", ""))
            if pd.isna(dt) or code not in CODES:
                continue

            title = clean_text(r.get("title", ""))
            body = clean_text(r.get("company_hint", ""))
            source = "TDnet"
            url = clean_text(r.get("pdf_url", ""))

            rows.append({
                "event_id": make_id("tdnet", code, dt, title, url),
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "date": dt.normalize().strftime("%Y-%m-%d"),
                "code": code,
                "title": title,
                "body": body,
                "source": source,
                "url": url,
            })

    return pd.DataFrame(rows)


def collect_all_language_events():
    parts = []

    local = load_local_news()
    if not local.empty:
        parts.append(local)

    gdelt = collect_gdelt_news()
    if not gdelt.empty:
        parts.append(gdelt)

    yf = collect_yfinance_news()
    if not yf.empty:
        parts.append(yf)

    if not parts:
        raise RuntimeError("ニュース/開示テキストが0件です。")

    ev = pd.concat(parts, ignore_index=True)
    ev["date"] = pd.to_datetime(ev["date"])
    ev["code"] = ev["code"].apply(normalize_code)
    ev = ev[ev["code"].isin(CODES.keys())].copy()
    ev = ev.drop_duplicates(subset=["event_id"])
    ev = ev.sort_values(["code", "date", "source"]).reset_index(drop=True)

    ev["full_text"] = (ev["title"].astype(str) + " " + ev["body"].astype(str)).map(clean_text)
    ev["tokens"] = ev["full_text"].map(extract_words)

    ev.to_csv(OUT_DIR / "language_events_collected.csv", index=False, encoding="utf-8-sig")

    print("\n=== COLLECTED LANGUAGE EVENTS ===")
    print(f"events: {len(ev)}")
    print(ev.groupby("code").size().to_string())

    return ev


def select_vocab(events):
    counts = {}

    for toks in events["tokens"]:
        for w in toks:
            counts[w] = counts.get(w, 0) + 1

    vocab = set(SEED_WORDS)
    for w, c in counts.items():
        if c >= MIN_WORD_TOTAL_COUNT:
            vocab.add(w)

    vocab = sorted(vocab)
    pd.DataFrame([{"word": w, "count": counts.get(w, 0)} for w in vocab]).to_csv(
        OUT_DIR / "vocab.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\n[VOCAB] words={len(vocab)}")
    return vocab


def build_pair_features(events, vocab):
    print("\n=== BUILD: word pairs ===")

    vocab_set = set(vocab)
    pair_counter = {}

    event_tokens = []

    for _, r in events.iterrows():
        toks = [w for w in r["tokens"] if w in vocab_set]
        toks = list(dict.fromkeys(toks))
        event_tokens.append(toks)

        for a, b in itertools.combinations(sorted(toks), 2):
            key = f"{safe_col(a)}__{safe_col(b)}"
            pair_counter[key] = pair_counter.get(key, 0) + 1

    top_pairs = sorted(pair_counter.items(), key=lambda x: x[1], reverse=True)[:MAX_PAIR_FEATURES]
    pair_names = [p for p, _ in top_pairs]

    pd.DataFrame(top_pairs, columns=["pair", "count"]).to_csv(
        OUT_DIR / "token_pairs.csv", index=False, encoding="utf-8-sig"
    )

    rows = []

    for i, (_, r) in enumerate(events.iterrows()):
        date = pd.to_datetime(r["date"]).normalize()
        code = r["code"]
        toks = [w for w in event_tokens[i] if w in vocab_set]
        pairs = set()

        for a, b in itertools.combinations(sorted(toks), 2):
            pairs.add(f"{safe_col(a)}__{safe_col(b)}")

        row = {
            "date": date,
            "code": code,
            "event_count": 1,
            "text_len": len(r["full_text"]),
        }

        for p in pair_names:
            row[f"pair_{p}"] = 1 if p in pairs else 0

        rows.append(row)

    daily = pd.DataFrame(rows)

    if daily.empty:
        raise RuntimeError("pair features are empty")

    pair_cols = [c for c in daily.columns if c.startswith("pair_")]

    daily = daily.groupby(["date", "code"]).agg({
        "event_count": "sum",
        "text_len": "sum",
        **{c: "sum" for c in pair_cols},
    }).reset_index()

    daily.to_csv(OUT_DIR / "daily_pair_features.csv", index=False, encoding="utf-8-sig")

    print(f"[PAIR] daily rows={len(daily)} pair cols={len(pair_cols)}")
    return daily, pair_cols


def add_pair_timing_features(df, pair_cols):
    print("\n=== ADD: pair timing features ===")

    df = df.sort_values(["code", "date"]).copy()

    for c in pair_cols:
        df[f"{c}_roll5"] = (
            df.groupby("code")[c]
            .rolling(5, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        df[f"{c}_roll20"] = (
            df.groupby("code")[c]
            .rolling(20, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        mean60 = (
            df.groupby("code")[c]
            .rolling(60, min_periods=10)
            .mean()
            .reset_index(level=0, drop=True)
        )

        std60 = (
            df.groupby("code")[c]
            .rolling(60, min_periods=10)
            .std()
            .reset_index(level=0, drop=True)
        )

        df[f"{c}_z60"] = ((df[c] - mean60) / (std60 + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)

        all_days_since = []
        for _, sub in df.groupby("code"):
            counter = 999
            arr = []
            for x in sub[c].fillna(0).tolist():
                if x > 0:
                    counter = 0
                else:
                    counter += 1
                arr.append(counter)
            all_days_since.extend(arr)

        df[f"{c}_days_since"] = all_days_since
        df[f"{c}_since_sin30"] = np.sin(2 * np.pi * df[f"{c}_days_since"] / 30)
        df[f"{c}_since_cos30"] = np.cos(2 * np.pi * df[f"{c}_days_since"] / 30)

    return df


def build_word2vec_features(events, prices):
    print("\n=== BUILD: Word2Vec features ===")

    try:
        from gensim.models import Word2Vec
    except Exception as e:
        print(f"[WARN] gensim unavailable: {e}")
        return pd.DataFrame()

    sentences = []
    for toks in events["tokens"]:
        toks = [w for w in toks if w not in STOP_WORDS]
        if len(toks) >= 2:
            sentences.append(toks)

    if len(sentences) < 5:
        print("[WARN] not enough sentences for Word2Vec")
        return pd.DataFrame()

    model = Word2Vec(
        sentences=sentences,
        vector_size=W2V_DIM,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        epochs=30,
        seed=42,
    )

    rows = []

    for (date, code), g in events.groupby(["date", "code"]):
        vecs = []
        for toks in g["tokens"]:
            for w in toks:
                if w in model.wv:
                    vecs.append(model.wv[w])

        if vecs:
            arr = np.mean(np.vstack(vecs), axis=0)
        else:
            arr = np.zeros(W2V_DIM)

        row = {
            "date": pd.to_datetime(date).normalize(),
            "code": code,
        }
        for j in range(W2V_DIM):
            row[f"w2v_{j}"] = float(arr[j])
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "word2vec_features.csv", index=False, encoding="utf-8-sig")
    print(f"[W2V] rows={len(out)} dim={W2V_DIM}")
    return out


def build_sentence_embedding_features(events):
    print("\n=== BUILD: Sentence Embedding features ===")

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
    except Exception as e:
        print(f"[WARN] sentence-transformers unavailable: {e}")
        return pd.DataFrame()

    texts = events["full_text"].astype(str).tolist()

    if len(texts) < 10:
        print("[WARN] not enough texts for sentence embeddings")
        return pd.DataFrame()

    try:
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        emb = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    except Exception as e:
        print(f"[WARN] embedding failed: {e}")
        return pd.DataFrame()

    n_comp = min(SENT_EMB_DIM_REDUCED, emb.shape[0] - 1, emb.shape[1])
    if n_comp < 2:
        return pd.DataFrame()

    pca = PCA(n_components=n_comp, random_state=42)
    emb_red = pca.fit_transform(emb)

    tmp = events[["date", "code"]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()

    for j in range(n_comp):
        tmp[f"sent_emb_{j}"] = emb_red[:, j]

    emb_cols = [c for c in tmp.columns if c.startswith("sent_emb_")]
    out = tmp.groupby(["date", "code"])[emb_cols].mean().reset_index()

    out.to_csv(OUT_DIR / "sentence_embedding_features.csv", index=False, encoding="utf-8-sig")
    print(f"[SENT EMB] rows={len(out)} dim={len(emb_cols)}")
    return out


def add_sequence_features(df, base_feature_cols):
    print("\n=== BUILD: Small Transformer-like sequence features ===")

    df = df.sort_values(["code", "date"]).copy()

    use_cols = base_feature_cols[: min(120, len(base_feature_cols))]
    if not use_cols:
        return df, []

    seq_cols = []

    for c in use_cols:
        df[f"seq_{c}_mean{SEQ_LOOKBACK_DAYS}"] = (
            df.groupby("code")[c]
            .rolling(SEQ_LOOKBACK_DAYS, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"seq_{c}_std{SEQ_LOOKBACK_DAYS}"] = (
            df.groupby("code")[c]
            .rolling(SEQ_LOOKBACK_DAYS, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

        # attention風: 直近ほど重い加重平均
        weights = np.arange(1, SEQ_LOOKBACK_DAYS + 1, dtype=float)
        weights = weights / weights.sum()

        def weighted_last(x):
            arr = np.asarray(x, dtype=float)
            w = weights[-len(arr):]
            w = w / w.sum()
            return float(np.sum(arr * w))

        df[f"seq_{c}_attn{SEQ_LOOKBACK_DAYS}"] = (
            df.groupby("code")[c]
            .rolling(SEQ_LOOKBACK_DAYS, min_periods=1)
            .apply(weighted_last, raw=True)
            .reset_index(level=0, drop=True)
        )

        seq_cols.extend([
            f"seq_{c}_mean{SEQ_LOOKBACK_DAYS}",
            f"seq_{c}_std{SEQ_LOOKBACK_DAYS}",
            f"seq_{c}_attn{SEQ_LOOKBACK_DAYS}",
        ])

    return df, seq_cols


def add_labels(df):
    df = df.sort_values(["code", "date"]).copy()

    for h in HOLD_DAYS_LIST:
        df[f"future_close_{h}d"] = df.groupby("code")["close"].shift(-h)
        df[f"ret_{h}d"] = (df[f"future_close_{h}d"] - df["close"]) / df["close"]
        df[f"target_up_{h}d"] = (df[f"ret_{h}d"] > 0).astype(int)

    return df


def build_full_dataset(prices, pair_daily, pair_cols, w2v, sent_emb):
    df = prices.copy()
    df["code"] = df["code"].apply(normalize_code)

    pair_daily = pair_daily.copy()
    pair_daily["date"] = pd.to_datetime(pair_daily["date"])
    pair_daily["code"] = pair_daily["code"].apply(normalize_code)

    df = df.merge(pair_daily, on=["date", "code"], how="left")

    for c in ["event_count", "text_len"] + pair_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = add_pair_timing_features(df, pair_cols)

    if not w2v.empty:
        w2v = w2v.copy()
        w2v["date"] = pd.to_datetime(w2v["date"])
        w2v["code"] = w2v["code"].apply(normalize_code)
        df = df.merge(w2v, on=["date", "code"], how="left")

    if not sent_emb.empty:
        sent_emb = sent_emb.copy()
        sent_emb["date"] = pd.to_datetime(sent_emb["date"])
        sent_emb["code"] = sent_emb["code"].apply(normalize_code)
        df = df.merge(sent_emb, on=["date", "code"], how="left")

    for c in df.columns:
        if c.startswith("w2v_") or c.startswith("sent_emb_"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    pair_timing_cols = [
        c for c in df.columns
        if c.startswith("pair_") and c not in pair_cols
    ]

    w2v_cols = [c for c in df.columns if c.startswith("w2v_")]
    sent_cols = [c for c in df.columns if c.startswith("sent_emb_")]

    d_cols = ["event_count", "text_len"] + pair_cols + pair_timing_cols + w2v_cols + sent_cols
    d_cols = [c for c in d_cols if c in df.columns]

    df, seq_cols = add_sequence_features(df, d_cols)

    df = add_labels(df)

    df.to_csv(OUT_DIR / "language_stack_dataset.csv", index=False, encoding="utf-8-sig")

    feature_groups = {
        "A_PAIR_ONLY": ["event_count", "text_len"] + pair_cols,
        "B_PAIR_AND_TIMING": ["event_count", "text_len"] + pair_cols + pair_timing_cols,
        "C_B_PLUS_WORD2VEC": ["event_count", "text_len"] + pair_cols + pair_timing_cols + w2v_cols,
        "D_C_PLUS_SENTENCE_EMB": ["event_count", "text_len"] + pair_cols + pair_timing_cols + w2v_cols + sent_cols,
        "E_D_PLUS_SMALL_TRANSFORMER": ["event_count", "text_len"] + pair_cols + pair_timing_cols + w2v_cols + sent_cols + seq_cols,
    }

    for k, cols in feature_groups.items():
        feature_groups[k] = [c for c in cols if c in df.columns]

    print("\n=== FEATURE GROUPS ===")
    for k, cols in feature_groups.items():
        print(f"{k}: {len(cols)}")

    return df, feature_groups


def get_model():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
        )


def split_by_time(df):
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 120:
        raise RuntimeError("データ日数が少なすぎます。")

    n = len(dates)
    train_end = pd.to_datetime(dates[int(n * 0.60)])
    valid_end = pd.to_datetime(dates[int(n * 0.80)])

    train = df[df["date"] < train_end].copy()
    valid = df[(df["date"] >= train_end) & (df["date"] < valid_end)].copy()
    test = df[df["date"] >= valid_end].copy()

    return train, valid, test


def profit_factor(pnls):
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    if not losses:
        return 999.0 if wins else 0.0
    return sum(wins) / abs(sum(losses))


def evaluate(df, prob_col, ret_col, thr):
    d = df.dropna(subset=[prob_col, ret_col]).copy()
    trades = d[d[prob_col] >= thr].copy()

    if trades.empty:
        return {"trades": 0, "winrate": 0, "sum_pnl": 0, "avg_pnl": 0, "pf": 0}

    pnls = trades[ret_col].astype(float).tolist()
    wins = [1 if p > 0 else 0 for p in pnls]

    return {
        "trades": len(pnls),
        "winrate": float(np.mean(wins)),
        "sum_pnl": float(np.sum(pnls)),
        "avg_pnl": float(np.mean(pnls)),
        "pf": float(profit_factor(pnls)),
    }


def run_stack_ab(df, feature_groups):
    results = []
    predictions = []

    for ab_name, features in feature_groups.items():
        print("\n==============================")
        print(f"A/B/C/D/E TEST: {ab_name}")
        print(f"features: {len(features)}")
        print("==============================")

        if not features:
            continue

        for code in CODES.keys():
            code_df = df[df["code"] == code].copy()

            if len(code_df) < 120:
                continue

            for hold_days in HOLD_DAYS_LIST:
                target_col = f"target_up_{hold_days}d"
                ret_col = f"ret_{hold_days}d"

                sub = code_df.dropna(subset=[target_col, ret_col]).copy()
                if sub[target_col].nunique() < 2:
                    continue

                train, valid, test = split_by_time(sub)

                if len(train) < 40 or len(valid) < 10 or len(test) < 10:
                    continue

                X_train = train[features].replace([np.inf, -np.inf], 0).fillna(0)
                y_train = train[target_col].astype(int)
                X_valid = valid[features].replace([np.inf, -np.inf], 0).fillna(0)
                X_test = test[features].replace([np.inf, -np.inf], 0).fillna(0)

                if X_train.sum().sum() == 0:
                    continue

                model = get_model()
                model.fit(X_train, y_train)

                valid_eval = valid.copy()
                test_eval = test.copy()

                valid_eval["prob_up"] = model.predict_proba(X_valid)[:, 1]
                test_eval["prob_up"] = model.predict_proba(X_test)[:, 1]

                best = None

                for thr in np.arange(0.50, 0.81, 0.05):
                    m = evaluate(valid_eval, "prob_up", ret_col, thr)
                    if m["trades"] < 3:
                        continue

                    score = m["sum_pnl"] + m["pf"] * 0.01 + m["winrate"] * 0.01

                    if best is None or score > best["score"]:
                        best = {
                            "stack": ab_name,
                            "code": code,
                            "hold_days": hold_days,
                            "threshold": round(float(thr), 4),
                            "score": float(score),
                            "feature_count": len(features),
                            **{f"valid_{k}": v for k, v in m.items()},
                        }

                if best is None:
                    continue

                test_m = evaluate(test_eval, "prob_up", ret_col, best["threshold"])

                row = {
                    **best,
                    **{f"test_{k}": v for k, v in test_m.items()},
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "test_rows": len(test),
                    "train_start": train["date"].min(),
                    "train_end": train["date"].max(),
                    "valid_start": valid["date"].min(),
                    "valid_end": valid["date"].max(),
                    "test_start": test["date"].min(),
                    "test_end": test["date"].max(),
                }

                results.append(row)

                pred = test_eval[["date", "code", "close", ret_col, target_col]].copy()
                pred["stack"] = ab_name
                pred["hold_days"] = hold_days
                pred["prob_up"] = test_eval["prob_up"]
                pred["threshold"] = best["threshold"]
                pred["signal"] = np.where(pred["prob_up"] >= best["threshold"], "BUY", "NO_TRADE")
                predictions.append(pred)

                print(
                    f"[BEST] {ab_name} {code} hold={hold_days} "
                    f"thr={best['threshold']} "
                    f"valid_trades={best['valid_trades']} "
                    f"valid_sum={best['valid_sum_pnl']:.4f} "
                    f"test_trades={test_m['trades']} "
                    f"test_sum={test_m['sum_pnl']:.4f}"
                )

    res = pd.DataFrame(results)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    res.to_csv(OUT_DIR / "stack_ab_results.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT_DIR / "stack_ab_predictions.csv", index=False, encoding="utf-8-sig")

    if not res.empty:
        summary = (
            res.groupby("stack")
            .agg(
                rows=("code", "count"),
                total_test_trades=("test_trades", "sum"),
                total_test_sum_pnl=("test_sum_pnl", "sum"),
                avg_test_winrate=("test_winrate", "mean"),
                avg_test_pf=("test_pf", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(OUT_DIR / "stack_ab_summary.csv", index=False, encoding="utf-8-sig")

        print("\n=== STACK A/B/C/D/E SUMMARY ===")
        print(summary.to_string(index=False))

        print("\n=== BEST BY STACK/CODE ===")
        best_by = (
            res.sort_values("test_sum_pnl", ascending=False)
            .groupby(["stack", "code"])
            .head(1)
            .sort_values(["code", "stack"])
        )
        show_cols = ["stack", "code", "hold_days", "threshold", "test_trades", "test_winrate", "test_sum_pnl", "test_pf"]
        print(best_by[show_cols].to_string(index=False))

    return res, pred


def main():
    print("=== Language Stack A/B/C/D/E Test ===")

    prices = load_prices()
    events = collect_all_language_events()

    if len(events) < 500:
        print("\n[WARN] events are still small.")
        print("A/B is runnable, but C/D/E are reference-level until you collect more news.")

    vocab = select_vocab(events)
    pair_daily, pair_cols = build_pair_features(events, vocab)
    w2v = build_word2vec_features(events, prices)
    sent_emb = build_sentence_embedding_features(events)

    dataset, feature_groups = build_full_dataset(prices, pair_daily, pair_cols, w2v, sent_emb)

    print("\n=== DATA ===")
    print(f"prices   : {prices.shape}")
    print(f"events   : {events.shape}")
    print(f"pairs    : {pair_daily.shape}")
    print(f"w2v      : {w2v.shape}")
    print(f"sent_emb : {sent_emb.shape}")
    print(f"dataset  : {dataset.shape}")
    print(f"dataset output: {OUT_DIR / 'language_stack_dataset.csv'}")

    run_stack_ab(dataset, feature_groups)

    print("\n=== OUTPUTS ===")
    print(f"{OUT_DIR / 'language_events_collected.csv'}")
    print(f"{OUT_DIR / 'token_pairs.csv'}")
    print(f"{OUT_DIR / 'word2vec_features.csv'}")
    print(f"{OUT_DIR / 'sentence_embedding_features.csv'}")
    print(f"{OUT_DIR / 'language_stack_dataset.csv'}")
    print(f"{OUT_DIR / 'stack_ab_results.csv'}")
    print(f"{OUT_DIR / 'stack_ab_summary.csv'}")


if __name__ == "__main__":
    main()