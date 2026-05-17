# JStockLLM: Context Time Rotation A/B Test

## 目的

LLM系Sentence Embeddingで得た文脈ベクトルに対して、発言時期をベクトル回転として織り込むと予測性能が上がるかを検証する。

## 比較

### A_CONTEXT_ONLY

```text
ニュース本文
↓
Sentence Embedding
↓
過去10日系列
↓
Transformer
```

### B_CONTEXT_TIME_ROTATION

```text
ニュース本文
↓
Sentence Embedding
↓
発言時期でEmbeddingを回転
↓
過去10日系列
↓
Transformer
```

## 時期回転の考え方

同じ文脈でも、いつ出たかによって市場での意味が変わるという仮説に基づく。

```text
同じ「利上げ」でも
決算期前
FOMC前
日銀会合前
では意味が変わる可能性がある
```

## Summary

| mode                    |   rows |   total_test_trades |   total_test_sum_pnl |   avg_test_winrate |   avg_test_pf |
|:------------------------|-------:|--------------------:|---------------------:|-------------------:|--------------:|
| A_CONTEXT_ONLY          |     15 |                1423 |              15.5378 |           0.561852 |       2.48966 |
| B_CONTEXT_TIME_ROTATION |     14 |                1329 |              12.9664 |           0.554872 |       2.34827 |

## 考察テンプレ

- B が A を上回る場合：文脈ベクトルに時期情報を織り込む仮説は有効。
- A が B を上回る場合：回転がノイズになっている可能性。
- 両者が近い場合：ニュース件数不足、または時期情報がすでに文脈内に含まれている可能性。

## 次にやること

- ニュース件数を1000件以上に増やす
- TDnet/四季報/EDINETを追加
- 銘柄別に回転周期を最適化する
- 30日回転/90日回転/決算期回転を比較する
- 価格特徴量を足した最終モデルにする
