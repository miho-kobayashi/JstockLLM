# JStockLLM
## Context Dynamics A/B Test
国内株ニュースに対して

```text
意味(LLM)
+
時間
+
文章の運動
```

を加えると予測性能が上がるか検証。

対象：

- 7203 トヨタ
- 8306 三菱UFJ
- 6758 ソニーG
- 9984 ソフトバンクG
- 1321 日経225ETF
- 1570 日経レバETF
- 1357 ダブルインバース

期間：

約2年

価格：

3416日

ニュース：

197件

---

# 背景

相場では以前から

```text
価格
↓
速度
↓
加速度
↓
ジャーク
```

を見ると予測性能が上がる仮説を使用。

今回の仮説：

文章も同様に

```text
意味
↓
速度
↓
加速度
↓
ジャーク
```

を持つのでは？

---

# 仮説

同じニュースでも

```text
利上げ
```

は

2024:

```text
銀行株↑
```

2025:

```text
景気悪化懸念
```

になる可能性。

つまり：

```text
意味
=
文章
+
時間
+
運動
```

---

# 実験

比較：

---

## A_CONTEXT_ONLY

通常LLM

```text
ニュース
↓
Sentence Embedding
↓
Transformer
↓
BUY判定
```

特徴量：

```text
Embedding
+
event_count
```

結果：

33 features

---

## B_CONTEXT_TIME_ROTATION

意味ベクトルを時期で回転

仮説：

```text
意味ベクトルは
時間で方向が変わる
```

実装：

```text
Embedding
↓
sin/cos周期回転
↓
Transformer
```

特徴量：

33 features

---

## C_CONTEXT_PLUS_DYNAMICS

文脈を力学系として扱う

追加：

速度

```text
v(t)
=
embedding_t
-
embedding_(t-1)
```

加速度

```text
a(t)
=
v_t
-
v_(t-1)
```

ジャーク

```text
j(t)
=
a_t
-
a_(t-1)
```

追加特徴：

```text
Embedding速度
Embedding加速度
Embeddingジャーク
文長
単語数
句読点数
エネルギー
曲率
```

特徴量：

33+力学特徴

---

# 結果

## SUMMARY

A_CONTEXT_ONLY

```text
trades      = 1425
sum pnl     = +12.74
winrate     = 54.4%
PF          = 2.24
```

---

B_CONTEXT_TIME_ROTATION

```text
trades      = 1423
sum pnl     = +15.54
winrate     = 56.2%
PF          = 2.49
```

---

C_CONTEXT_PLUS_DYNAMICS

```text
trades      = 1422
sum pnl     = +15.56
winrate     = 56.2%
PF          = 2.49
```

BEST

---

# 順位

1位

```text
C_CONTEXT_PLUS_DYNAMICS
sum≈15.56
```

2位

```text
B_CONTEXT_TIME_ROTATION
sum≈15.54
```

3位

```text
A_CONTEXT_ONLY
sum≈12.74
```

---

# 考察①

文脈だけより

```text
文脈
+
時間
```

の方が強い。

仮説：

市場は

```text
単語
```

ではなく

```text
その話題が
いつ出たか
```

を織り込む。

---

# 考察②

さらに

```text
文章速度
文章加速度
文章ジャーク
```

追加で改善。

仮説：

文章には

```text
意味
```

だけでなく

```text
テンポ
リズム
運動
```

が存在。

---

例：

ニュース

```text
決算
↓
増配
↓
上方修正
```

穏やか

---

ニュース

```text
戦争
↓
利上げ
↓
破綻
```

急加速

ジャーク大

---

# 考察③

今回一番面白い結果：

```text
文章
=
静止した意味ベクトル
```

より

```text
文章
=
時間とともに動く状態量
```

の方が強かった。

つまり：

```text
LLM
```

を

```text
力学系
```

として扱える可能性。

---

# 現時点仮説

未来：

```text
embedding
↓
velocity
↓
acceleration
↓
jerk
↓
future embedding
↓
next news
↓
next price
```

予測できるかもしれない。

---

# 次にやること

[ ] 本物Time2Vec追加

[ ] Fourier周期特徴

[ ] Embedding FFT

[ ] 文脈エネルギー保存則

[ ] Hidden State Dynamics

[ ] Neural ODE

[ ] Koopman Operator

[ ] Mamba(State Space)

[ ] 未来ニュース予測

---

# 暫定結論

今回最強：

```text
文章意味
+
時期
+
文章力学
```

仮説：

市場は

```text
意味
```

だけでなく

```text
意味の運動
```

を価格に織り込む。
