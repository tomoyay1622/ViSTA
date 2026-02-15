# ViSTA (Visualization of Solving Trace for ASP)
ASPプログラムの実行トレースと、その可視化を目的としたツール群

## 使い方
- グラウンダーgringoを用いて、ASPプログラムをグラウンディングする
```
$ gringo example/test.lp > example/test.aspif
```

- aspif_translatorを用いて、重み付きルールや選択ルールを通常ルールへ変換する
```
$ python3 converter.py example/test.aspif > example/test.con
```

- solver-tracerを用いて、プログラムの求解トレース(JSON形式)を得る
```
$ python3 cdnl-tracer.py example/test.con > example/test.json
```

- 求解トレースをビジュアライザで表示する。

[このツール](https://tomoyay1622.github.io/ViSTA/)にexample/test.jsonの内容を入力する

