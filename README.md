BEGAN(TensorFlow)
====

## Description
BEGANのTensorFlowを使用した実装

## Usage
'python3 main.py'

## File
BEGAN_tensorflow
 ├ main.py
 │  └ 起動用
 ├ operator/
 │  └ operator.py
 │    └ 学習、テストの定義
 ├ function/
 │  └ img_gen.py
 │    └ 画像出力関数定義
 ├ models/
 │  └ BEGAN.py
 │    └ Generator・Discriminatorの定義
 ├ layer/
 │  └ layers.py
 │    └ Conv層・ELU層などの定義
 ├ Data/
 │  └  celeba/
 │    └ 学習用データ
