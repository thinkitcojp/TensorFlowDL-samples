TensorFlowではじめるDeepLearning実践入門　サンプルコード
====

このプログラムはインプレスが発行する書籍「[TensorFlowではじめるDeepLearning実践入門](https://book.impress.co.jp/books/1117101113)」におけるサンプルコードです。

# 構成

本プログラムのフォルダ構成は以下のようになっています。

| フォルダ名 | 説明 |
---|---
| graph | 計算グラフの解説用ソースコード類。 |
| nn | 全結合ニューラルネットワークによる手書き数字認識のソースコード類。 |
| board | TensorBoardの解説用ソースコード類。 |
| conv | 畳み込みニューラルネットワークによる手書き数字認識のソースコード類。 |
| save_model | モデル保存の解説用ソースコード類。|
| s_mnist | RNNによる手書き数字認識のソースコード類。 |
| word2vec | word2vecの解説用ソースコード類。 |
| tfrecord | TFRecordの読み書きの解説用ソースコード類。 |
| nic | イメージキャプショニング用のソースコード類。 |

# 必要環境
書籍に記載済み。後ほどrequirement.txtを作成予定。

# ライセンス
このプログラムは[MITライセンス](https://opensource.org/licenses/mit-license.php)です。

# 正誤表

- 124ページ中段数式　（誤）z * -log(x) + (1-z) * -log(1-x') -> (正) z * -log(x) + (1-z) * -log(1-x)
- 78ページリスト3.14コメント　（誤）10階ごとに精度を検証 -> (正) 100ステップごとに精度を検証

# お問い合わせ

書籍の内容に関するお問い合わせは出版社の[お問い合わせフォーム](https://book.impress.co.jp/books_inquiry/form.html?ic=1117101113)からお願いします、なおお答えできるのは本書に記載の内容に関することに限ります。
[https://book.impress.co.jp/books/1117101113](https://book.impress.co.jp/books/1117101113)

