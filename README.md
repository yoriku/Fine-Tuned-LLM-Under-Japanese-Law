# Fine-Tuned-LLM-Under-Japanese-Law
 Llama3.1を日本語判例でファインチューニングしたプログラム群．LLMの勉強用に．

> [!NOTE]
> このプログラム群はLLMの勉強を行いながら作成したので，間違っている点があるかもしれません．  
> 未保障であることにご注意ください．

## 説明
このプログラムは，[Llama-3.1-Swallow-8B-Instruct-v0.1](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1)を
[日本の判例のデータ](https://github.com/japanese-law-analysis/data_set)でファインチューニングするプログラムです．  
判決・決定の要旨から理由を生成することを目的とします．  

## 実行環境
Ubuntu 22.04 LTS + Python 3.10 (venv) + CUDA v11.8

## 準備
1. 仮想環境を作成し，ライブラリをインストールする
~~~ cmd
python3 -m venv .llm
source .llm/bin/activate

pip install -r requirements.txt
~~~

（以下の2つの手順は，`dataset/output.json`を用いて実行する場合は不要です．）  
2. [日本の判例のデータ](https://github.com/japanese-law-analysis/data_set)をダウンロードする．  
3. `data_set/precedent/`の中身を`dataset`ディレクトリにコピーする．

## 実行手順
### データセットの作成
以下のプログラムを実行し，ファインチューニング用のデータセットを作成する．  
`dataset/output.json`にこの処理を行った後のデータを入れてありますので，それを利用する場合，この工程は不要です．  
  
データセットは，準備でダウンロードしたデータから，
要旨と理由のセットとなっている判決・決定を抽出して作成します．  
理由の切り出しは，「当裁判所の判断」又は「その理由は，次のとおりである」から「判決する」又は「決定する」までの区間です．  

~~~ python
python3 src/make_dataset.py
~~~

### ファインチューニング
以下のプログラムを実行し，ファインチューニングを行う．  
現状の設定では，GPUのメモリを40GB程度使用して，3時間程度学習に時間がかかります．
GPUにデータが乗り切らない場合は，`CUTOFF_LEN`を短くしてください．  
ただし，これを短くすると精度が著しく下がります．

~~~ python
python3 src/fine_tuning.py
~~~

### モデルのテスト
まず，`dataset`ディレクトリ内に，テスト用の入力データを用意する．  
次に，以下のプログラムを実行し，ファインチューニング後のモデルを実行してみる．  

~~~ python
python3 src/test_model.py
~~~

## 実行例と今後の課題
実行結果の例として，[検察官がした刑事確定訴訟記録の閲覧申出一部不許可処分に対する準抗告棄却決定に対する特別抗告事件](https://www.courts.go.jp/app/hanrei_jp/detail2?id=38040)のテストデータを，
`dataset/test.json`に用意してあります．
出力結果は，`content/results.txt`に例示しています．  
それっぽい文章になってますが，単語の間違いや意味の繋がらない理論構成が見える結果となったので，
これらを考慮できるような処理やデータの収集を検討中です．

## 参考サイト
- [Google Colab で Llama-3.1-Swallow を試す](https://note.com/npaka/n/n7b93ed74d05c)
- [MetaのオープンソースLLM「Llama 3」を日本語版にファインチューニング（SFT](https://qiita.com/bostonchou/items/bf4a34dcbaf45828f886)
