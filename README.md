# Transformer: Attention is all you need

[Attention is all you need](https://arxiv.org/abs/1706.03762)のpytorch実装。

- train.py: Transformerの学習を行うスクリプト
- translate.py: 学習済みモデルをロードして、入力データを翻訳するスクリプト

<br>



## 環境構築

```python3
git clone https://github.com/marucha80t/pytorch-transformer.git
cd ./pytorch-transformer
pip install -r requirements.txt
```

<br>



## 使用方法

翻訳モデルの学習

```python3
python train.py --train ./data/sample_train.tsv
                --valid ./data/sample_valid.tsv
                --savedir ./checkpoints
                --gpu
```



学習した翻訳モデルを利用して翻訳

```python3
python translate.py --model ./checkpoints/checkpoint_best.pt
                    --input ./data/sample_test.txt
                    --gpu
```

<br>



## 参考

- [Attention is all you need](https://arxiv.org/abs/1706.03762)