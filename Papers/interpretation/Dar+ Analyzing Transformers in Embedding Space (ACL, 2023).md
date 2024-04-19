# title: Analyzing Transformers in Embedding Space

### info:
- https://aclanthology.org/2023.acl-long.893.pdf
- https://www.youtube.com/watch?v=fhzZ0twiZBA

# 概要
- 行列計算により、GPT系列（BERTも可）の各層を解釈する手法の提案
	- 各parameter行列が表すtokenまたはtoken同士の関係を解釈できるようになる
- GPT family にのみ適用可能。
	- GPTの最後のLM head（埋め込み行列E）を利用するため。

# 前提
- transformerを積み重ねたLLMにおいて、
	- 残差streamが大きな川のようなもの
	- この大きな川に対し、Attention層やFFで作られた情報を少しずつ足していくイメージ。
-  

# 計算方法
<img width="640" alt="image" src="https://github.com/yomoginna/paper-survey/assets/98722875/74dbe1cb-e881-4817-a1dd-00c8d0ddde36">

- Symbol: 解釈したい行列
- Projection: Symbolを解釈するための計算方法
- Approximate Projection: LayerNormとバイアスを無視した擬似右逆行列による計算方法

- 埋め込み行列Eについて：
	- introductionで
	>	「本研究の手法はGPTファミリーにのみ適用できる。**GPTファミリーには線形言語モデリングヘッド（LMヘッド）** があります。これは単純に出力埋め込み行列です。**我々のフレームワークは、線形言語モデリングヘッドが必要**です。」
	- と言っていることから、EはGPTの最後のLM headを意味していると思う。

	- 行列 $A ∈ R^{N ×d}$ が与えられた場合、 $A$ を埋め込み空間に射影するためには、**埋め込み行列E**との積である $\hat{A} = AE ∈ R^{N \times e}$ を計算します。
	- ☆ memo: 
		- 行列Aは $N \times d$ の大きさのとある行列を指しているだけで、AttentionのAとは関係が無い。
		- 埋め込み行列: $E \in R^{e \times d}$
		- $E$ の右逆行列 $E'$は、 $E' \in R^{d \times e}$

<img width="702" alt="image" src="https://github.com/yomoginna/paper-survey/assets/98722875/83240d48-537e-459e-bf8a-5f1539c8df52">


