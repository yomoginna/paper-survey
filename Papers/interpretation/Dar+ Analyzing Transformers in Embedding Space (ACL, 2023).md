# title: Analyzing Transformers in Embedding Space

### info:
- https://aclanthology.org/2023.acl-long.893.pdf
- https://www.youtube.com/watch?v=fhzZ0twiZBA

# 概要
- GPTの最後のLM head（埋め込み行列E）を利用し、GPT系列の各層を解釈する手法の提案（BERTでも効果あり）
	- 各parameter行列が表すtokenまたはtoken同士の関係を解釈できるようになる
 	- モデル内の内積を見ることで、それらの内積を語彙アイテムのペア間の相互作用として解釈できる。以下に適用する。
		- (a) AttetionのQとKの間の相互作用
  		- (b) Attentionモジュールの出力でそれらを射影するparametersとの相互作用に

（読み手の解釈）
- Q. なぜitem間の相互作用として見ることができるのか？
	- A. 行列の変形をした際の形状が、E'WEの形になるから。(埋め込み行列E)
		- WはEに何らかの処理を加えることで、E'に変化させる。したがって、WはEをE'に変化させるための関数、すなわちEとE'の間の相互作用として見ることができる。
- <img width="690" alt="image" src="https://github.com/yomoginna/paper-survey/assets/98722875/be73f797-abdf-4ae5-a08a-37d6a0f24e1f">


# 前提知識
- transformerを積み重ねたLLMにおいて、
	- 残差streamが大きな川のようなもの
	- この大きな川に対し、Attention層やFFで作られた情報を少しずつ足していくイメージ。
 	- そのため、transformerを経るにつれて、FFの状態は徐々に最終出力に近づいていく。（最終出力とFFの一致率が徐々に上がっていく）


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
	- （読み手のmemo・解釈）: 
		- 行列Aは $N \times d$ の大きさのとある行列を指しているだけで、AttentionのAとは関係が無い。
		- 埋め込み行列: $E \in R^{e \times d}$
		- $E$ の右逆行列 $E'$は、 $E' \in R^{d \times e}$

<img width="702" alt="image" src="https://github.com/yomoginna/paper-survey/assets/98722875/83240d48-537e-459e-bf8a-5f1539c8df52">


