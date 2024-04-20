# title: Analyzing Transformers in Embedding Space

### info:
- https://aclanthology.org/2023.acl-long.893.pdf
- https://www.youtube.com/watch?v=fhzZ0twiZBA

# 概要
- GPTの最後のLM head（埋め込み行列E）を利用し、GPT系列の各層を解釈する手法の提案（BERTでも効果あり）
	- 各位置のベクトルは単に埋め込み行列Eの適用でtokenを表すベクトルに変換する。
 	- Attention（FFの2層もAttentionと同様に考える。先行研究より）については、行列の計算により、inputを各層に通す前後のtoken同士の関係を解釈する。
	 	- モデル内の内積を見ることで、それらの内積を語彙アイテムのペア間の相互作用として解釈できる。以下に適用する。
			- (a) AttetionのQとKの間の相互作用
	  		- (b) Attentionモジュールの出力でそれらを射影するparametersとの相互作用

（読み手の解釈）
- Q. なぜitem間の相互作用として見ることができるのか？
	- A. 行列の変形をした際の形状が、 $\hat{X}W\hat{X}^T$ の形になるから。(埋め込み行列E, 残差streamの表現X)
		- WはXに何らかの処理を加えることで、元の情報を変化させる。したがって、前の表現と後の表現の間の相互作用として見ることができる。

# 前提知識
- transformerを積み重ねたLLMにおいて、
	- 残差streamが大きな川のようなもの
	- この大きな川に対し、Attention層やFFで作られた情報を少しずつ足していくイメージ。
 		- transformerを経るにつれて、FFの状態は徐々に最終出力に近づいていく。（最終出力とFFの一致率が徐々に上がっていく）
  	- （ここは読み手の解釈）したがって、残差streamの最終状態に対して適用されるLM headを、transformerの全過程に適用な埋め込み行列Eとみなして計算を行う。


# 埋め込み空間への射影方法
Table 1にしたがって計算する。

<img width="640" alt="image" src="https://github.com/yomoginna/paper-survey/assets/98722875/74dbe1cb-e881-4817-a1dd-00c8d0ddde36">

- Symbol: 解釈したい行列
- Projection: Symbolを解釈するための計算方法
- Approximate Projection: LayerNormとバイアスを無視した擬似右逆行列による計算方法


- 埋め込み行列Eについて：
	- introductionで
	>	「本研究の手法はGPTファミリーにのみ適用できる。**GPTファミリーには線形言語モデリングヘッド（LMヘッド）** があります。これは単純に出力埋め込み行列です。**我々のフレームワークは、線形言語モデリングヘッドが必要**です。」
	- と言っていることから、EはGPTの最後のLM headを意味していると思う。

	- 行列 $A ∈ R^{N ×d}$ が与えられた場合、 $A$ を埋め込み空間に射影するためには、**埋め込み行列E**との積である $\hat{A} = AE ∈ R^{N \times e}$ を計算します。
	- （読み手の解釈とメモ）: 
		- 行列Aは $N \times d$ の大きさのとある行列を指しているだけで、AttentionのAとは関係が無い。
		- 埋め込み行列: $E \in R^{e \times d}$
		- $E$ の右逆行列 $E'$は、 $E' \in R^{d \times e}$



- 計算式の導出例：
	- 埋め込み行列について：
		- $E'$ を $E$ の右逆行列とし、 $EE^′ = I ∈ R^{d×d}$
		- この時、  $$A = A{(EE’)}=\hat{A}E’ \tag{a}$$ 
		- (行列 $A$ は $N\times d$ の大きさのとある行列を指しているだけで、Attentionとは関係が無い。)
	- Attention module:
		-  $A^i$ はKとQから求めた注意の値 (softmax(...))
		- $W_V$ はValue計算時の重み行列 ( $V_{att} =XW_V$  )
		- $W_O$ は、 $A^i$ を利用しValueから情報を取得した後に（FFへ出力を転送する直前に）、埋め込み空間に射影するための重み行列
		-  $W^i_{VO} := W^i_VW^i_O ∈ R^{d×d}$ 
		- この時、各Attention headの出力は、 $A^iXW^i_{VO} = A^i \hat{X}E'W^i_{VO}$
		- 式(a)より、 $(A^iXW^i_{VO})E = A^i\hat{X}(E'W^i_{VO}E)$
		- $A^i$ の役割は更新されたN個の入力ベクトルの表現を混合することだけである。
		- よって重要なのは $\hat{X}(E'W^i_{VO}E)$ であると仮定する。
		- この式から、前の隠れ状態（ $\hat{X}$ ）が埋め込み空間で取られ、次の隠れ状態に残差ストリームを介して組み込まれる出力が、埋め込み空間で生成されることがわかる。
		-  $E'W^i_{VO}E$ は、埋め込み空間の表現( $\hat{X}$ )を受け取り、同じ空間で新しい表現を出力する遷移行列。
		- 同様に、行列 $W^i_{QK}$ は双線形写像と見做すことが出来る。
			-   memo: 
				- 双線形写像は、2つのベクトル空間から別のベクトル空間への写像であり、両方の引数に対して線形性を持つ写像です。つまり、それぞれの引数に対して線形性が保たれるという性質を持つ。
				- B:V×W→Uに対応する行列𝑀が存在するとします。この行列𝑀を用いて双線形写像𝐵を抽出する式は次のようになります： $𝐵(𝑣,𝑤)=𝑣^𝑇𝑀𝑤$ 
		- 次の操作を行うと
		- $$XW^i_{QK}X^T = (XEE')W^i_{QK}(XEE')^T = (XE)E'W^i_{QK}E'^T(XE)^T = \hat{X}(E'W^i_{QK}E'^T)\hat{X}^T.$$
		- 異なる位置のトークン間の相互作用は、語彙アイテムのペア間の相互作用を表すe×e行列によって決定されることがわかる。( $e=|E|$ )


<img width="690" alt="image" src="https://github.com/yomoginna/paper-survey/assets/98722875/be73f797-abdf-4ae5-a08a-37d6a0f24e1f">

