## info
@misc{ren2024analyzing,
      title={Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning}, 
      author={Weijieying Ren and Xinlong Li and Lei Wang and Tianxiang Zhao and Wei Qin},
      year={2024},
      eprint={2402.18865},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}


## abstract
LLMsが複雑かつ多様なドメイン特有の下流タスク上で継続的にfine-tuneされた場合，historical taskでの推論性能が劇的に低下し，これはcatastorophic forgetting （壊滅的な忘却(?)）問題
として知られている．
学習の柔軟性(plasticity: 可塑性、柔軟性) と記憶の安定性の間のトレードオフを維持する必要がある．
多くの既存研究でmemory再生，正則化，パラメータ分離などの戦略が探求されてきたが，継続的なLLMsのfine-tuningシナリオにおける様々な隣接する最小値の幾何学的な接続についてはほとんどわかっていない．
この研究では，mode接続の長さを通じて異なる最小値の幾何学的接続を探る．これは異なる最小値が低損失の谷によって接続できることを意味する．
広範な実験を通して，我々はLLMsの継続的な学習シナリオにおけるmode接続現象を明らかにし，それが柔軟性と安定性のバランスを取れることを発見した．
これらの発見に基づき，我々はシンプルかつ効果的な手法：補完ベースLoRA (I-LoRA) を提案する．これは二重メモリの経験再生フレームワーク(?)であり，LoRAパラメータ補完に基づく．
広範な経験と8つのドメイン特化CLベンチマークにおける分析により，I-LoRAが一貫して以前のSOTA手法を大きく上回ることが示された．
強いベースラインと将来の研究の洞察を与える．

## background

## problem ("However, ...")

## done

## not done

## experiment & settings

## comment
