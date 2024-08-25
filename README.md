# 0825 vllmに仕様変更 by hatakeyama
'''
#8x8bの実行法
#terminal1でvllmを立ち上げる
model_name=team-hatakeyama-phase2/Tanuki-8x8B-dpo-v1.0
export CUDA_VISIBLE_DEVICES=2,3
conda activate eval3
export LIBRARY_PATH="/usr/local/cuda-12.2/lib64/stubs:$LIBRARY_PATH"
python -m vllm.entrypoints.openai.api_server --model $model_name --max-model-len 2048 --port 8000 --tensor-parallel-size 2 --gpu-memory-utilization 0.8 --trust-remote-code

#terminal2で評価を行う
model_name=team-hatakeyama-phase2/Tanuki-8x8B-dpo-v1.0
python3 src/japanese-task-evaluation.py --model_name $model_name



'''

# 手順
```
git clone https://github.com/matsuolab/llm-leaderboard-v1.git -b jglue
cd llm-leaderboard-v1
docker build --no-cache -t llm-leaderboard . 
cd $HOME
docker run --env-file $HOME/llm-leaderboard-v1/.env --mount type=bind,src=$HOME,dst=$HOME --workdir $HOME/llm-leaderboard-v1 --ipc host --gpus all --rm --name `whoami`_llm-leaderboard llm-leaderboard python3 src/japanese-task-evaluation.py
```

# Weights & BiasesのLaunchを用いて、Hugging Face上のモデルを日本語タスクJGLEUに対して評価！

このgit repositoryでは、Hugging Face上のモデルを日本語タスクJGLEUに対して精度評価するためのscriptを管理しています。
LLMの進展が激しいですが、様々なLLMに対して日本語のタスクに対する精度がはどうなっているのかを検証するといったプロジェクトをWeights＆Biases（以下W&B）が進めました。
下記がそのプロジェクトのレポートです。

[W&B REPORT: LLMのJGLUEによる日本語タスクベンチマーク](https://wandb.ai/wandb/LLM_evaluation_Japan/reports/LLM-JGLUE---Vmlldzo0NTUzMDE2 "LLMのJGLUEによる日本語タスクベンチマーク")

そしてW&BのLaucnhを用いて、この評価が誰でも自身の環境で、上記のレポート内で行った評価を行うことができるようにプロジェクトも公開しました


# Launchとは？
W&B Launchは、ML開発者がモダンなMLワークフローを支える高スケールで専門的なハードウェアをシームレスに使用することを可能にし、学習のスケールアップやモデル評価フローの構築、推論のためのモデル読み込みなどの煩わしさを解消してくれます。
<img width="1257" alt="image" src="https://github.com/olachinkei/llm-evaluation-japanese-task/assets/135185730/01cc695d-65ee-4736-aa9c-a2b2b3eb682a">


詳しくは、[W＆BのDoc](https://docs.wandb.ai/ja/guides/launch#docusaurus_skipToContent_fallback)を参考にしてください。


# 実行のステップ
実行のプロセスは、W&BのReport "[W&B Launchを使ってHuggingFace上のLLMを日本語タスクJGLEUで評価](https://wandb.ai/wandb/LLM_evaluation_Japan_public/reports/W-B-Launch-Hugging-Face-LLM-JGLEU---Vmlldzo0NzU2MzIz)"にて解説をしているので、そちらを参照して下さい。

# 注意
Hugging Face上のすべてのモデルの実行を検証しているわけではありませんので、ご容赦ください。
また、まだまだプロンプトも工夫の余地があり、ジョブやコードをupdateする予定がある旨、ご容赦ください。



