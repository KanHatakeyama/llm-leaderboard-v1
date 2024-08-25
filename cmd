

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

#8Bの場合
export LANG=ja_JP.UTF-8
model_name=weblab-GENIAC/Tanuki-8B-dpo-v1.0

export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server --model $model_name --max-model-len 2048 --port 8000 --gpu-memory-utilization 0.9 
python3 src/japanese-task-evaluation.py --model_name $model_name

