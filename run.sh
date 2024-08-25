#!/bin/bash

# 使用するGPU数
TOTAL_GPUS=8

# モデル名の配列
MODEL_NAMES=("weblab-GENIAC/team_haijima_submit" "weblab-GENIAC/team_kawagoshi_submit" "weblab-GENIAC/team_ozaki_submit" "weblab-GENIAC/team_hatakeyama_submit" "weblab-GENIAC/team_hatakeyama_submit_dpo" "weblab-GENIAC/team_kumagai_submit" "weblab-GENIAC/team_nakamura_submit" "weblab-GENIAC/team_sannai")

# 2つのGPUでモデルを動かすため、1つのGPUセットで使用するGPUの数
GPUS_PER_MODEL=2

# 同時に動かすモデル数（最大値）
MAX_PARALLEL_MODELS=$((TOTAL_GPUS / GPUS_PER_MODEL))

# モデル数
TOTAL_MODELS=${#MODEL_NAMES[@]}

# 実行中のモデル数をカウントする変数
running_models=0

# GPUの使用状況を確認し、使用可能なGPUペアを返す関数
find_available_gpus() {
  local available_gpus=()

  for (( i=0; i<$TOTAL_GPUS; i++ )); do
    if ! nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
      available_gpus+=($i)
    fi
  done

  if [[ ${#available_gpus[@]} -ge $GPUS_PER_MODEL ]]; then
    echo "${available_gpus[@]:0:$GPUS_PER_MODEL}"
  else
    echo ""
  fi
}

# モデルを動かす関数
run_model() {
  local gpu_ids=$1
  local model_name=$2
  # model_name を Dockerのコンテナ名として有効な形式に変換
  safe_model_name=$(echo "$model_name" | tr -cd 'a-zA-Z0-9_.-')
  sudo docker run --env-file $HOME/llm-leaderboard-v1/.env --mount type=bind,src=$HOME,dst=$HOME \
  --workdir $HOME/llm-leaderboard-v1 --ipc host --gpus "\"device=$gpu_ids\"" \
  --rm --name "`whoami`_${safe_model_name}_llm-leaderboard" llm-leaderboard python3 src/japanese-task-evaluation.py --model_name $model_name &
}

# モデルを実行するためのループ
for (( i=0; i<$TOTAL_MODELS; i++ ))
do
  # 空いているGPUペアを探す
  available_gpus=""
  while [[ -z "$available_gpus" || $running_models -ge $MAX_PARALLEL_MODELS ]]; do
    if [[ $running_models -ge $MAX_PARALLEL_MODELS ]]; then
      echo "現在 $MAX_PARALLEL_MODELS 個のモデルが実行中です。次のモデル実行まで待機します..."
      wait -n  # いずれかのバックグラウンドジョブが終了するのを待機
      running_models=$((running_models - 1))  # 終了したらカウントを減らす
    else
      echo "使用可能なGPUを検索中..."
      available_gpus=$(find_available_gpus)
      if [[ -z "$available_gpus" ]]; then
        echo "GPUが全て使用中です。次のチェックまで待機します..."
        sleep 10  # 10秒待機して再チェック
      fi
    fi
  done

  gpu_ids=$(echo $available_gpus | tr ' ' ',')
  model_name=${MODEL_NAMES[$i]}

  echo "モデル $model_name を GPU $gpu_ids で実行中..."
  run_model $gpu_ids $model_name
  running_models=$((running_models + 1))  # 新しいモデルをカウントに追加
done

# 全てのモデルの実行を待機
wait
echo "全モデルの実行が完了しました。"