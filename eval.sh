MODEL_NAMES=("weblab-GENIAC/team_haijima_submit" "weblab-GENIAC/team_kawagoshi_submit" "weblab-GENIAC/team_ozaki_submit" "weblab-GENIAC/team_hatakeyama_submit" "weblab-GENIAC/team_hatakeyama_submit_dpo" "weblab-GENIAC/team_kumagai_submit" "weblab-GENIAC/team_nakamura_submit" "weblab-GENIAC/team_sannai")
model_name=${MODEL_NAMES[0]}
pids=()
sudo docker run --env-file $HOME/llm-leaderboard-v1/.env --mount type=bind,src=$HOME,dst=$HOME --workdir $HOME/llm-leaderboard-v1 --ipc host --gpus '"device=0,1"' --rm --name `whoami`_01_llm-leaderboard llm-leaderboard python3 src/japanese-task-evaluation.py --model_name $model_name &
pids[$!]=$!
model_name=${MODEL_NAMES[1]}
sudo docker run --env-file $HOME/llm-leaderboard-v1/.env --mount type=bind,src=$HOME,dst=$HOME --workdir $HOME/llm-leaderboard-v1 --ipc host --gpus '"device=2,3"' --rm --name `whoami`_02_llm-leaderboard llm-leaderboard python3 src/japanese-task-evaluation.py --model_name $model_name &
pids[$!]=$!
model_name=${MODEL_NAMES[2]}
sudo docker run --env-file $HOME/llm-leaderboard-v1/.env --mount type=bind,src=$HOME,dst=$HOME --workdir $HOME/llm-leaderboard-v1 --ipc host --gpus '"device=4,5"' --rm --name `whoami`_03_llm-leaderboard llm-leaderboard python3 src/japanese-task-evaluation.py --model_name $model_name &
pids[$!]=$!
model_name=${MODEL_NAMES[3]}
sudo docker run --env-file $HOME/llm-leaderboard-v1/.env --mount type=bind,src=$HOME,dst=$HOME --workdir $HOME/llm-leaderboard-v1 --ipc host --gpus '"device=6,7"' --rm --name `whoami`_04_llm-leaderboard llm-leaderboard python3 src/japanese-task-evaluation.py --model_name $model_name &
pids[$!]=$!
wait ${pids[@]}

model_name=${MODEL_NAMES[4]}
pids=()
sudo docker run --env-file $HOME/llm-leaderboard-v1/.env --mount type=bind,src=$HOME,dst=$HOME --workdir $HOME/llm-leaderboard-v1 --ipc host --gpus '"device=0,1"' --rm --name `whoami`_05_llm-leaderboard llm-leaderboard python3 src/japanese-task-evaluation.py --model_name $model_name &
pids[$!]=$!
model_name=${MODEL_NAMES[5]}
sudo docker run --env-file $HOME/llm-leaderboard-v1/.env --mount type=bind,src=$HOME,dst=$HOME --workdir $HOME/llm-leaderboard-v1 --ipc host --gpus '"device=2,3"' --rm --name `whoami`_06_llm-leaderboard llm-leaderboard python3 src/japanese-task-evaluation.py --model_name $model_name &
pids[$!]=$!
model_name=${MODEL_NAMES[6]}
sudo docker run --env-file $HOME/llm-leaderboard-v1/.env --mount type=bind,src=$HOME,dst=$HOME --workdir $HOME/llm-leaderboard-v1 --ipc host --gpus '"device=4,5"' --rm --name `whoami`_07_llm-leaderboard llm-leaderboard python3 src/japanese-task-evaluation.py --model_name $model_name &
pids[$!]=$!
model_name=${MODEL_NAMES[7]}
sudo docker run --env-file $HOME/llm-leaderboard-v1/.env --mount type=bind,src=$HOME,dst=$HOME --workdir $HOME/llm-leaderboard-v1 --ipc host --gpus '"device=6,7"' --rm --name `whoami`_08_llm-leaderboard llm-leaderboard python3 src/japanese-task-evaluation.py --model_name $model_name &
pids[$!]=$!
wait ${pids[@]}