@echo off

set seq_len=8
set model_name=PatchTST
set root_path_name="Abilene_dataset_path"
set data_path_name=Abilene
set model_id_name=Abilene
set data_name=Abilene
set random_seed=2023

python -u run_longExp.py ^
  --random_seed %random_seed% ^
  --is_training 1 ^
  --root_path %root_path_name% ^
  --data_path %data_path_name% ^
  --model_id %model_id_name%_%seq_len% ^
  --model %model_name% ^
  --data %data_name% ^
  --features M ^
  --seq_len %seq_len% ^
  --label_len 0 ^
  --pred_len 1 ^
  --enc_in 4 ^
  --dec_in 4 ^
  --e_layers 3 ^
  --n_heads 16 ^
  --d_model 128 ^
  --d_ff 256 ^
  --dropout 0.2 ^
  --fc_dropout 0.2 ^
  --head_dropout 0 ^
  --patch_len 4 ^
  --stride 2 ^
  --des Exp ^
  --train_epochs 100 ^
  --patience 10 ^
  --lradj TST ^
  --pct_start 0.4 ^
  --itr 1 --batch_size 8 ^
  --use_gpu 1 ^
  --embed fixed ^
  --learning_rate 0.0001 ^
  --inverse