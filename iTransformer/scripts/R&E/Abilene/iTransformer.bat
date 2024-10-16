set CUDA_VISIBLE_DEVICES=0

set model_name=iTransformer

python run4RE.py ^
  --is_training 1 ^
  --root_path "Abilene_dataset_path" ^
  --data_path Abilene ^
  --model_id Abilene ^
  --model %model_name% ^
  --data Abilene ^
  --features M ^
  --seq_len 8 ^
  --label_len 0 ^
  --pred_len 1 ^
  --e_layers 4 ^
  --enc_in 862 ^
  --dec_in 862 ^
  --c_out 862 ^
  --des Exp ^
  --d_model 2048 ^
  --d_ff 512 ^
  --batch_size 16 ^
  --learning_rate 0.001 ^
  --itr 1 ^
  --embed fixed ^
  --freq s ^
  --use_gpu 1 ^
  --inverse 