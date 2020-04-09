CUDA_VISIBLE_DEVICES=3 nohup command > log.file 2>&1 &
tail -f nohup.log
python -u ..
lsof -i:6006
ssh -L 16006:127.0.0.1:16006 chenting@10.141.209.121
ps -aux | grep

############
跑完了 CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --cfg configs/cus_ada.yml --id cus_ada > trainlog/ada.log 2>&1 &
language_eval: 0
save_checkpoint_every: 500
val_images_use: 320

跑完了 CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --cfg configs/cus_att2all2.yml --id cus_att2all2 > trainlog/att2all2.log 2>&1 &
跑完了 CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --cfg configs/cus_att2in.yml --id cus_att2in > trainlog/att2in.log 2>&1 &
跑完了 CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --cfg configs/cus_att2in2.yml --id cus_att2in2 > trainlog/att2in2.log 2>&1 &
language_eval: 0
save_checkpoint_every: 1000
val_images_use: 1000
############

10217 CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --cfg configs/cus_newfc.yml --id cus_newfc > trainlog/newfc.log 2>&1 &
language_eval: 0
save_checkpoint_every: 1000
val_images_use: 1000

报错 CUDA_VISIBLE_DEVICES=2 nohup python -u train.py --cfg configs/cus_showtell.yml --id cus_showtell > trainlog/showtell.log 2>&1 &

2210 CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --cfg configs/cus_ada.yml --id cus_ada_lang > trainlog/ada_lang.log 2>&1 &
language_eval: 1
save_checkpoint_every: 500
val_images_use: 320

17077 CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --cfg configs/cus_newfc.yml --id cus_newfc_lang > trainlog/newfc_lang.log 2>&1 &
language_eval: 1
save_checkpoint_every: 1000
val_images_use: 1000


# 暂时先放弃基于RL的优化
bash scripts/copy_model.sh cus_att2all2 cus_att2all2_rl
报错了 CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --cfg configs/cus_att2all2_rl.yml --id cus_att2all2_rl > trainlog/cus_att2all2_rl.log 2>&1 &

# 评价
CUDA_VISIBLE_DEVICES=2 python eval.py --force 1 --dump_images 0 --num_images 5000 --model model.pth --infos_path infos.pkl --language_eval 1

$ python eval.py --input_json cocotest.json --input_fc_dir data/cocotest_bu_fc --input_att_dir data/cocotest_bu_att --input_label_h5 none --num_images -1 --model model.pth --infos_path infos.pkl --language_eval 0