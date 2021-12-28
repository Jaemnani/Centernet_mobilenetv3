CUDA_VISIBLE_DEVICES=3 python main.py ctdet --exp_id coco_mobilenetv3_small --arch mobilenetv3_small --batch_size 16 --lr 5e-4 --num_epochs 200 --lr_step 180,190 --save_all --gpus 3
