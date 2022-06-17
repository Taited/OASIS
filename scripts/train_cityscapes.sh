export CUDA_VISIBLE_DEVICES=6,7
python train.py --name oasis_cityscapes --dataset_mode cityscapes --gpu_ids 0,1 \
--dataroot dataset/cityscape --batch_size 20
