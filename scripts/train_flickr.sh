export CUDA_VISIBLE_DEVICES=3,5,6,7

python train.py --name oasis_flickr --dataset_mode Flickr --gpu_ids 0,1,2,3 \
--dataroot dataset/flickr --batch_size 36 --num_workers 8 --add_vgg_loss
