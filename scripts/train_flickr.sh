export CUDA_VISIBLE_DEVICES=2,3,5,6

python train.py --name oasis_flickr --dataset_mode Flickr --gpu_ids 0,1,2,3 \
--dataroot dataset/flickr --batch_size 20 
