set -ex
python train.py --dataroot /dataset/SEN12_cls  --name ScENtrain --model ccycle_gan --netG CRTB --CRBnum 4 --batch_size 4 --lambda_cls 0.01 --classes_num 7
