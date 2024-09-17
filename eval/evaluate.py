import os
import torch
import pyiqa
import argparse
import shutil
from skimage.metrics import peak_signal_noise_ratio
import cv2

device=torch.device("cuda")
psnr=pyiqa.create_metric('psnr',device=device)
lpips=pyiqa.create_metric('lpips',device=device)
ssim=pyiqa.create_metric('ssim',device==device)
fid=pyiqa.create_metric('fid',device=device)#fid需要分两个文件夹计算
niqe=pyiqa.create_metric('niqe',device=device)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='test')      #解析命令行参数
    parser.add_argument('--data', type=str, help='evaluatedata.')
    args = parser.parse_args()


    # all_resultpath='../results'
    # resultsfold=os.listdir(all_resultpath)
    # for fold in resultsfold:
    fold=args.data
    datapath = './results/' + fold + '/test_latest/images'  # 改这个
    fidpath='./fidresult'
    fidsubpath=os.path.join(fidpath,fold)
    if not os.path.exists(fidsubpath):
        os.mkdir(fidsubpath)
    fakepath=os.path.join(fidsubpath,'fake')
    realpath=os.path.join(fidsubpath,'real')
    if not os.path.exists(fakepath):
        os.mkdir(fakepath)
    if not os.path.exists(realpath):
        os.mkdir(realpath)

    totalpsnr=0
    totaloldpsnr=0


    totalniqe=0
    totalssim=0
    totallpips=0
    num=0
    imglist=os.listdir(datapath)
    for imgname in imglist:
        if len(imgname.split('fake_B'))==2:
            fakeimg=os.path.join(datapath,imgname)
            gtimg=os.path.join(datapath,imgname.split('fake_B')[0]+'real_B'+'.png')
            shutil.copy(fakeimg,fakepath)
            shutil.copy(gtimg,realpath)
            totalssim+=ssim(fakeimg,gtimg)
            totallpips+=lpips(fakeimg,gtimg)
            totalniqe+=niqe(fakeimg,gtimg)
            # result=psnr(fakeimg,gtimg)
            totalpsnr+=psnr(fakeimg,gtimg)
            num+=1

    fidresult=fid(fakepath,realpath)
    avgpsnr=totalpsnr/400.0
    avgniqe=totalniqe/400.0
    avgssim=totalssim/400.0
    avglpips=totallpips/400.0


    print(fold,'psnr:',avgpsnr)
    print(fold, 'ssim:', avgssim)
    print(fold, 'fid:', fidresult)
    print(fold, 'niqe:', avgniqe)
    print(fold, 'lpips:', avglpips)
    # print(fold, 'psnr:', avgpsnr)