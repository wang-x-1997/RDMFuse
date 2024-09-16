#coding= utf_8
from __future__ import print_function
import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
import time
import glob

def RGB2YCbCr(img):
    img = img * 255.0
    r, g, b = torch.split(img, 1, dim=1)
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    cb = -0.148 * r - 0.291 * g + 0.439 * b + 128
    cr = 0.439 * r - 0.368 * g - 0.071 * b + 128
    y = y / 255.0
    cb = cb / 255.0
    cr = cr / 255.0
    img_ycbcr = torch.cat([y, cb, cr], dim=1)
    return img_ycbcr

def YCbCr2RGB(img, img_Y):
    img = RGB2YCbCr(img) * 255
    y, cb, cr = torch.split(img, 1, dim=1)
    r = 1.164 * (img_Y * 255 - 16) + 1.596 * (cr - 128)
    g = 1.164 * (img_Y * 255 - 16) - 0.392 * (cb - 128) - 0.813 * (cr - 128)
    b = 1.164 * (img_Y * 255 - 16) + 2.017 * (cb - 128)
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    img_rgb = torch.cat([b, g, r], dim=1)
    return img_rgb*255

def RGB2Y(img):

    r, g, b = torch.split(img, 1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y

def prepare_data(dataset):
    data_dir =dataset
    data = glob.glob(os.path.join(data_dir, "IR*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "IR*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "IR*.bmp")))
    a = data[0][len(str(data_dir))+1:-6]
    data.sort(key=lambda x:int(x[len(str(data_dir))+2:-4]))
    return data
def prepare_data1(dataset):
    data_dir = dataset
    data = glob.glob(os.path.join(data_dir, "VIS*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "VIS*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "VIS*.bmp")))
    data.sort(key=lambda x:int(x[len(str(data_dir))+3:-4]))
    return data

def change(out):
    out1 = out.cpu()
    out_img = out1.data[0]
    out_img = out_img.numpy()
    out_img = out_img.transpose(1, 2, 0)
    return out_img
def change_gray(out):
    out1 = out.cpu()
    out_img = out1.data[0]
    out_img = out_img.squeeze()
    out_img = out_img.numpy()
    return out_img
def count_parameters_in_MB(model):
    return print(np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6)
def load_image(x):
  imgA = Image.open(x)
  imgA = imgA.convert('L')
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  imgA = imgA.unsqueeze(0)
  return imgA

def load_rgb(x):
  imgA = Image.open(x)
  # imgA = imgA.convert('L')
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  imgA = imgA.unsqueeze(0)
  return imgA

def norm_1(x):
    max1 = torch.max(x)
    min1 = torch.min(x)
    return (x-min1)/(max1-min1 + 1e-10)


start=time.time()
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)
device = torch.device("cuda")
model_path =r'./model/model.pth'
model = torch.load(model_path)
count_parameters_in_MB(model)
image_B_list = prepare_data(r'D:\Image_Data\IRVI\AUIF Datasets\16x\Test_LLVIP/')
image_A_list = prepare_data1(r'D:\Image_Data\IRVI\AUIF Datasets\16x\Test_LLVIP/')
save_path = "./Fused image/"
if os.path.exists(save_path):
    pass
else:
    os.makedirs(save_path)
all_time = []

for i in range(len(image_A_list)):
    S = time.time()
    imgB = load_image(image_B_list[i])
    imgA_RGB = load_rgb(image_A_list[i])

    if not opt.cuda:
        model = model.to(device)
        imgB = (imgB/255).to(device)
        imgA_RGB = (imgA_RGB/255).to(device)
        imgA =  RGB2Y(imgA_RGB)

    with torch.no_grad():
        R,ill, IR, Fout, ill_en= model(imgA,imgB)
        Fused = YCbCr2RGB(imgA_RGB,Fout*ill_en)
        out = change(Fused)

    cv2.imwrite(os.path.join(save_path,str(i+1)+'.jpg'),out)
    print('mask'+str(i+1)+' has saved')
    all_time.append(time.time() - S)

print('Mean [%f], var [%f]'% (np.mean(all_time), np.std(all_time)))

