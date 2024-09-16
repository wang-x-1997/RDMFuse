import os
import numpy as np
import torch.utils.data as data1
import torch
from PIL import Image

def load_image(x):
  imgA = Image.open(x)
  imgA = imgA.convert('L')
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  return imgA

def load_rgb(x):
  imgA = Image.open(x)
  # imgA = imgA.convert('L')
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  return imgA

def make_dataset(root, train=True):
    dataset = []

    if train:
      dir_img = os.path.join(r'D:\Image_Data\IRVI\LLVIP\256')
      for index in range(9600):
        imgA = str(index + 1) + '-1.jpg'
        imgB = str(index + 1) + '-2.jpg'
        dataset.append([os.path.join(dir_img, imgA), os.path.join(dir_img, imgB)])

    return dataset


class fusiondata(data1.Dataset):

  def __init__(self, root, transform=None, train=True):
    self.train = train

    if self.train:
      self.train_set_path = make_dataset(root, train)

  def __getitem__(self, idx):
    if self.train:
      imgA_path, imgB_path = self.train_set_path[idx]

      imgB = load_image(imgB_path)
      imgA_RGB = load_rgb(imgA_path)

      return imgB, imgA_RGB

  def __len__(self):
    if self.train:
      return 9600

