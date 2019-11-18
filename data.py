import os
import numpy as np

import PIL.Image as Image
import cv2
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import json

import time

def load_data(file_path):

	n_files = len(os.listdir(file_path + "left"))
	print(n_files)
	data_path = []
	for i in range(0, n_files):
		left_path = os.path.join(file_path + "left/%04d.jpg"%i)
		right_path = os.path.join(file_path + "right/%04d.jpg"%i)
		data_path.append((left_path, right_path))

	return data_path

class ImageDataset(Dataset):
	def __init__(self, file_path, image_transform = None):
		self.image_transform = image_transform
		self.image_path = load_data(file_path)


	def __len__(self):
		return len(self.input_images)

	def __getitem__(self, idx):
		left_path = self.image_path[idx][0]
		right_path = self.image_path[idx][1]
		left_img = cv2.imread(left_path)
		right_img = cv2.imread(right_path)

		return [left_img, right_img]

if __name__ == '__main__':
	data_path = "./datasets/keyframe1/rect_data/"
	dataloader = ImageDataset(data_path)

	l,r = dataloader.__getitem__(2)
	cv2.imshow("test", r)
	cv2.waitKey(0)
