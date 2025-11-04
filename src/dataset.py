import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter
from data_path import make_data_path, DataTransform
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd


class MyDataSet(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):

        img, anno_class_img = self.pull_item(index)

        return img, anno_class_img

    def pull_item(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)

        # print("Before transform:", np.array(img).mean())        
        anno_path = self.anno_list[index]
        anno_class_img = Image.open(anno_path) 
        # PIL (height, width, channel(RGB))
        # opencv (height, width, channel(BGR))

        img, anno_class_img = self.transform(self.phase, img, anno_class_img)
        # print("After transform:", np.array(img).mean())   
        # print("After transform:", img.mean().item())
        return img, anno_class_img
    

if __name__ == "__main__":
    rootpath = './datasets/VOC2012/'
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    # create transform
    transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)

    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path(rootpath=rootpath)

    train_dataset = MyDataSet(img_list= train_img_list,
                              anno_list=train_annotation_list,
                              phase='train',
                              transform=transform)
    
    val_dataset = MyDataSet(img_list= val_img_list,
                              anno_list=val_annotation_list,
                              phase='val',
                              transform=transform)
    
    # print("Train dataset len: ", train_dataset.__len__())
    # print("Train Tensor image shape: ", train_dataset.__getitem__(0)[0].shape)
    # print("Train Tensor image: ", train_dataset.__getitem__(0)[0])

    # print("Train Tensor anno shape: ", train_dataset.__getitem__(0)[1].shape)
    # print("Train Tensor anno: ", train_dataset.__getitem__(0)[1])
    
    batch_size = 4

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iteration = iter(dataloader_dict['train'])

    images, anno_class_images = next(batch_iteration)
    img = images[0].numpy().transpose(1,2,0)  # (channel (RGB), h,w) -> (h, w, channel(RGB))
    plt.imshow(img)
    plt.show()

    img = anno_class_images[0].numpy()
    plt.imshow(img)
    plt.show()
