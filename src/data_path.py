
import os.path as osp
from utils.augmentation import Compose, Scale, Resize, RandomRotation, RandomMirror, Normalize_Tensor

def make_data_path(rootpath):
    original_image_template = osp.join(rootpath, "JPEGImages", "%s.jpg")
    annotation_image_template = osp.join(rootpath, "SegmentationClass", "%s.png")

    # train val
    train_ids = osp.join(rootpath, "ImageSets/Segmentation/train.txt")
    val_ids = osp.join(rootpath, "ImageSets/Segmentation/val.txt")

    train_image_list = []
    train_annotation_list = []

    val_image_list = []
    val_annotation_list = []

    for line in open(train_ids):
        img_id = line.strip()
        img_path = (original_image_template % img_id)
        anno_path = (annotation_image_template % img_id) 

        train_image_list.append(img_path)
        train_annotation_list.append(anno_path)
    
    for line in open(val_ids):
        img_id = line.strip()
        img_path = (original_image_template % img_id)
        anno_path = (annotation_image_template % img_id)

        val_image_list.append(img_path)
        val_annotation_list.append(anno_path)

    return train_image_list, train_annotation_list, val_image_list, val_annotation_list

class DataTransform:
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train" : Compose([
                Scale(scale=[0.5, 1.5]), # image can scale into only 50% or even 150%
                RandomRotation(angle=[-10, 10]),
                RandomMirror(),
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ]),
            "val": Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)
  
if __name__ == "__main__":
    rootpath = './datasets/VOC2012/'
    train_image_list, train_annotation_list, val_image_list, val_annotation_list = make_data_path(rootpath=rootpath)

    print("Len train image: ", len(train_image_list))
    print("Anno train image: ", len(train_annotation_list))
    print("Len val image: ", len(val_image_list))
    print("Anno vak image: ", len(val_annotation_list))

    print(train_image_list[0])
    print(train_annotation_list[0])
