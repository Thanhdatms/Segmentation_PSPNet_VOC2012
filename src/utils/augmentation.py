import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, anno_class_img):
        for transform in self.transforms:
            img, anno_class_img = transform(img, anno_class_img)
        
        return img, anno_class_img
    
class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):
        width = img.size[0]
        height = img.size[1]

        scale = np.random.uniform(self.scale[0], self.scale[1]) # create scale parameter based on Gaussian distribution

        scale_w = int(width*scale)
        scale_h = int(height*scale)

        img = img.resize((scale_w, scale_h), Image.BICUBIC)
        anno_class_img = anno_class_img.resize((scale_w, scale_h), Image.NEAREST)
        
        if scale > 1.0: # nếu ảnh sau khi scale mà lớn hơn 1 lớn    
            left = scale_w - width
            left = int(np.random.uniform(0, left))

            top = scale_h - height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop((left, top, left+width, top+height))
        else:
            p_palette = anno_class_img.copy().getpalette()
            img_original = img.copy()
            anno_class_original = anno_class_img.copy()

            pad_width = width - scale_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height - scale_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0,0,0))
            img.paste(img_original, (pad_width_left, pad_height_top))

            anno_padded = Image.new(anno_class_img.mode, (width, height), (0))
            anno_padded.paste(anno_class_original, (pad_width_left, pad_height_top))
            anno_padded.putpalette(p_palette)
            anno_class_img = anno_padded    
        
        return img, anno_class_img
    

class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):
        rotate_angle = np.random.uniform(self.angle[0], self.angle[1])

        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img
    
class RandomMirror(object):
    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)

        return img, anno_class_img
    
class Resize(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        anno_class_img = anno_class_img.resize((self.input_size, self.input_size), Image.NEAREST)

        return img, anno_class_img
    

class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):
        # convert image from numpy to tensor
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, self.color_mean, self.color_std)

        # ambigious(255) -> convert to 0 (black)
        anno_class_img = np.array(anno_class_img)
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # convert numpy to torch
        anno_class_img = torch.from_numpy(anno_class_img).long()

        return img, anno_class_img
