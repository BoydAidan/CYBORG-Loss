import os
import torch
import torch.utils.data as data_utl
from PIL import Image
import torchvision.transforms as transforms
# from torchvision.transforms.transforms import ColorJitter, GaussianBlur, RandomAdjustSharpness
from tqdm import tqdm
import sys

class datasetLoader(data_utl.Dataset):

    def __init__(self, split_file, root, train_test, random=True, c2i={}, map_location='',map_size=7,im_size=224,network='densenet'):
        self.class_to_id = c2i
        self.id_to_class = []
        self.map_location = map_location
        self.map_size = map_size
        self.image_size = im_size

        # Class assignment
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0

        # Image pre-processing
        self.data = []
        if network == "xception":
            self.transform = transforms.Compose([
                transforms.Resize([self.image_size, self.image_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([self.image_size, self.image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.map_transform = transforms.Compose([
            transforms.Resize([self.map_size, self.map_size]),
            transforms.ToTensor(),
        ])

        # Reading data from CSV file
        SegInfo=[]
        print("Reading in data for:",train_test)
        with open(split_file, 'r') as f:
            for l in tqdm(f.readlines()):
                v= l.strip().split(',')
                if train_test == v[0]:
                    image_name = v[2]
                    imagePath = root+image_name
                    img = Image.open(imagePath).convert('RGB')
                    tranform_img = self.transform(img)
                    img.close()
                    if train_test == 'train' and os.path.exists(self.map_location + image_name.split("/")[-1]):
                        human_map = Image.open(self.map_location + image_name.split("/")[-1])
                        transform_human_map = self.map_transform(human_map)
                        transform_human_map = transform_human_map.type(torch.float)
                        transform_human_map = torch.squeeze(transform_human_map)
                        transform_human_map = transform_human_map - torch.min(transform_human_map)
                        transform_human_map = transform_human_map / torch.max(transform_human_map)
                        human_map.close()
                    else:
                        transform_human_map = 0
                    c = v[1]
                    if c not in self.class_to_id:
                        self.class_to_id[c] = cid
                        self.id_to_class.append(c)
                        cid += 1
                    # Storing data with imagepath and class
                    self.data.append([imagePath, self.class_to_id[c],tranform_img[0:3,:,:],transform_human_map])
        print("Class assignments:",self.class_to_id)

        self.split_file = split_file
        self.root = root
        self.random = random
        self.train_test = train_test


    def __getitem__(self, index):
        imagePath, cls, img, hmap = self.data[index]
        imageName = imagePath.split('/')[-1]

        # return tranform_img[0:3,:,:], cls, imageName
        return img, cls, imageName, hmap

    def __len__(self):
        return len(self.data)

