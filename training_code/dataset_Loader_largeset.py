import torch
import torch.utils.data as data_utl
from PIL import Image
import torchvision.transforms as transforms
# from torchvision.transforms.transforms import ColorJitter, GaussianBlur, RandomAdjustSharpness
from tqdm import tqdm
import sys

class datasetLoader(data_utl.Dataset):

    def __init__(self, split_file, root, train_test, random=True, c2i={},add_transforms=False,im_size=224,network='densenet'):
        self.class_to_id = c2i
        self.id_to_class = []
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


        # Reading data from CSV file
        print("Reading in data for:",train_test)
        with open(split_file, 'r') as f:
            for l in tqdm(f.readlines()):
                v= l.strip().split(',')
                if train_test == v[0]:
                    image_name = v[2]#.replace(v[2].split(".")[-1],"png")
                    imagePath = root +image_name
                    c = v[1]
                    if c not in self.class_to_id:
                        self.class_to_id[c] = cid
                        self.id_to_class.append(c)
                        cid += 1
                    # Storing data with imagepath and class
                    self.data.append([imagePath, self.class_to_id[c]])
        print("Class assignments:",self.class_to_id)

        self.split_file = split_file
        self.root = root
        self.random = random
        self.train_test = train_test


    def __getitem__(self, index):
        imagePath, cls = self.data[index]
        imageName = imagePath.split('/')[-1]

        # Reading of the image
        path = imagePath
        img = Image.open(path).convert('RGB')

        # Applying transformation
        tranform_img = self.transform(img)
        img.close()

        return tranform_img[0:3,:,:], cls, imageName

    def __len__(self):
        return len(self.data)

