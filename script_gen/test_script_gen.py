import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import argparse
import cv2


def main(csv_dest,base_images,middle_path,name,test_type):

    csv_file = open(csv_dest + name + ".csv","w+")
   
    mode = "test"
    print("Creating test set...")
    test_images = []
    for samp in os.listdir(base_images + middle_path + name):
        
        if os.path.isdir(base_images + middle_path + name + "/" + samp):
            print(samp)
            for subsamp in os.listdir(base_images + middle_path + name + "/" + samp):
                if ".png" not in subsamp and ".jpg" not in subsamp and ".JPG" not in subsamp:
                    print(subsamp)
                    continue
                line = mode + "," + test_type + ",/" + middle_path + name + "/" + samp + "/" + subsamp + "\n"
                test_images.append(line)
        else:

            if ".png" not in samp and ".jpg" not in samp and ".JPG" not in samp:
                continue
            line = mode + "," + test_type + ",/" + middle_path + name + "/" + samp + "\n"
            test_images.append(line)
    

    for test_image in test_images:
        csv_file.write(test_image)


if __name__ == "__main__":

    # name="NeuralTextures"
    # base_images = "/scratch365/aboyd3/DataSynFace/TestData/"
    # csv_dest = "/scratch365/aboyd3/SynFace/csvs/test_sets/"
    # middle_path = "FaceForensics/aligned/"
    # test_type = "Synthetic"

    name="stargan_aligned"
    base_images = "/scratch365/aboyd3/DataSynFace/TestData/"
    csv_dest = "/scratch365/aboyd3/SynFace/csvs/gan_test_sets/"
    middle_path = "StarGANv2/"
    test_type = "Synthetic"

    # /media/aidan/01ffa814-1eb9-4b66-aebe-f83dda84de0c/SynFace_Datasets/CelebA/CelebA/Img/img_celeba/
    

    # loo_attacks = ['artificial','contacts','contacts+print','disease','postmortem','print','synthetic']
    
    # for attack in loo_attacks:
    main(csv_dest,base_images,middle_path,name,test_type)