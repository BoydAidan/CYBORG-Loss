import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import argparse
import cv2
from shutil import copy2
from tqdm import tqdm


def image_in_list(image,array):
    for check in array:
        if np.array_equal(image,check):
            print("Duplicate found...")
            return True
    return False


def main(csv_dest):

    multiplier = 6
    # csv_file = open(csv_dest + "large_dataset_" + str(multiplier) + "x.csv","w+")
    mode = "train"
    num_real = 0
    num_spoof = 0
    img_source = "Original"
    used_images_real = []
    used_images_spoof = []
    for f in os.listdir("/scratch365/aboyd3/DataSynFace/BlurredImages/aligned/10/"):

        if ".png" not in f and ".jpg" not in f:
            continue

        if 'real' in f:
            # if b == 0:
            copy2("/scratch365/aboyd3/DataSynFace/BlurredImages/aligned/" + img_source + "/" + f,train_real_loc)
            spoof = "Real"
            num_real += 1
            used_images_real.append(cv2.imread("/scratch365/aboyd3/DataSynFace/BlurredImages/aligned/Original/" + f,cv2.IMREAD_GRAYSCALE))
        else:
            # if b == 0:
            copy2("/scratch365/aboyd3/DataSynFace/BlurredImages/aligned/" + img_source + "/" + f,train_fake_loc)
            spoof = "Synthetic"
            num_spoof += 1
            used_images_spoof.append(cv2.imread("/scratch365/aboyd3/DataSynFace/BlurredImages/aligned/Original/" + f,cv2.IMREAD_GRAYSCALE))

        # csv_file.write(mode + "," + spoof + ",/" + img_source + "/" + f + "\n")
        
    mode = "test"
    print("Creating validation set...")

    print("Collecting SREFI images...")
    srefi_images = []
    img_source = "SREFI"
    for srefi in os.listdir(data_path + img_source):
        if ".png" not in srefi and ".jpg" not in srefi:
            continue
        line = mode + ",Synthetic,/" + img_source + "/" + srefi + "\n"
        srefi_images.append(line)
    
    print("Collecting StyleGAN images...")
    sg2_images = []
    img_source = "SG2"
    for sg2 in os.listdir(data_path + img_source):
        if ".png" not in sg2 and ".jpg" not in sg2:
            continue
        line = mode + ",Synthetic,/" + img_source + "/" + sg2 + "\n"
        sg2_images.append(line)
        
    print("Collecting BXGrid images...")
    real_images = []
    img_source = "ND-Real"
    for real in os.listdir(data_path + img_source):
        if ".png" not in real and ".jpg" not in real:
            continue
        line = mode + ",Real,/" + img_source + "/" + real + "\n"
        real_images.append(line)
    
    random.seed(42)
    random.shuffle(srefi_images)
    random.seed(42)
    random.shuffle(sg2_images)
    random.seed(42)
    random.shuffle(real_images)
    real_shrunk = real_images[:10000]
    srefi_shrunk = srefi_images[:5000]
    sg2_shrunk = sg2_images[:5000]

    print("Collecting large set of StyleGAN images...")
    sg2_50k_images = []
    img_source = "SG2_extended"
    for sg2 in tqdm(os.listdir(data_path + img_source)):
        if "._" in sg2:
            continue
        if ".png" not in sg2 and ".jpg" not in sg2:
            continue
        
        # image = cv2.imread(data_path + img_source + "/" + sg2,cv2.IMREAD_GRAYSCALE)
        # if image_in_list(image,used_images_spoof):
        #     continue
        line = mode + ",Synthetic,/" + img_source + "/" + sg2 + "\n"
        if line not in sg2_shrunk:
            sg2_50k_images.append(line)

    print("Collecting SREFI images...")
    srefi_50k_images = []
    img_source = "SREFI"
    for srefi in tqdm(os.listdir(data_path + img_source)):
        if "._" in srefi:
            continue
        if ".png" not in srefi and ".jpg" not in srefi:
            continue
        # image = cv2.imread(data_path + img_source + "/" + srefi,cv2.IMREAD_GRAYSCALE)
        # if image_in_list(image,used_images_spoof):
        #     continue
        line = mode + ",Synthetic,/" + img_source + "/" + srefi + "\n"
        if line not in srefi_shrunk:
            srefi_50k_images.append(line)

    print("Collecting BXGrid images...")
    real_images_large = []
    img_source = "ND-Real"
    for real in tqdm(os.listdir(data_path + img_source)):
        if ".png" not in real and ".jpg" not in real:
            continue
        # image = cv2.imread(data_path + img_source + "/" + real,cv2.IMREAD_GRAYSCALE)
        # if image_in_list(image,used_images_real):
        #     continue
        line = mode + ",Real,/" + img_source + "/" + real + "\n"
        if line not in real_shrunk:
            real_images_large.append(line)
    
    print(len(real_images_large))
    random.seed(42)
    random.shuffle(srefi_50k_images)
    random.seed(42)
    random.shuffle(sg2_50k_images)
    random.seed(42)
    # random.shuffle(celeba_images)
    random.shuffle(real_images_large)
    # celeba_shrunk = celeba_images[:3000]
    celeba_shrunk = real_images_large[:(multiplier-1)*num_real]
    srefi_50k_shrunk = srefi_50k_images[:int(((multiplier-1)/2)*num_spoof)]
    sg2_50k_shrunk = sg2_50k_images[:int(((multiplier-1)/2)*num_spoof)]

    for real in celeba_shrunk:
        # csv_file.write(real.replace("test","train"))
        # copy2(data_path + "/celeba-hq/" + real.split("/")[-1].replace("\n",""),train_real_loc)
        copy2(data_path + "/ND-Real/" + real.split("/")[-1].replace("\n",""),train_real_loc)
    for spoof in srefi_50k_shrunk:  
        # csv_file.write(spoof.replace("test","train"))
        copy2(data_path + "/SREFI/" + spoof.split("/")[-1].replace("\n",""),train_fake_loc)
    for spoof in sg2_50k_shrunk:  
        # csv_file.write(spoof.replace("test","train"))
        copy2(data_path + "/SG2_extended/" + spoof.split("/")[-1].replace("\n",""),train_fake_loc)

    for real in real_shrunk:
        # csv_file.write(real)
        copy2(data_path + "/ND-Real/" + real.split("/")[-1].replace("\n",""),val_real_loc)
    for spoof in srefi_shrunk:  
        # csv_file.write(spoof)
        copy2(data_path + "/SREFI/" + spoof.split("/")[-1].replace("\n",""),val_fake_loc)
    for spoof in sg2_shrunk:  
        # csv_file.write(spoof)
        copy2(data_path + "/SG2/" + spoof.split("/")[-1].replace("\n",""),val_fake_loc)

if __name__ == "__main__":

    csv_dest = "/scratch365/aboyd3/SynFace/csvs/"

    data_path = "/scratch365/aboyd3/DataSynFace/BlurredImages/aligned/"

    train_real_loc = "/scratch365/aboyd3/DataSynFace/DFFD/large_data_6x/train/face/0_real/"
    train_fake_loc = "/scratch365/aboyd3/DataSynFace/DFFD/large_data_6x/train/face/1_fake/"
    val_real_loc = "/scratch365/aboyd3/DataSynFace/DFFD/large_data_6x/val/face/0_real/"
    val_fake_loc = "/scratch365/aboyd3/DataSynFace/DFFD/large_data_6x/val/face/1_fake/"

    # loo_attacks = ['artificial','contacts','contacts+print','disease','postmortem','print','synthetic']
    
    # for attack in loo_attacks:
    main(csv_dest)