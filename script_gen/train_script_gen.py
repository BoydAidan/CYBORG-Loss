import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import argparse
import cv2
from shutil import copy2


def main(csv_dest,base_images):

    # print("Loading original images...")
    # train_images = []
    # for im in os.listdir("/scratch365/aboyd3/SynFace/Data/BlurredImages/aligned/Original/"):
    #     loaded_im = cv2.imread("/scratch365/aboyd3/SynFace/Data/BlurredImages/aligned/Original/"+im)
    #     train_images.append(loaded_im)

    mode = "train"
    # if os.path.exists(csv_dest + attack + img_source + ".csv"):
    #     print("Training file already exists...")
    #     return

    # csv_file = open(csv_dest + "human_aided.csv","w+")
    # blur_levels = [2,4,6,8,10,12,14,16]

    csv_file = open(csv_dest + "no_blur.csv","w+")
    blur_levels = [0,2,4,6,8,10,12,14,16]

    for b in blur_levels:
        if b == 0:
            img_source = "Original"
        else:
            img_source = str(b)
        # img_source = "unblurred"
        for f in os.listdir(base_images):

            if ".png" not in f and ".jpg" not in f:
                continue

            if 'real' in f:
                if b == 0:
                    copy2(data_path + img_source + "/" + f,train_real_loc)
                spoof = "Real"
            else:
                if b == 0:
                    copy2(data_path + img_source + "/" + f,train_fake_loc)
                spoof = "Synthetic"

            csv_file.write(mode + "," + spoof + ",/" + img_source + "/" + f + "\n")
        
        
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

    for real in real_shrunk:
        csv_file.write(real)
        copy2(data_path + "/ND-Real/" + real.split("/")[-1].replace("\n",""),val_real_loc)
    for spoof in srefi_shrunk:  
        csv_file.write(spoof)
        copy2(data_path + "/SREFI/" + spoof.split("/")[-1].replace("\n",""),val_fake_loc)
    for spoof in sg2_shrunk:  
        csv_file.write(spoof)
        copy2(data_path + "/SG2/" + spoof.split("/")[-1].replace("\n",""),val_fake_loc)

if __name__ == "__main__":

    base_images = "/scratch365/aboyd3/DataSynFace/BlurredImages/aligned/10/"
    csv_dest = "/scratch365/aboyd3/SynFace/csvs/"

    data_path = "/scratch365/aboyd3/DataSynFace/BlurredImages/aligned/"

    train_real_loc = "/scratch365/aboyd3/DataSynFace/DFFD/train/Real/"
    train_fake_loc = "/scratch365/aboyd3/DataSynFace/DFFD/train/Fake/"
    val_real_loc = "/scratch365/aboyd3/DataSynFace/DFFD/eval/Real/"
    val_fake_loc = "/scratch365/aboyd3/DataSynFace/DFFD/eval/Fake/"

    # loo_attacks = ['artificial','contacts','contacts+print','disease','postmortem','print','synthetic']
    
    # for attack in loo_attacks:
    main(csv_dest,base_images)