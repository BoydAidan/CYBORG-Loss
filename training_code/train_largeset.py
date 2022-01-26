import os
import argparse
import json
from Evaluation import evaluation
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_Loader_largeset import datasetLoader
from tqdm import tqdm
import sys
sys.path.append("../")
# sys.path.append("./")
from xception.network.models import model_selection


# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=20)
parser.add_argument('-nEpochs', type=int, default=50)
parser.add_argument('-csvPath', required=False, default= '../csvs/large_dataset_7x.csv',type=str)
parser.add_argument('-datasetPath', required=False, default= '../Data/train/original_data/',type=str)
parser.add_argument('-outputPath', required=False, default= '../model_output_local/cvpr_test_largemodels/',type=str)
parser.add_argument('-network', default= 'densenet',type=str)
parser.add_argument('-nClasses', default= 2,type=int)

args = parser.parse_args()
device = torch.device('cuda')

print(args)


# Definition of model architecture
if args.network == "resnet":
    im_size = 224
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.nClasses)
    model = model.to(device)
elif args.network == "inception":
    im_size = 299
    model = models.inception_v3(pretrained=True,aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.nClasses)
    model = model.to(device)  
elif args.network == "xception":
    im_size = 299
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    model = model.to(device)
else:
    im_size = 224
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, args.nClasses)
    model = model.to(device)

print(model)

# Creation of Log folder: used to save the trained model
log_path = os.path.join(args.outputPath, 'Logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)


# Creation of result folder: used to save the performance of trained model on the test set
result_path = os.path.join(args.outputPath , 'Results')
if not os.path.exists(result_path):
    os.mkdir(result_path)


if os.path.exists(result_path + "/DesNet121_Histogram.jpg") and os.path.exists(log_path + "/DesNet121_best.pth"):
    print("Training already completed for this setup, exiting...")
    sys.exit()


class_assgn = {'Real':0,'Synthetic':1}

# Dataloader for train and test data
dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train',c2i=class_assgn,im_size=im_size,network=args.network)
dl = torch.utils.data.DataLoader(dataseta, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataset = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id,im_size=im_size,network=args.network)
test = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataloader = {'train': dl, 'test':test}

lr = 0.005
solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
lr_sched = optim.lr_scheduler.StepLR(solver, step_size=12, gamma=0.1)
print("Using SGD with LR:",lr)

criterion = nn.CrossEntropyLoss()

# File for logging the training process
with open(os.path.join(log_path,'params.json'), 'w') as out:
    hyper = vars(args)
    json.dump(hyper, out)
log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}

#####################################################################################
#
############### Training of the model and logging ###################################
#
#####################################################################################
train_loss=[]
test_loss=[]
bestAccuracy = 0
bestEpoch=0
for epoch in range(args.nEpochs):

    for phase in ['train', 'test']:
        train = (phase=='train')
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        tloss = 0.
        acc = 0.
        tot = 0
        c = 0
        testPredScore = []
        testTrueLabel = []
        imgNames=[]
        with torch.set_grad_enabled(train):
            for data, cls, imageName in tqdm(dataloader[phase]):

                # Data and ground truth
                data = data.to(device)
                cls = cls.to(device)

                # Running model over data
                outputs = model(data)

                # Prediction of accuracy
                pred = torch.max(outputs,dim=1)[1]
                corr = torch.sum((pred == cls).int())
                acc += corr.item()
                tot += data.size(0)
                loss = criterion(outputs, cls)

                # Optimization of weights for training data
                if phase == 'train':
                    solver.zero_grad()
                    loss.backward()
                    solver.step()
                    log['iterations'].append(loss.item())
                elif phase == 'test':
                    temp = outputs.detach().cpu().numpy()
                    scores = np.stack((temp[:,0], np.amax(temp[:,1:args.nClasses], axis=1)), axis=-1)
                    testPredScore.extend(scores)
                    testTrueLabel.extend((cls.detach().cpu().numpy()>0)*1)
                    imgNames.extend(imageName)

                    
                tloss += loss.item()
                c += 1

        # Logging of train and test results
        if phase == 'train':
            log['epoch'].append(tloss/c)
            log['train_acc'].append(acc/tot)
            print('Epoch: ', epoch, 'Train loss: ',tloss/c, 'Accuracy: ', acc/tot)
            train_loss.append(tloss / c)

        elif phase == 'test':
            log['validation'].append(tloss / c)
            log['val_acc'].append(acc / tot)
            print('Epoch: ', epoch, 'Test loss:', tloss / c, 'Accuracy: ', acc / tot)
            if args.network != "xception":
                lr_sched.step()
            test_loss.append(tloss / c)
            accuracy = acc / tot
            if (accuracy >= bestAccuracy):
                bestAccuracy =accuracy
                testTrueLabels = testTrueLabel
                testPredScores = testPredScore
                bestEpoch = epoch
                save_best_model = os.path.join(log_path,'final_model.pth')
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': solver.state_dict(),
                }
                torch.save(states, save_best_model)
                testImgNames= imgNames

    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': solver.state_dict(),
    }
    with open(os.path.join(log_path,'model_log.json'), 'w') as out:
        json.dump(log, out)
    torch.save(states, os.path.join(log_path,'current_model.pth'))


# Plotting of train and test loss
plt.figure()
plt.xlabel('Epoch Count')
plt.ylabel('Loss')
plt.plot(np.arange(0, args.nEpochs), train_loss[:], color='r')
plt.plot(np.arange(0, args.nEpochs), test_loss[:], 'b')
plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
plt.savefig(os.path.join(result_path,'model_Loss.jpg'))


# Evaluation of test set utilizing the trained model
obvResult = evaluation()
errorIndex, predictScore, threshold = obvResult.get_result(testImgNames, testTrueLabels, testPredScores, result_path)


