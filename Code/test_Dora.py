# This is a sample Python script.
# This is a sample Python script.
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import os
import argparse

'''
LAST UPDATED 11/10/2021, lsdr
'''

## Process images in parallel

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file


parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=False)  # Path of file
parser.add_argument("--split", default="test", type=str, required=False)  # validate, test, train

args = parser.parse_args()

# If no path provided, use parent directory
if args.path is None:
    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
else:
    PATH = args.path

DATA_DIR = os.path.join(PATH, 'Data')
SPLIT = args.split


BATCH_SIZE = 32
LR = 0.0001

## Image processing
CHANNELS = 3
IMAGE_SIZE = 128

NICKNAME = "Dora"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5


#---- Define the model ---- #
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 128, (3, 3))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, num_classes)
        self.act = torch.relu

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.act(self.conv2(self.act(x)))
        return self.linear(self.global_avg_pool(x).view(-1, 128))


## ------------------ Data Loadaer definition

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''

    def __init__(self, list_IDs, type_data, target_type):
        # Initialization'
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type

    def __len__(self):
        # Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        if self.type_data == 'train':
            y = xdf_dset.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")
        else:
            y = xdf_dset_test.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")

        if self.target_type == 2:
            labels_ohe = [int(e) for e in y]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)

            for idx, label in enumerate(range(OUTPUTS_a)):
                if label == y:
                    labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        if self.type_data == 'train':
            file = os.path.join(DATA_DIR, xdf_dset.id.get(ID))
        else:
            file = os.path.join(DATA_DIR, xdf_dset_test.id.get(ID))

        img = cv2.imread(file)

        if img is None:
            print(f"Warning: Could not read image {file}")
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # IMAGENET NORMALIZATION (Critical for ResNet!)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        X = torch.FloatTensor(img)
        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        return X, y


def read_data(target_type):
    ## Only the test set
    ## xdf_dset_test ( data set )
    ## read the data from the file

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids_test = list(xdf_dset_test.index)

    # Datasets
    partition = {
        'test': list_of_ids_test
    }

    # Data Loader
    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return test_generator


def predict_with_tta(model, xdata, device, n_augmentations=4):
    """
    Apply Test-Time Augmentation to improve predictions
    Averages predictions across multiple augmentations
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        # Original image
        predictions.append(model(xdata.to(device)).cpu())

        # Horizontal flip
        xdata_flip = torch.flip(xdata, [3])
        predictions.append(model(xdata_flip.to(device)).cpu())

        # Brightness variations
        for brightness in [0.9, 1.1]:
            xdata_bright = torch.clamp(xdata * brightness, -3, 3)  # Clamp to reasonable range
            predictions.append(model(xdata_bright.to(device)).cpu())

    # Average all predictions
    return torch.mean(torch.stack(predictions), dim=0)


def model_definition(pretrained=True):
    """
    Load trained model for testing - MUST match training architecture exactly
    """

    if pretrained == True:
        # Load pretrained ResNet34 (same as training)
        model = models.resnet34(pretrained=False)

        # CRITICAL: Match the EXACT classifier head from training
        num_features = model.fc.in_features

        # This MUST be identical to training
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, OUTPUTS_a)
        )
    else:
        model = CNN(OUTPUTS_a)

    # Load the saved weights
    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
    model = model.to(device)

    # Set to evaluation mode
    model.eval()

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

    criterion = nn.BCEWithLogitsLoss()

    return model, criterion

def test_model(test_ds, list_of_metrics, list_of_agg, pretrained=True):
    # Create the test instructions to
    # Load the model
    # Create the loop to validate the data
    # You can use a dataloader

    model, criterion = model_definition(pretrained)

    pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

    model.eval()

    # Create the evaluation
    with torch.no_grad():
        with tqdm(total=len(test_ds), desc="Testing") as pbar:
            for xdata, xtarget in test_ds:
                xtarget = xtarget.to(device)

                # Use Test-Time Augmentation for better predictions
                output = predict_with_tta(model, xdata, device)
                output = output.to(device)

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

                pbar.update(1)

    # Remove the first row (zeros)
    pred_logits = pred_logits[1:]
    real_labels = real_labels[1:]

    # Apply threshold
    pred_labels = pred_logits.copy()
    pred_labels[pred_labels >= THRESHOLD] = 1
    pred_labels[pred_labels < THRESHOLD] = 0

    # Run the statistics
    test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels, pred_labels)

    print("\nTest Results:")
    for met, dat in test_metrics.items():
        print(f"  {met}: {dat:.5f}")

    # Save the results in the Excel file
    # Remember to write a string with the result (NO A COLUMN FOR each)
    xfinal_pred_labels = []
    for i in range(len(pred_labels)):
        joined_string = ",".join(str(int(e)) for e in pred_labels[i])
        xfinal_pred_labels.append(joined_string)

    xdf_dset_test['results'] = xfinal_pred_labels
    xdf_dset_test.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

    print(f"\nResults saved to: results_{NICKNAME}.xlsx")


def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict

def process_target(target_type):
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''

    dict_target = {}
    xerror = 0

    if target_type == 2:
        target = np.array(xdf_data['target'].apply( lambda x : x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    if target_type == 1:
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names=(xtarget)
        xdf_data['target_class'] = final_target

    ## We add the column to the main dataset

    return class_names


if __name__ == '__main__':

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    ## Processing Train dataset
    ## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
    class_names = process_target(target_type = 2)

    ## Balancing classes , all groups have the same number of observations
    xdf_dset_test= xdf_data[xdf_data["split"] == SPLIT].copy()

    ## read_data creates the dataloaders, take target_type = 2

    test_ds = read_data(target_type = 2)

    OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    test_model(test_ds, list_of_metrics, list_of_agg, pretrained=True)