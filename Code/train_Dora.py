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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import os
import argparse
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime


# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Create log filename with timestamp
log_filename = f'logs/training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)


## Process images in parallel

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

parser = argparse.ArgumentParser(description='Train Multi-Label Image Classifier')
parser.add_argument("--path", type=str, default=None, help='Path to exam directory')
args = parser.parse_args()

# If no path provided, use parent directory
if args.path is None:
    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
else:
    PATH = args.path

DATA_DIR = os.path.join(PATH, 'Data')


n_epoch = 20
BATCH_SIZE = 30
LR = 0.0001

## Image processing
CHANNELS = 3
IMAGE_SIZE = 100

NICKNAME = "Dora"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True


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

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs, type_data, target_type):
        #Initialization'
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type

    def __len__(self):
        #Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        #Generates one sample of data'
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
            labels_ohe = [ int(e) for e in y]
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

            # DATA AUGMENTATION for training only
            if self.type_data == 'train':
                # Horizontal flip (50% chance)
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)

                # Brightness adjustment (50% chance)
                if random.random() > 0.5:
                    brightness = random.uniform(0.7, 1.3)
                    img = np.clip(img * brightness, 0, 255).astype(np.uint8)

                # Rotation (50% chance)
                if random.random() > 0.5:
                    angle = random.randint(-20, 20)
                    M = cv2.getRotationMatrix2D((IMAGE_SIZE // 2, IMAGE_SIZE // 2), angle, 1)
                    img = cv2.warpAffine(img, M, (IMAGE_SIZE, IMAGE_SIZE))

                # Gaussian noise (30% chance) - UNINDENTED
                if random.random() > 0.7:
                    noise = np.random.normal(0, 5, img.shape)
                    img = np.clip(img + noise, 0, 255).astype(np.uint8)

                # Random crop and resize (40% chance) - UNINDENTED
                if random.random() > 0.6:
                    crop_ratio = random.uniform(0.8, 1.0)  # Crop 80-100% of image
                    crop_size = int(IMAGE_SIZE * crop_ratio)
                    top = random.randint(0, IMAGE_SIZE - crop_size)
                    left = random.randint(0, IMAGE_SIZE - crop_size)
                    img = img[top:top + crop_size, left:left + crop_size]
                    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

                # Random blur (30% chance) - UNINDENTED
                if random.random() > 0.7:
                    kernel_size = random.choice([3, 5, 7])
                    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

                # Color saturation adjustment (40% chance) - UNINDENTED
                if random.random() > 0.6:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.6, 1.4)  # Saturation
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # Random contrast (40% chance) - UNINDENTED
                if random.random() > 0.6:
                    alpha = random.uniform(0.7, 1.3)  # Contrast
                    img = np.clip(alpha * img, 0, 255).astype(np.uint8)


        # Convert BGR to RGB (OpenCV loads as BGR, ResNet expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize for pretrained models (ImageNet stats)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        X = torch.FloatTensor(img)
        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        return X, y


def read_data(target_type):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file


    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])

    ds_targets = xdf_dset['target_class']

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)


    # Datasets
    partition = {
        'train': list_of_ids,
        'test' : list_of_ids_test
    }

    # Data Loaders

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}

    training_set = Dataset(partition['train'], 'train', target_type)
    training_generator = data.DataLoader(training_set, **params)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return training_generator, test_generator

def save_model(model):
    # Open the file

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

def model_definition(pretrained=True):
    # Define a Keras sequential model
    # Compile the model

    # Load pretrained ResNet18 model

    if pretrained == True:
        model = models.resnet18(pretrained=True)
        # Freeze early layers (conv1, bn1, layer1, layer2, layer3)
        # Only train layer4 and fc (final layers)
        layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']

        for name, param in model.named_parameters():
            # Check if parameter belongs to layers we want to freeze
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False  # Freeze this parameter

        print("Frozen layers: conv1, bn1, layer1, layer2, layer3")
        print("Training layers: layer4, fc")
        # 👆 END OF FREEZING CODE 👆

        # Replace final layer with dropout for regularization
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, OUTPUTS_a)
        )
    else:
        model = CNN(OUTPUTS_a)

    model = model.to(device)

    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # More patient scheduler for pretrained model
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Add this print statement right after the scheduler line
    print(f"Scheduler initialized with patience=3, will reduce LR by 0.5x when plateau detected")

    save_model(model)
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Model: ResNet18 (pretrained={pretrained})")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    logger.info(f"Optimizer: AdamW (lr={LR}, weight_decay={optimizer.param_groups[0]['weight_decay']})")
    logger.info(f"Scheduler: ReduceLROnPlateau (patience=3)")

    return model, optimizer, criterion, scheduler


def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on, pretrained = False):
    # Use a breakpoint in the code line below to debug your script.

    model, optimizer, criterion, scheduler = model_definition(pretrained)

    cont = 0
    train_loss_item = list([])
    test_loss_item = list([])

    pred_labels_per_hist = list([])

    model.phase = 0

    met_test_best = 0
    # Early stopping variables
    patience_counter = 0
    patience_limit = 5  # Stop if no improvement for 5 epochs
    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        train_hist = list([])
        test_hist = list([])

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:

            for xdata,xtarget in train_ds:

                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()

                output = model(xdata)

                loss = criterion(output, xtarget)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()
                cont += 1

                steps_train += 1

                train_loss_item.append([epoch, loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                if len(train_hist) == 0:
                    train_hist = xtarget.cpu().numpy()
                else:
                    train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])

                pbar.update(1)
                pbar.set_postfix_str("Test Loss: {:.5f}".format(train_loss / steps_train))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_train_loss = train_loss / steps_train

        ## Finish with Training

        ## Testing the model

        model.eval()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        test_loss, steps_test = 0, 0
        met_test = 0

        with torch.no_grad():

            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata,xtarget in test_ds:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    optimizer.zero_grad()

                    output = model(xdata)

                    loss = criterion(output, xtarget)

                    test_loss += loss.item()
                    cont += 1

                    steps_test += 1

                    test_loss_item.append([epoch, loss.item()])

                    pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                    if len(pred_labels_per_hist) == 0:
                        pred_labels_per_hist = pred_labels_per
                    else:
                        pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                    if len(test_hist) == 0:
                        tast_hist = xtarget.cpu().numpy()
                    else:
                        test_hist = np.vstack([test_hist, xtarget.cpu().numpy()])

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                    pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        #acc_test = accuracy_score(real_labels[1:], pred_labels)
        #hml_test = hamming_loss(real_labels[1:], pred_labels)

        avg_test_loss = test_loss / steps_test

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)


        xstrres = xstrres + " - "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        print(xstrres)
        logger.info(xstrres)

        # Update learning rate scheduler based on test performance
        scheduler.step(met_test)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        logger.info(f"Learning rate: {current_lr:.6f}")
        logger.info(f"Train loss: {avg_train_loss:.5f} | Test loss: {avg_test_loss:.5f}")

        # Early stopping check
        if met_test > met_test_best:
            patience_counter = 0  # Reset counter when improving
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter}/{patience_limit} epochs")
            logger.warning(f"No improvement for {patience_counter}/{patience_limit} epochs")

            if patience_counter >= patience_limit:
                print(f"\n🛑 Early stopping triggered!")
                logger.info("=" * 70)
                logger.info("EARLY STOPPING TRIGGERED!")
                print(f"Best validation F1: {met_test_best:.5f}")
                logger.info(f"Best validation F1: {met_test_best:.5f}")
                logger.info(f"Stopped at epoch {epoch}")
                logger.info("=" * 70)

                break  # Exit training loop

        if met_test > met_test_best and SAVE_MODEL:

            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
            xdf_dset_results = xdf_dset_test.copy()

            ## The following code creates a string to be saved as 1,2,3,3,
            ## This code will be used to validate the model
            xfinal_pred_labels = []
            for i in range(len(pred_labels)):
                joined_string = ",".join(str(int(e)) for e in pred_labels[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_results['results'] = xfinal_pred_labels

            xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index = False)
            print("The model has been saved!")
            logger.info(f"✓ NEW BEST MODEL SAVED! F1: {met_test:.5f} (previous: {met_test_best:.5f})")
            met_test_best = met_test

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETED!")
    logger.info(f"Best validation F1: {met_test_best:.5f}")
    logger.info(f"Total epochs trained: {epoch + 1}")
    logger.info(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    logger.info(f"Model saved as: model_{NICKNAME}.pt")
    logger.info(f"Results saved as: results_{NICKNAME}.xlsx")
    logger.info("=" * 70)


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
    xavg = 0
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
        ## The target comes as a string  x1, x2, x3,x4
        ## the following code creates a list
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

    # CONFIGURATION LOGGING
    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Device: {device}")
    logger.info(f"Nickname: {NICKNAME}")
    logger.info(f"Epochs: {n_epoch}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Learning Rate: {LR}")
    logger.info(f"Image Size: {IMAGE_SIZE}")
    logger.info(f"Threshold: {THRESHOLD}")
    logger.info(f"Excel file: {FILE_NAME}")
    logger.info("=" * 70)

    ## Process Classes
    ## Input and output


    ## Processing Train dataset
    ## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
    class_names = process_target(target_type = 2)

    ## Comment

    xdf_train_full = xdf_data[xdf_data["split"] == 'train'].copy()

    # Split training data: 85% train, 15% validation
    # Try stratified split, fall back to regular split if fails
    try:
        xdf_dset, xdf_dset_test = train_test_split(
            xdf_train_full,
            test_size=0.15,
            random_state=42,
            stratify=xdf_train_full['target']
        )
        print("Using stratified split")
        logger.info("Using stratified split")
    except ValueError:
        print("Stratified split failed (rare classes), using random split")
        logger.warning("Stratified split failed (rare classes), using random split")
        xdf_dset, xdf_dset_test = train_test_split(
            xdf_train_full,
            test_size=0.15,
            random_state=42,
            shuffle=True
        )

    # Reset indices so dataloader works correctly
    xdf_dset = xdf_dset.reset_index(drop=True)
    xdf_dset_test = xdf_dset_test.reset_index(drop=True)

    print(f"Training samples: {len(xdf_dset)}")
    print(f"Validation samples: {len(xdf_dset_test)}")

    #DATASET LOGGING
    logger.info(f"Training samples: {len(xdf_dset)}")
    logger.info(f"Validation samples: {len(xdf_dset_test)}")

    ## read_data creates the dataloaders, take target_type = 2

    train_ds, test_ds = read_data(target_type=2)

    OUTPUTS_a = len(class_names)

    logger.info(f"Number of classes: {OUTPUTS_a}")
    logger.info(f"Class names: {class_names}")
    logger.info(f"Training batches: {len(train_ds)}")
    logger.info(f"Validation batches: {len(test_ds)}")
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)


    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    #ERROR HANDLING
    try:
        train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on='f1_macro', pretrained=True)
    except Exception as e:
        logger.error("=" * 70)
        logger.error("TRAINING FAILED!")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 70)
        raise  # Re-raise the error

    # Create a quick summary file
    summary_file = f'training_summary_{NICKNAME}.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Training Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Nickname: {NICKNAME}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {n_epoch}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Dropout: 0.5\n")
        f.write(f"Weight Decay: 0.001\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training samples: {len(xdf_dset)}\n")
        f.write(f"Validation samples: {len(xdf_dset_test)}\n")
        f.write(f"Number of classes: {OUTPUTS_a}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Full log: {log_filename}\n")
        f.write(f"Model: model_{NICKNAME}.pt\n")
        f.write(f"Results: results_{NICKNAME}.xlsx\n")

    logger.info(f"Summary saved to: {summary_file}")

