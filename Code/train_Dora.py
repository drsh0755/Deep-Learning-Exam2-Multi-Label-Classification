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


n_epoch = 30
BATCH_SIZE = 32
LR = 0.0001

## Image processing
CHANNELS = 3
IMAGE_SIZE = 128

NICKNAME = "Dora"

# ==================== MULTI-MODEL CONFIGURATION ====================
# To train multiple models, uncomment the full list below:
# MODELS_TO_TRAIN = ['resnet34', 'resnet50', 'vgg19', 'alexnet']

# Current configuration: ResNet50 only
MODELS_TO_TRAIN = ['resnet50']
CURRENT_MODEL = 'resnet50'  # Will be updated during training loop

# FUTURE USE: To train different models, modify MODELS_TO_TRAIN:
# - Single model:   ['resnet50']
# - Multiple:       ['resnet34', 'resnet50']
# - All models:     ['resnet34', 'resnet50', 'vgg19', 'alexnet']
# ===================================================================

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.35
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
    def __init__(self, list_IDs, type_data, target_type, use_mixup=False):
        #Initialization'
        self.use_mixup = use_mixup
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

        # Mixup augmentation (only for training)
        if self.type_data == 'train' and self.use_mixup and random.random() > 0.5:
            # Get another random sample
            mix_idx = random.randint(0, len(self.list_IDs) - 1)
            X2, y2 = self.__getitem__(mix_idx)

            # Mixup
            lam = np.random.beta(0.2, 0.2)
            X = lam * X + (1 - lam) * X2
            y = lam * y + (1 - lam) * y2

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

    training_set = Dataset(partition['train'], 'train', target_type, use_mixup=False)
    training_generator = data.DataLoader(training_set, **params)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return training_generator, test_generator


def save_model(model):
    # Save with both model-specific and submission-ready names

    # Model-specific name (for reference)
    summary_filename = f'summary_{NICKNAME}_{CURRENT_MODEL}.txt'
    print(model, file=open(summary_filename, "w"))
    logger.info(f"✓ Model architecture saved to: {summary_filename}")

    # Submission-ready name (overwrites each time)
    submission_filename = f'summary_{NICKNAME}.txt'
    print(model, file=open(submission_filename, "w"))
    logger.info(f"✓ Submission file saved to: {submission_filename}")


def model_definition(pretrained=True, unfreeze_stage='initial', use_focal_loss=False, label_smoothing=0.0, training_data=None):
    """
    Define model architecture with flexible unfreezing strategy

    Args:
        pretrained: Use pretrained ResNet18
        unfreeze_stage: 'initial', 'partial', or 'full'
            - 'initial': Train only layer4 + fc (epochs 0-10)
            - 'partial': Train layer3 + layer4 + fc (epochs 10-20)
            - 'full': Train all layers (epochs 20+)
        use_focal_loss: Use Focal Loss instead of BCE
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
    """

    if pretrained == True:
        # ==================== MODEL SELECTION ====================
        if CURRENT_MODEL == 'resnet34':
            model = models.resnet34(pretrained=True)
            logger.info("Using ResNet34 (21M parameters)")

        elif CURRENT_MODEL == 'resnet50':
            model = models.resnet50(pretrained=True)
            logger.info("Using ResNet50 (25M parameters)")

        elif CURRENT_MODEL == 'vgg19':
            model = models.vgg19(pretrained=True)
            logger.info("Using VGG19 (143M parameters)")

        elif CURRENT_MODEL == 'alexnet':
            model = models.alexnet(pretrained=True)
            logger.info("Using AlexNet (61M parameters)")

        else:
            raise ValueError(f"Unknown model: {CURRENT_MODEL}")
        # ========================================================

        # Implement progressive unfreezing strategy
        if unfreeze_stage == 'initial':
            # Stage 1: Freeze conv1, bn1, layer1, layer2, layer3
            # Only train layer4 and fc
            layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']

            for name, param in model.named_parameters():
                if any(layer in name for layer in layers_to_freeze):
                    param.requires_grad = False

            logger.info("🔒 Freezing Strategy: INITIAL")
            logger.info("   Frozen: conv1, bn1, layer1, layer2, layer3")
            logger.info("   Training: layer4, fc")

        elif unfreeze_stage == 'partial':
            # Stage 2: Freeze conv1, bn1, layer1, layer2
            # Train layer3, layer4, and fc
            layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2']

            for name, param in model.named_parameters():
                if any(layer in name for layer in layers_to_freeze):
                    param.requires_grad = False

            logger.info("🔓 Freezing Strategy: PARTIAL")
            logger.info("   Frozen: conv1, bn1, layer1, layer2")
            logger.info("   Training: layer3, layer4, fc")

        elif unfreeze_stage == 'full':
            # Stage 3: Train all layers
            for param in model.parameters():
                param.requires_grad = True

            logger.info("🔥 Freezing Strategy: FULL")
            logger.info("   Training: ALL LAYERS")

        else:
            raise ValueError(f"Invalid unfreeze_stage: {unfreeze_stage}")

        # Replace final fully connected layer with improved architecture
        # Handle different architectures
        if CURRENT_MODEL in ['resnet34', 'resnet50']:
            # ResNet models use 'fc'
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, OUTPUTS_a)
            )
            logger.info(f"✓ ResNet classifier: {num_features} → 512 → {OUTPUTS_a}")

        elif CURRENT_MODEL == 'vgg19':
            # VGG models use 'classifier'
            num_features = 4096  # VGG19 has 4096 features before final layer
            model.classifier = nn.Sequential(
                model.classifier[0],  # Keep first Linear layer
                nn.ReLU(True),
                nn.Dropout(0.3),
                model.classifier[3],  # Keep second Linear layer
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, OUTPUTS_a)
            )
            logger.info(f"✓ VGG classifier: 4096 → 512 → {OUTPUTS_a}")

        elif CURRENT_MODEL == 'alexnet':
            # AlexNet uses 'classifier'
            num_features = 4096  # AlexNet has 4096 features
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(9216, 4096),  # AlexNet's first layer
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, OUTPUTS_a)
            )
            logger.info(f"✓ AlexNet classifier: 9216 → 4096 → 512 → {OUTPUTS_a}")

    else:
        # Use simple CNN if not pretrained
        model = CNN(OUTPUTS_a)
        logger.info("Using simple CNN (non-pretrained)")

    # Move model to device
    model = model.to(device)

    # Optimizer: AdamW with weight decay for regularization
    # Reduced weight decay from 0.01 to 0.001 for better learning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-3,  # 0.001 instead of 0.01
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Loss function selection
    if use_focal_loss:
        # Focal Loss: Better for class imbalance
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, pred, target):
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred, target, reduction='none'
                )
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                return focal_loss.mean()

        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        logger.info("Loss: Focal Loss (alpha=0.25, gamma=2.0)")



    elif label_smoothing > 0:

        # BCE with Label Smoothing (NO class weights due to extreme imbalance)

        class BCEWithLogitsLossSmooth(nn.Module):

            def __init__(self, smoothing=0.1):
                super().__init__()

                self.smoothing = smoothing

            def forward(self, pred, target):
                # Smooth labels: 1 → (1 - smoothing), 0 → smoothing/2

                target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing

                return nn.functional.binary_cross_entropy_with_logits(pred, target_smooth)

        criterion = BCEWithLogitsLossSmooth(smoothing=label_smoothing)

        logger.info(f"Loss: BCE with Label Smoothing (smoothing={label_smoothing})")

    else:
        # Standard BCE Loss
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Loss: BCEWithLogitsLoss")

    # Learning rate scheduler: ReduceLROnPlateau
    # More patient than before (patience=5 instead of 3)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize F1 score
        factor=0.5,  # Reduce LR by 50%
        patience=5,  # Wait 5 epochs before reducing
        min_lr=1e-7  # Don't go below this LR
    )

    # Alternative: Cosine Annealing with Warm Restarts
    # Uncomment to use instead of ReduceLROnPlateau
    # from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=10,              # Initial restart period
    #     T_mult=2,            # Period multiplier after restart
    #     eta_min=1e-6         # Minimum learning rate
    # )
    # logger.info("Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")

    # Save model architecture
    save_model(model)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params

    # Log model information
    logger.info("=" * 70)
    logger.info("MODEL ARCHITECTURE")
    logger.info("=" * 70)
    logger.info(f"Model: {CURRENT_MODEL.upper()} (pretrained={pretrained})")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
    logger.info(f"Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)")
    logger.info("=" * 70)
    logger.info("OPTIMIZER & SCHEDULER")
    logger.info("=" * 70)
    logger.info(f"Optimizer: AdamW")
    logger.info(f"  - Learning Rate: {LR}")
    logger.info(f"  - Weight Decay: {optimizer.param_groups[0]['weight_decay']}")
    logger.info(f"  - Betas: (0.9, 0.999)")
    logger.info(f"Scheduler: ReduceLROnPlateau")
    logger.info(f"  - Mode: max (maximize F1)")
    logger.info(f"  - Factor: 0.5 (reduce by 50%)")
    logger.info(f"  - Patience: 5 epochs")
    logger.info(f"  - Min LR: 1e-7")
    logger.info("=" * 70)

    return model, optimizer, criterion, scheduler


def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on,
                   pretrained=False, unfreeze_stage='initial',
                   use_focal_loss=False, label_smoothing=0.0,
                   stage_epochs=10):
    # Use a breakpoint in the code line below to debug your script.

    model, optimizer, criterion, scheduler = model_definition(
        pretrained=pretrained,
        unfreeze_stage=unfreeze_stage,
        use_focal_loss=use_focal_loss,
        label_smoothing=label_smoothing
    )

    cont = 0
    train_loss_item = list([])
    test_loss_item = list([])

    pred_labels_per_hist = list([])

    model.phase = 0

    met_test_best = 0
    # Early stopping variables
    patience_counter = 0
    patience_limit = 5  # Stop if no improvement for 5 epochs
    for epoch in range(stage_epochs):
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


def find_optimal_threshold(pred_logits, real_labels, thresholds=np.arange(0.3, 0.7, 0.05)):
    """
    Find optimal threshold for each class independently
    Returns array of optimal thresholds per class
    """
    best_thresholds = []

    for class_idx in range(pred_logits.shape[1]):
        best_f1 = 0
        best_thresh = 0.5

        for threshold in thresholds:
            pred_binary = (pred_logits[:, class_idx] >= threshold).astype(int)
            f1 = f1_score(real_labels[:, class_idx], pred_binary, average='binary', zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = threshold

        best_thresholds.append(best_thresh)

    logger.info(f"✓ Optimal thresholds per class: {[f'{t:.2f}' for t in best_thresholds]}")
    return np.array(best_thresholds)

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


def calculate_class_weights(xdf_train):
    """
    Calculate weights for imbalanced classes using inverse frequency
    """
    class_counts = np.zeros(10)

    for target_str in xdf_train['target_class']:
        labels = [int(x) for x in target_str.split(',')]
        class_counts += np.array(labels)

    # Inverse frequency weighting
    total_samples = len(xdf_train)
    weights = total_samples / (10 * class_counts + 1e-6)  # Small epsilon to avoid division by zero

    # Normalize weights so they average to 1
    weights = weights / weights.mean()

    weights_tensor = torch.FloatTensor(weights).to(device)

    logger.info(f"Class frequencies: {class_counts.astype(int)}")
    logger.info(f"Class weights: {[f'{w:.3f}' for w in weights]}")

    return weights_tensor


def train_progressive_unfreezing(train_ds, test_ds, list_of_metrics, list_of_agg, save_on='f1_macro'):
    """
    Train model with progressive unfreezing strategy in 3 stages
    """

    logger.info("=" * 70)
    logger.info("🚀 STARTING PROGRESSIVE UNFREEZING TRAINING")
    logger.info("=" * 70)

    best_f1_overall = 0.0

    # ==================== STAGE 1: Initial Training ====================
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: Training final layers only (layer4 + fc)")
    logger.info("Epochs: 15")
    logger.info("=" * 70)

    model, optimizer, criterion, scheduler = model_definition(
        pretrained=True,
        unfreeze_stage='initial',
        use_focal_loss=True,
        label_smoothing=0.0,  # Use label smoothing
        training_data=xdf_dset  # Pass training data for class weights
    )

    # Train stage 1
    best_f1_stage1 = train_single_stage(
        model, optimizer, criterion, scheduler,
        train_ds, test_ds, list_of_metrics, list_of_agg,
        save_on=save_on, stage_epochs=15, stage_name="Stage1"
    )

    logger.info(f"✓ Stage 1 completed! Best F1: {best_f1_stage1:.5f}")
    best_f1_overall = max(best_f1_overall, best_f1_stage1)

    # ==================== STAGE 2: Partial Unfreezing ====================
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: Unfreezing more layers (layer3 + layer4 + fc)")
    logger.info("Epochs: 15")
    logger.info("=" * 70)

    model, optimizer, criterion, scheduler = model_definition(
        pretrained=True,
        unfreeze_stage='partial',
        use_focal_loss=True,
        label_smoothing=0.0,
        training_data=xdf_dset
    )

    # Load best model from stage 1 WITH MODEL NAME
    model.load_state_dict(torch.load(f'model_{NICKNAME}_{CURRENT_MODEL}.pt'))
    logger.info("✓ Loaded best model from Stage 1")

    # Reduce learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR * 0.5  # Half the learning rate
    logger.info(f"✓ Reduced learning rate to {LR * 0.5}")

    # Train stage 2
    best_f1_stage2 = train_single_stage(
        model, optimizer, criterion, scheduler,
        train_ds, test_ds, list_of_metrics, list_of_agg,
        save_on=save_on, stage_epochs=15, stage_name="Stage2"
    )

    logger.info(f"✓ Stage 2 completed! Best F1: {best_f1_stage2:.5f}")
    best_f1_overall = max(best_f1_overall, best_f1_stage2)

    # ==================== STAGE 3: Full Unfreezing ====================
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: Fine-tuning all layers")
    logger.info("Epochs: 15")
    logger.info("=" * 70)

    model, optimizer, criterion, scheduler = model_definition(
        pretrained=True,
        unfreeze_stage='full',
        use_focal_loss=True,
        label_smoothing=0.0,
        training_data=xdf_dset
    )

    # Load best model from stage 2
    model.load_state_dict(torch.load(f'model_{NICKNAME}_{CURRENT_MODEL}.pt'))  # ← ADD _{CURRENT_MODEL}
    logger.info("✓ Loaded best model from Stage 2")

    # Further reduce learning rate for full fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR * 0.1  # Even lower LR
    logger.info(f"✓ Reduced learning rate to {LR * 0.1}")

    # Train stage 3
    best_f1_stage3 = train_single_stage(
        model, optimizer, criterion, scheduler,
        train_ds, test_ds, list_of_metrics, list_of_agg,
        save_on=save_on, stage_epochs=15, stage_name="Stage3"
    )

    logger.info(f"✓ Stage 3 completed! Best F1: {best_f1_stage3:.5f}")
    best_f1_overall = max(best_f1_overall, best_f1_stage3)

    # ==================== FINAL SUMMARY ====================
    logger.info("\n" + "=" * 70)
    logger.info("🎉 PROGRESSIVE UNFREEZING COMPLETED!")
    logger.info("=" * 70)
    logger.info(f"Stage 1 Best F1: {best_f1_stage1:.5f}")
    logger.info(f"Stage 2 Best F1: {best_f1_stage2:.5f}")
    logger.info(f"Stage 3 Best F1: {best_f1_stage3:.5f}")
    logger.info(f"Overall Best F1: {best_f1_overall:.5f}")
    logger.info("=" * 70)

    return best_f1_overall


def train_single_stage(model, optimizer, criterion, scheduler,
                       train_ds, test_ds, list_of_metrics, list_of_agg,
                       save_on='f1_macro', stage_epochs=10, stage_name=""):
    """
    Train a single stage of progressive unfreezing
    Returns best F1 score achieved in this stage
    """

    met_test_best = 0
    patience_counter = 0
    patience_limit = 5

    for epoch in range(stage_epochs):
        # ========== TRAINING PHASE ==========
        model.train()
        train_loss, steps_train = 0, 0
        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        with tqdm(total=len(train_ds), desc=f"{stage_name} Epoch {epoch}") as pbar:
            for xdata, xtarget in train_ds:
                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()
                output = model(xdata)
                loss = criterion(output, xtarget)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()
                steps_train += 1

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

                pbar.update(1)
                pbar.set_postfix_str(f"Train Loss: {train_loss / steps_train:.5f}")

        # Calculate training metrics
        pred_labels = pred_logits[1:].copy()
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)
        avg_train_loss = train_loss / steps_train

        # ========== VALIDATION PHASE ==========
        model.eval()
        test_loss, steps_test = 0, 0
        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        with torch.no_grad():
            with tqdm(total=len(test_ds), desc=f"{stage_name} Validation {epoch}") as pbar:
                for xdata, xtarget in test_ds:
                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    output = model(xdata)
                    loss = criterion(output, xtarget)

                    test_loss += loss.item()
                    steps_test += 1

                    pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

                    pbar.update(1)
                    pbar.set_postfix_str(f"Val Loss: {test_loss / steps_test:.5f}")

                # Calculate validation metrics with threshold optimization
                # ALWAYS optimize thresholds (removed epoch >= 5 check)
                optimal_thresholds = find_optimal_threshold(pred_logits[1:], real_labels[1:])

                # Apply per-class thresholds
                pred_labels = np.zeros_like(pred_logits[1:])
                for i, thresh in enumerate(optimal_thresholds):
                    pred_labels[:, i] = (pred_logits[1:, i] >= thresh).astype(int)

                test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)
                avg_test_loss = test_loss / steps_test

                # Get F1 score
                met_test = test_metrics[save_on]

                # Logging
                xstrres = f"{stage_name} Epoch {epoch}: "
                for met, dat in train_metrics.items():
                    xstrres += f' Train {met} {dat:.5f}'
                xstrres += " - "
                for met, dat in test_metrics.items():
                    xstrres += f' Val {met} {dat:.5f}'

                print(xstrres)
                logger.info(xstrres)
                logger.info(
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} | Train loss: {avg_train_loss:.5f} | Val loss: {avg_test_loss:.5f}")

                # Update scheduler
                scheduler.step(met_test)

                # Early stopping check
                if met_test > met_test_best:
                    patience_counter = 0

                    # Save with both model-specific and submission-ready names

                    # Model-specific files (for reference)
                    model_filename = f"model_{NICKNAME}_{CURRENT_MODEL}.pt"
                    torch.save(model.state_dict(), model_filename)
                    logger.info(f"✓ Saved model: {model_filename}")

                    # Submission-ready model (overwrites each time)
                    submission_model = f"model_{NICKNAME}.pt"
                    torch.save(model.state_dict(), submission_model)
                    logger.info(f"✓ Submission model saved to: {submission_model}")

                    # SAVE optimal thresholds - both versions
                    threshold_filename = f'optimal_thresholds_{NICKNAME}_{CURRENT_MODEL}.npy'
                    np.save(threshold_filename, optimal_thresholds)

                    submission_threshold = f'optimal_thresholds_{NICKNAME}.npy'
                    np.save(submission_threshold, optimal_thresholds)
                    logger.info(f"✓ Saved optimal thresholds: {optimal_thresholds}")

                    # Save results - both versions
                    xdf_dset_results = xdf_dset_test.copy()
                    xfinal_pred_labels = []
                    for i in range(len(pred_labels)):
                        joined_string = ",".join(str(int(e)) for e in pred_labels[i])
                        xfinal_pred_labels.append(joined_string)
                    xdf_dset_results['results'] = xfinal_pred_labels

                    # Model-specific results
                    xdf_dset_results.to_excel(f'results_{NICKNAME}_{CURRENT_MODEL}.xlsx', index=False)

                    # Submission-ready results (overwrites each time)
                    xdf_dset_results.to_excel(f'results_{NICKNAME}.xlsx', index=False)

                    # logger.info(f"✓ Saved model: {model_filename}")
                    # logger.info(f"✓ Saved thresholds: {threshold_filename}")
                    # logger.info(f"✓ Saved results: results_{NICKNAME}_{CURRENT_MODEL}.xlsx")

                    logger.info(f"✓ NEW BEST! F1: {met_test:.5f} (previous: {met_test_best:.5f})")
                    met_test_best = met_test
                else:
                    patience_counter += 1
                    logger.warning(f"No improvement {patience_counter}/{patience_limit}")

                    if patience_counter >= patience_limit:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

    return met_test_best


if __name__ == '__main__':

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    print("\n" + "=" * 70)
    print("⚠️  DATA VALIDATION CHECK")
    print("=" * 70)
    print(f"Total rows in Excel: {len(xdf_data)}")
    print(f"\nSplit distribution:")
    print(xdf_data['split'].value_counts())
    print(f"\nUnique values in 'split' column: {xdf_data['split'].unique()}")

    # Check for duplicates
    duplicates = xdf_data.duplicated(subset=['id']).sum()
    print(f"\nDuplicate rows (by 'id'): {duplicates}")

    if duplicates > 0:
        print(f"⚠️  WARNING: Found {duplicates} duplicate image IDs!")
        print("Removing duplicates...")
        xdf_data = xdf_data.drop_duplicates(subset=['id'], keep='first')
        print(f"✓ After deduplication: {len(xdf_data)} rows")

    print("=" * 70 + "\n")

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

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
    xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()

    # Reset indices so dataloader works correctly
    xdf_dset = xdf_dset.reset_index(drop=True)
    xdf_dset_test = xdf_dset_test.reset_index(drop=True)

    print(f"✓ Using ACTUAL test split for validation")
    print(f"Training samples: {len(xdf_dset)}")
    print(f"Validation samples: {len(xdf_dset_test)}")

    logger.info(f"Training samples: {len(xdf_dset)}")
    logger.info(f"Validation samples: {len(xdf_dset_test)}")


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

    # ERROR HANDLING - MULTI-MODEL TRAINING
    best_model_name = None
    best_model_f1 = 0.0
    model_results = {}

    logger.info("\n" + "=" * 70)
    logger.info("🚀 MULTI-MODEL SEQUENTIAL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Models to train: {MODELS_TO_TRAIN}")
    logger.info(f"Total models: {len(MODELS_TO_TRAIN)}")
    logger.info("=" * 70)

    for idx, model_name in enumerate(MODELS_TO_TRAIN, 1):
        CURRENT_MODEL = model_name

        logger.info("\n" + "🔥" * 35)
        logger.info(f"📊 MODEL {idx}/{len(MODELS_TO_TRAIN)}: {CURRENT_MODEL.upper()}")
        logger.info("🔥" * 35 + "\n")

        try:
            # Train this model with progressive unfreezing
            best_f1 = train_progressive_unfreezing(
                train_ds, test_ds, list_of_metrics, list_of_agg, save_on='f1_macro'
            )

            # Track results
            model_results[model_name] = best_f1

            # Update best model tracker
            if best_f1 > best_model_f1:
                best_model_f1 = best_f1
                best_model_name = model_name

            logger.info("\n" + "=" * 70)
            logger.info(f"✅ {model_name.upper()} TRAINING COMPLETED!")
            logger.info(f"   Best F1 Score: {best_f1:.5f}")
            logger.info(f"   Files saved:")
            logger.info(f"     - model_{NICKNAME}_{model_name}.pt")
            logger.info(f"     - optimal_thresholds_{NICKNAME}_{model_name}.npy")
            logger.info(f"     - results_{NICKNAME}_{model_name}.xlsx")
            logger.info(f"     - summary_{NICKNAME}_{model_name}.txt")
            logger.info("=" * 70)

        except Exception as e:
            logger.error("\n" + "=" * 70)
            logger.error(f"❌ {model_name.upper()} TRAINING FAILED!")
            logger.error(f"Error: {str(e)}")
            logger.error("=" * 70)
            model_results[model_name] = 0.0
            continue  # Continue with next model

    # ==================== FINAL SUMMARY ====================
    logger.info("\n" + "🎉" * 35)
    logger.info("🏆 ALL MODELS TRAINING COMPLETED! 🏆")
    logger.info("🎉" * 35)

    logger.info("\n" + "=" * 70)
    logger.info("📊 FINAL RESULTS SUMMARY")
    logger.info("=" * 70)

    # Sort results by F1 score (best to worst)
    sorted_results = sorted(model_results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Model':<15} {'F1 Score':<12} {'Status'}")
    print("-" * 70)
    for rank, (model_name, f1_score) in enumerate(sorted_results, 1):
        status = "⭐ BEST" if model_name == best_model_name else ""
        print(f"#{rank:<5} {model_name:<15} {f1_score:.5f}      {status}")
        logger.info(f"#{rank:<5} {model_name:<15} {f1_score:.5f}      {status}")

    print("=" * 70)
    logger.info("=" * 70)
    logger.info(f"🏆 WINNER: {best_model_name.upper()}")
    logger.info(f"🏆 BEST F1: {best_model_f1:.5f}")
    logger.info("=" * 70)

    # Create comprehensive summary file
    summary_file = f'FINAL_SUMMARY_{NICKNAME}.txt'
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MULTI-MODEL TRAINING - FINAL SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nickname: {NICKNAME}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total Models Trained: {len(MODELS_TO_TRAIN)}\n")
        f.write(f"Image Size: {IMAGE_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write("=" * 70 + "\n\n")

        f.write("RESULTS (Best to Worst):\n")
        f.write("-" * 70 + "\n")
        for rank, (model_name, f1_score) in enumerate(sorted_results, 1):
            status = "⭐ BEST MODEL" if model_name == best_model_name else ""
            f.write(f"  #{rank}. {model_name:<15} F1 = {f1_score:.5f}  {status}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write(f"BEST F1 SCORE: {best_model_f1:.5f}\n")
        f.write("=" * 70 + "\n\n")

        f.write("FILES CREATED FOR EACH MODEL:\n")
        f.write("-" * 70 + "\n")
        for model_name in MODELS_TO_TRAIN:
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  - model_{NICKNAME}_{model_name}.pt\n")
            f.write(f"  - optimal_thresholds_{NICKNAME}_{model_name}.npy\n")
            f.write(f"  - results_{NICKNAME}_{model_name}.xlsx\n")
            f.write(f"  - summary_{NICKNAME}_{model_name}.txt\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("TO TEST A SPECIFIC MODEL:\n")
        f.write("-" * 70 + "\n")
        f.write(f"python3 test_Dora.py --model resnet34\n")
        f.write(f"python3 test_Dora.py --model resnet50\n")
        f.write(f"python3 test_Dora.py --model vgg19\n")
        f.write(f"python3 test_Dora.py --model alexnet\n")
        f.write("\n" + "=" * 70 + "\n")

    logger.info(f"\n✓ Final summary saved to: {summary_file}")
    print(f"\n✅ All done! Check {summary_file} for complete results.")

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

