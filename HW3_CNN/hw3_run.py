
# %%
import sys
import logging

nblog = open("nb.log", "a+")
sys.stdout.echo = nblog
sys.stderr.echo = nblog

get_ipython().log.handlers[0].stream = nblog
get_ipython().log.setLevel(logging.INFO)

%autosave 5

# %% [markdown]
# ## HW3 Image Classification
# #### Solve image classification with convolutional neural networks(CNN).
# #### If you have any questions, please contact the TAs via TA hours, NTU COOL, or email to mlta-2023-spring@googlegroups.com

# %% [markdown]
# ### Import Packages

# %%
_exp_name = "hw3"

# %%
# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random

# %%
myseed = 1314520  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# %% [markdown]
# ### Transforms

# %%
# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
	transforms.Resize(256),
    transforms.CenterCrop(224),  # Resize the image into a fixed shape (height = width = 224)
    transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],       # Normalize to ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # Random crop & resize
    transforms.RandomHorizontalFlip(p=0.5),                # Flip horizontally
    transforms.RandomRotation(15),                         # Random rotation ±15°
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.1),                     # 10% chance to grayscale
    transforms.ToTensor(),                                 # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],       # Normalize to ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# %% [markdown]
# ### Datasets

# %%
class FoodDataset(Dataset):

    def __init__(self, path=None, tfm=test_tfm, files=None, labels=None):
        super(FoodDataset).__init__()
        self.transform = tfm

        if files is not None:
            self.files = files
            self.labels = labels  # 训练用
        else:
            # 自动加载路径
            self.files = sorted([
                os.path.join(path, x) for x in os.listdir(path)
                if x.endswith(".jpg")
            ])
            try:
                # 推测标签（用于 train/valid）
                self.labels = [int(f.split("/")[-1].split("_")[0]) for f in self.files]
            except:
                self.labels = [-1] * len(self.files)  # 测试集无标签

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        if self.transform:
            im = self.transform(im)

        label = self.labels[idx]
        return im, label

# %% [markdown]
# ### Model

# %%
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            # nn.Linear(512*4*4, 1024),
            nn.Linear(512*7*7, 1024),    # input 224x224
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    
from torchvision.models import mobilenet_v2

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = mobilenet_v2(weights=None)  # or weights='DEFAULT' if允许预训练
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)

from torchvision.models import efficientnet_b0

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)
    
from torchvision.models import resnet18

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

# %% [markdown]
# ### Configurations

# %%
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
# model = Classifier().to(device)
# Using Kfold, we will new a model for each fold.

# Using custom model
# from torchvision.models import resnet50, resnet18
# model = resnet18(weights=None).to(device)

# The number of batch size.
batch_size = 16

# The number of training epochs.
n_epochs = 200

# If no improvement in 'patience' epochs, early stop.
patience = 10

# # For the classification task, we use cross-entropy as the measurement of performance.
# criterion = nn.CrossEntropyLoss()

# # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# %% [markdown]
# ### Dataloader

# %%
# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
# train_set = FoodDataset("./mlhw3/train", tfm=train_tfm)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
# valid_set = FoodDataset("./mlhw3/valid", tfm=test_tfm)
# valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
# print(f"Train set size: {len(train_set)}, Valid set size: {len(valid_set)}")

# Going to use KFold

# %% [markdown]
# ### Start Training

# %%
def trainer(fold, train_loader, valid_loader, model, device, writer, val_accs, model_name):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    stale = 0
    best_acc = 0

    for epoch in range(n_epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        train_loader_bar = tqdm(train_loader, desc=f"Fold {fold}, Model {model_name} - Training Epoch {epoch + 1}/{n_epochs}")

        for batch in train_loader_bar:

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()
            #print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            train_loader_bar.set_postfix(
                loss=f"{sum(train_loss)/len(train_loss):.5f}",
                acc=f"{sum(train_accs)/len(train_accs):.5f}"
            )

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        valid_loader_bar = tqdm(valid_loader, desc=f"Fold {fold}, Model {model_name} - Validation Epoch {epoch + 1}/{n_epochs}")

        # Iterate the validation set by batches.
        for batch in valid_loader_bar:

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            valid_loader_bar.set_postfix(
                loss=f"{sum(valid_loss)/len(valid_loss):.5f}",
                acc=f"{sum(valid_accs)/len(valid_accs):.5f}"
            )
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        # — Log to TensorBoard under fold subdir —
        writer.add_scalar(f"Fold{fold}/Loss/Train", train_loss, epoch)
        writer.add_scalar(f"Fold{fold}/Loss/Val",   valid_loss, epoch)
        writer.add_scalar(f"Fold{fold}/Acc/Train",  train_acc, epoch)
        writer.add_scalar(f"Fold{fold}/Acc/Val",    valid_acc, epoch)
        writer.add_scalar(f"Fold{fold}/LR", optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(valid_loss)  # Step the scheduler with validation loss

        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save models
        if valid_acc > best_acc:
            # if len(val_accs) == 0 or valid_acc > max(val_accs):
            #     torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")  # save the best model
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"{_exp_name}_fold{fold}_model{model_name}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
    return best_acc

# %%
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()

models = [
    Classifier,
    ResNet18Classifier,
    MobileNetV2Classifier,
    EfficientNetB0Classifier
]
    
all_data_paths = ["./mlhw3/train", "./mlhw3/valid"]
all_data_files = []
all_data_labels = []
for path in all_data_paths:
    dataset = FoodDataset(path, tfm=None)  # 不设 transform
    all_data_files.extend(dataset.files)
    all_data_labels.extend(dataset.labels)

print(f"[info] Train set size: {len(all_data_files)}")

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=myseed)
val_accs = [[] for _ in range(len(models))]

writers = [SummaryWriter(f"./runs/{_exp_name}_model{model.__name__}") for model in models]

for fold, (train_idx, valid_idx) in enumerate(kf.split(all_data_files)):
    print(f"=== Fold {fold+1}/{n_splits} ===")

    # 构建两个 Dataset，并传入对应 transform
    train_dataset = FoodDataset(files=[all_data_files[i] for i in train_idx],
                            labels=[all_data_labels[i] for i in train_idx],
                            tfm=train_tfm)

    valid_dataset = FoodDataset(files=[all_data_files[i] for i in valid_idx],
                            labels=[all_data_labels[i] for i in valid_idx],
                            tfm=test_tfm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, pin_memory=True, num_workers=4)
    
    print(f"[info] Train set size: {len(train_dataset)}, Valid set size: {len(valid_dataset)}")
    
    # 对每个模型分别训练
    for model_id, model_class in enumerate(models):
        print(f"\n--- Training Model {model_id} ({model_class.__name__}) ---")
        model = model_class().to(device)
        
        # 开始训练
        acc = trainer(fold=fold+1,
                      train_loader=train_loader,
                      valid_loader=valid_loader,
                      model=model,
                      device=device,
                      writer=writers[model_id],
                      val_accs=val_accs[model_id],
                      model_name=model_class.__name__)
        
        val_accs[model_id].append(acc)


# %% [markdown]
# ### Dataloader for test

# %%
# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set_std = FoodDataset("./mlhw3/test", tfm=test_tfm)
test_loader_std = DataLoader(test_set_std, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
tta_repeats = 4
test_set_tta = ConcatDataset([
    FoodDataset("./mlhw3/test", tfm=train_tfm) for _ in range(tta_repeats)
])
test_loader_tta = DataLoader(test_set_tta, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
print(f"Test set size: {len(test_set_std)}, TTA set size: {len(test_set_tta)}")

# %% [markdown]
# ### Testing and generate prediction CSV

# %%
model_paths = [f"{_exp_name}_fold{fold+1}_model{model.__name__}_best.ckpt" for fold in range(n_splits) for model in models]

# Soft voting accumulation
logits_all_std, logits_all_tta = [], []

for path in model_paths:
    model = Classifier().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    # test_tfm 推理
    logits_std = []
    with torch.no_grad():
        for data, _ in test_loader_std:
            logits = model(data.to(device))
            logits_std.append(logits.cpu())
    logits_all_std.append(torch.cat(logits_std))  # [N, C]

    # train_tfm (TTA) 推理
    logits_tta = []
    with torch.no_grad():
        for data, _ in test_loader_tta:
            logits = model(data.to(device))
            logits_tta.append(logits.cpu())
    logits_cat = torch.cat(logits_tta)  # [N * tta_repeats, C]

    # 每 tta_repeats 取平均
    N = len(test_set_std)
    logits_avg = logits_cat.view(tta_repeats, N, -1).mean(dim=0)  # [N, C]
    logits_all_tta.append(logits_avg)



# %%
#create test csv
logits_std_mean = torch.stack(logits_all_std).mean(dim=0)
logits_tta_mean = torch.stack(logits_all_tta).mean(dim=0)

# 加权融合
final_logits = logits_std_mean * 0.8 + logits_tta_mean * 0.2
final_preds = final_logits.argmax(dim=1).numpy()

# 写入 CSV
df = pd.DataFrame()
df["Id"] = [f"{i:04d}" for i in range(len(test_set_std))]
df["Category"] = final_preds
df.to_csv("submission_tta_loader.csv", index=False)


