import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd
from torchmetrics import Precision
from torchmetrics import Recall
from torchmetrics import F1Score
from torchmetrics import ConfusionMatrix


import argparse
import logging
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setting up some basic configs for enabling logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device, epoch_no, hook):
    logger.info(f"Epoch: {epoch_no} - Testing Model on Complete Testing Dataset!")
    model.eval()
    hook.set_mode(smd.modes.EVAL) # setting the  debugger hook mode to EVAL
    running_loss = 0
    running_corrects = 0
    pred_list = []
    target_list = []
    with torch.no_grad(): #We do not want to caluculate gradients while testing
        for inputs, labels in test_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)
            running_loss += loss.item() * inputs.size(0) #calculate the running loss
            running_corrects += pred.eq(labels.view_as(pred)).sum().item() #calculate the running corrects
            pred_list.append(pred)
            target_list.append(labels)
        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects/ len(test_loader.dataset)
        logger.info( "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_acc
        ))
        logger.info("Starting calculating other metrics for Training phase")
        total_pred =  torch.cat(pred_list, dim=1)
        total_target =  torch.cat(target_list, dim=1)
        calculate_metrics( total_pred, total_target, 9 )


def train(model, train_loader, criterion, optimizer, device, epoch_no, hook):
    logger.info(f"Epoch: {epoch_no} - Training Model on Complete Training Dataset!")
    model.train()
    hook.set_mode(smd.modes.TRAIN) # setting the  debugger hook mode to TRAIN
    running_loss = 0
    running_corrects = 0
    running_samples = 0
    pred_list = []
    target_list = []
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        pred = outputs.argmax(dim=1,  keepdim=True)
        running_loss += loss.item() * inputs.size(0) #calculate the running loss
        running_corrects += pred.eq(labels.view_as(pred)).sum().item() #calculate the running corrects
        running_samples += len(inputs) #keep count of running samples
        pred_list.append(pred)
        target_list.append(labels)
        loss.backward()
        optimizer.step()
        if running_samples %500 == 0:
            logger.info("\nTrain set:  [{}/{} ({:.0f}%)]\t Loss: {:.2f}\tAccuracy: {}/{} ({:.2f}%)".format(
                running_samples,
                len(train_loader.dataset),
                100.0 * (running_samples / len(train_loader.dataset)),
                loss.item(),
                running_corrects,
                running_samples,
                100.0*(running_corrects/ running_samples)
            ))

    total_loss = running_loss / len(train_loader.dataset)
    total_acc = running_corrects/ len(train_loader.dataset)
    logger.info( "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        total_loss, running_corrects, len(train_loader.dataset), 100.0 * total_acc
    ))
    logger.info("Starting calculating other metrics for Training phase")
    total_pred =  torch.cat(pred_list, dim=1)
    total_target =  torch.cat(target_list, dim=1)
    calculate_metrics( total_pred, total_target, 9 )
    return model
    
def net():
    model = models.resnet50(pretrained = True) #using a pretrained resnet50 model with 50 layers
    
    for param in model.parameters():
        param.requires_grad = False #Freezing all the Conv layers
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential( nn.Linear( num_features, 256), #Adding our own fully connected layers
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 9),
                             nn.ReLU(inplace = True) # output should have 9 nodes as we have 9 classes of plant images.
                            )
    return model

def create_data_loaders(data, batch_size):
    
    train_dataset_path = os.path.join(data, "train")
    val_dataset_path = os.path.join(data, "val")
    test_dataset_path = os.path.join(data, "test")
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Using the standard values of ImageNet dataset as plant dataset is similar to it.
    ]) 
    
    validation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Using the standard values of ImageNet dataset as plant dataset is similar to it.
    ])
    
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Using the standard values of ImageNet dataset as plant dataset is similar to it.
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transform)    
    val_dataset = torchvision.datasets.ImageFolder(root=val_dataset_path, transform=validation_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=testing_transform)
    
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size )
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size )
    
    return train_data_loader, test_data_loader, val_data_loader

def calculate_metrics( pred, target, num_classes ):
    precision = Precision(average='macro', num_classes=num_classes)
    recall = Recall(average='macro', num_classes=num_classes)
    f1_score = F1Score(num_classes=num_classes)
    confusion_matrix = ConfusionMatrix(num_classes=num_classes)
    logger.info(f" Precision: \n {precision(pred, target)}" )
    logger.info(f" Recall: \n {recall(pred, target)}" )
    logger.info(f" F1 Score: \n {f1_score(pred, target)}" )
    logger.info(f" Confusion Matrix: \n {confusion_matrix(pred, target)}" )

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    logger.info(f"Hyperparameters : LR: {args.lr},  Eps: {args.eps}, Weight-decay: {args.weight_decay}, Batch Size: {args.batch_size}, Epoch: {args.epochs}")
    logger.info(f"Data Dir Path: {args.data_dir}")
    logger.info(f"Model Dir  Path: {args.model_dir}")
    logger.info(f"Output Dir  Path: {args.output_dir}")
    
    model=net()
    model = model.to(device)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    
    train_data_loader, test_data_loader, val_data_loader = create_data_loaders(args.data_dir, args.batch_size )
    
    loss_criterion = nn.CrossEntropyLoss() #Using Categorical cross entropy as we are performing multi-class classification.
    hook.register_loss(loss_criterion) # adding hook for loss criterion as well
    #Using AdamW as it yieds usually better performance then Adam in most cases due to the way it uses weight decay in computations
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, eps= args.eps, weight_decay = args.weight_decay)
 
    #Adding in the epoch to train and test/validate for the same epoch at the same time.
    for epoch_no in range(1, args.epochs +1 ):
        logger.info(f"Epoch {epoch_no} - Starting Training phase.")
        model=train(model, train_data_loader, loss_criterion, optimizer, device, epoch_no, hook)
        logger.info(f"Epoch {epoch_no} - Starting Validation phase.")
        test(model, val_data_loader, loss_criterion, device, epoch_no, hook)
    
    logger.info("Starting to perform Testing of the trained model on the Test dataset.")
    test(model, test_data_loader, loss_criterion, device, 1,hook)
    logger.info("Completed Testing phase of the trained model on the Test dataset.")
    
    logger.info("Starting to Save the Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info("Completed Saving the Model")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    Adding all the hyperparameters needed to use to train your model.
    '''
    parser.add_argument(  "--batch_size", type = int, default = 64, metavar = "N", help = "input batch size for training (default: 64)" )
    parser.add_argument( "--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)"    )
    parser.add_argument( "--lr", type = float, default = 0.1, metavar = "LR", help = "learning rate (default: 1.0)" )
    parser.add_argument( "--eps", type=float, default=1e-8, metavar="EPS", help="eps (default: 1e-8)" )
    parser.add_argument( "--weight_decay", type=float, default=1e-2, metavar="WEIGHT-DECAY", help="weight decay coefficient (default 1e-2)" )
                        
    # Using sagemaker OS Environ's channels to locate training data, model dir and output dir to save in S3 bucket
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    args=parser.parse_args()
    
    main(args)