from __future__ import print_function, division
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms, models
from opensource.siamesetriplet.datasets import BalancedBatchSampler
from opensource.siamesetriplet.losses import  OnlineTripletLossV3, OnlineTripletLossV4, OnlineTripletLossV5
from extended_model import  HowGoodIsTheModel, LilNet, IdentityNN, InputNet, EmbeddingNet
from sklearn import svm
from six.moves import urllib

import numpy as np
import torchvision
from torchvision.datasets import MNIST

from torchvision import transforms
import matplotlib.pyplot as plt
from opensource.siamesetriplet.metrics import AverageNonzeroTripletsMetric, SimilarityMetrics
from opensource.siamesetriplet.trainer import fit, best_model_path, best_model_file_name

from opensource.siamesetriplet.utils import SemihardNegativeTripletSelector, RandomNegativeTripletSelector, \
    AllPositivePairSelector, AllTripletSelector, HardestNegativeTripletSelector, AllPositivePairSelector
from opensource.trainer import validate_recognition_task

import sys
from constants import *
import argparse

parser = argparse.ArgumentParser(description='Triplet loss in practice')
parser.add_argument('-m','--mode', help='set to "1" to use a path corresponding to main data \n '
                                        'or "0" to use a path corresponding to a small subset'
                                        ' of data usually for test. see "base_path" in code', default=0)
parser.add_argument('-p','--pretrained', help='set to 0 to use the vanilla Embedding Net model or set to 1 to use the '
                                              'resenet model with InputNet, IdentityNet and LilNet, '
                                              'only vanilla model is supported on the mnist dataset. '
                                              'To support resnet + Lilnet, adjust the input channels on LilNet, '
                                              'and InputNet accordingly on mnist', default=0)
parser.add_argument('-d','--data', help='set to "mnist" to use mnist dataset or "xray" to use the '
                                        'kaggle chest xray data', default='xray')
parser.add_argument('-t','--resume', help='Used to reload a previously saved model to continue training', default=0)
args = parser.parse_args()


if args.mode == 1:
	is_live = True
if args.pretrained == 0:
	use_vanilla_model = True
if args.resume == 1:
	resume_training = True
if args.data == "mnist":
	shared_params["dataset_type"] = "mnist"

torch.autograd.set_detect_anomaly(True)
cuda = torch.cuda.is_available()

if is_live:
    n_epochs = 30
    margin = 1
    batch = 25
    base_path = main_training_sample_path
log_interval = 20
train_loader, test_loader, validate_loader, model = None, None, None, None
device = torch.device("cuda:0" if cuda else "cpu")


os.makedirs(result_folder, exist_ok=True)

def load_mnist():
    global shared_params
    mean, std = 0.1307, 0.3081

    train_dataset = MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
    test_dataset = MNIST('./data', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
    shared_params["classes"] = 10

    # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
    train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=10, n_samples=25)
    test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=10, n_samples=25)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)
    return online_train_loader, online_test_loader

	
def loader(folder, dType=1, train="", nbatch=batch):
    global shared_params
    mean, std = 0.1307, 0.3081

    transform = transforms.Compose(transforms=[transforms.Resize(320),
                                               transforms.CenterCrop(440),
                                               transforms.ToTensor(),
                                               transforms.Normalize((mean,), (std,))
                                               ])

    data_dir = os.path.join(base_path, folder)
    data_sets = datasets.ImageFolder(data_dir, transform)
    labels = torch.LongTensor([label for _, label in data_sets]) 
    batch_sampler = BalancedBatchSampler(labels, n_classes=2, n_samples=nbatch)
    if dType == 1:
        print(f"{train} image size: {len(data_sets.samples)}")
    # triplets_dataset = TripletXRay(data_sets, dType) 
    shared_params["classes"] = 2
    data_loader = DataLoader(data_sets, batch_sampler=batch_sampler, num_workers=4)
    # draw(data_loader, train)
    return data_loader


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def dummy_plot(xvalue, data, label, xlabel, ylabel,):
    assert n_epochs == len(data), f"datasize {len(data)} != epochs {n_epochs}"
    plt.plot(xvalue, data, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([x for x in range(n_epochs, -1, -5)])
    plt.legend()
    plt.title("Loss over time")

def load_data():
    test_loader = loader(test_folder, train="Test")
    train_loader = loader(train_folder, dType=True, train="traing")
    validate_loader = loader(validate_folder, train="val", nbatch=7) 
    return train_loader, test_loader, validate_loader


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 128))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            dimension = model.get_embedding(images).data.cpu().numpy()

            embeddings[k:k + len(images)] = dimension
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def draw(test_loader, title=""):
    images, labels = next(iter(test_loader))
    # print ( len(images))
    # print ( len(labels))
    out = torchvision.utils.make_grid(images)
    imshow(out, title=[label_dir[x] for x in labels])

def get_model():
    base_model = InputNet()
    trunk_model = models.resnet18(pretrained=True)
    # print(trunk_model)
    #in_channel = 389376
    output_model = LilNet()
    # summary(trunk_model,(3,299,299))
    for param in trunk_model.parameters():
        param.requires_grad = True

    base_model.nextmodel = trunk_model
    trunk_model.conv1 = IdentityNN()
    trunk_model.bn1 = IdentityNN()
    trunk_model.relu = IdentityNN()
    trunk_model.maxpool = IdentityNN()
    trunk_model.layer1 = IdentityNN()
    trunk_model.layer4 = output_model
    trunk_model.avgpool = IdentityNN()
    trunk_model.fc = IdentityNN()

    if use_vanilla_model == "mnist":
        base_model = EmbeddingNet()
    base_model.to(device)
    
    return base_model 
    
def loadmodel(model, optimizer, save_path, test_name): 
     
    global best_model_file_name
    PATH = os.path.join(save_path, test_name + best_model_file_name)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['training_loss']
    print(f"starting from epoch {epoch} with a loss of {loss}")
   

def main(tripletLoss, testName):
    global train_loader, test_loader, validate_loader, model, use_saved_model

    model = get_model()
    if shared_params["dataset_type"] == "mnist":
    	train_loader, validate_loader = load_mnist()
    	test_loader = validate_loader
    	print(f"Loading mnist data")
    else:
    	train_loader, test_loader, validate_loader = load_data()
    print (f"Using {margin}; epoch {n_epochs}")

    
    loss_function = tripletLoss(margin=margin,
                               #triplet_selector=AllTripletSelector())
                                triplet_selector=SemihardNegativeTripletSelector(margin=margin, cpu=not cuda))

    
    lr = 1e-3
    w_decay = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    if resume_training:
        loadmodel(model, optimizer, best_model_path, testName)	
    scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    training_metric = AverageNonzeroTripletsMetric()
    training_metric.set_metric_name(training_and_validation_metric)
    training_loss, validation_loss, training_metric, validation_metric = fit(testName,train_loader, validate_loader, model, loss_function,
                                         optimizer, scheduler, n_epochs, cuda, log_interval,
                                         metrics=[training_metric],
                                         pegged_metric=training_and_validation_metric,
                                         save_path=best_model_path)

    label_over_n_epochs = range(1, n_epochs + 1)
    dummy_plot(label_over_n_epochs, training_loss, "Training Loss", xlabel="Epoch", ylabel="Loss")

    dummy_plot(label_over_n_epochs, validation_loss, "Validation Loss", xlabel="Epoch", ylabel="Loss")
    plt.savefig(os.path.join(result_folder, testName+loss_graph), bbox_inches='tight')
    #plt.show()
    plt.close()
    
    dummy_plot(label_over_n_epochs, training_metric, "Training Average Nonzero Triplets", xlabel="Epoch", ylabel="Loss")

    dummy_plot(label_over_n_epochs, validation_metric, "Validation Average Nonzero Triplets", xlabel="Epoch", ylabel="Loss")

    plt.savefig(os.path.join(result_folder, testName+metric_graph), bbox_inches='tight') 
    plt.close()

def do_recognition(loader, testName, types):

    model = get_model()
    
    
    
    checkpoint = torch.load(os.path.join(best_model_path, testName+best_model_file_name ))
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    anchors = checkpoint['anchors']
    validationloss = checkpoint['validation_loss']
    trainingloss = checkpoint['training_loss']
    svm= checkpoint['svm']
    message=checkpoint['message']
    model.to(device)
    print(f"[Resuming {testName} - {types}] Loading model saved at epoch {epoch} with val_loss:{validationloss} and train_loss:{trainingloss}")
    print(message)
    
    validation_function = HowGoodIsTheModel(pairSelector=AllPositivePairSelector(balance=True),
                                            anchors = anchors,
                                            margin=margin,
                                            cuda=cuda,
                                            svm=svm)
    # Note we call model.eval inside the test_epoch func.
    validation_memo = validate_recognition_task(types, loader, model, validation_function,
                                                cuda, testName, metrics=[SimilarityMetrics()])



if "__main__" == __name__: 
    for tripletLossLayer, testname in [(OnlineTripletLossV4, "_tripletLossXRay")]:
        main(tripletLossLayer, testname)
        if shared_params["dataset_type"] == "mnist":
        	train_loader, validate_loader = load_mnist()
        	test_loader = validate_loader
        else:
        	train_loader, test_loader, validate_loader = load_data()
        do_recognition(train_loader, testName=testname, types="_train")
        do_recognition(validate_loader, testName=testname, types="_validate")
        do_recognition(test_loader, testName=testname,  types= "_test")
