import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opensource.siamesetriplet.utils import pdist
from torch.nn.modules.distance import PairwiseDistance
from constants import *


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        input = nn.Conv2d(1, 32, 5)
        connector = nn.Linear(64 * 4 * 4, 256)
        if dataset_type == 'xray':
            input = nn.Conv2d(3, 32, 5)
            connector = nn.Linear(64 * 107 * 107, 256)
        self.convnet = nn.Sequential(input,
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))


        self.fc = nn.Sequential(connector,
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        #print(output.shape)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
        

class IdentityNN(nn.Module):
    def __init__(self):
        super(IdentityNN, self).__init__()

    def forward(self, x):
        return x

class InputNet(nn.Module):
    def __init__(self):
        super(InputNet, self).__init__()
        input = nn.Conv2d(1, 32, 3)
        if shared_params["dataset_type"] == 'xray':
            input = nn.Conv2d(3, 32, 5)
        self.convnet = nn.Sequential(input,
                                     nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=1),
                                     nn.ReLU(inplace=True))
        self.nextmodel = IdentityNN()

    def forward(self, x):
        output = self.convnet(x)
        output = self.nextmodel(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class LilNet(nn.Module):
    def __init__(self):
        super(LilNet, self).__init__()

        in_channel = 1024
        if shared_params["dataset_type"] == 'xray':
            in_channel = 746496

        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(in_channel, 1024),
                                nn.PReLU(),
                                nn.Linear(1024, 512),
                                nn.PReLU(),
                                nn.Linear(512, 256),
                                nn.PReLU(),
                                nn.Linear(256, 3),
                                )

    def forward(self, x):
        #print(x.size())
        #output = self.flat(x)
        #print(output.size())
        output = self.fc(x)
        output = F.normalize(output, p=2, dim=1)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class HowGoodIsTheModel(nn.Module):
    def __init__(self, pairSelector, margin, anchors, cuda, svm):
        super(HowGoodIsTheModel, self).__init__()

        self.pairSelector = pairSelector
        self.margin = margin
        self.name = "recog"
        self.anchors = torch.stack(anchors).cuda() if cuda else torch.stack(anchors).cpu()
        self.svm = svm
        self.cuda = cuda

    def forward(self, embeddings, labels):
        pp, nn = self.pairSelector.get_pairs(embeddings=embeddings, labels=labels)
        px = embeddings[pp[:,0]]
        py = embeddings[pp[:,1]]
        labs = labels[pp[:,0]]
        func = lambda x: self.anchors[x]
        anch = torch.stack([func(x) for x in labs])  
        

        #p_euclidean_dist = CosineSimilarity() # (px - py).pow(2).sum(1) #.sqrt()
        # p_euclidean_dist2 = pdist([px, py])
        # print(p_euclidean_dist2)
        ap = torch.diagonal(torch.cdist(anch, px))
        an = torch.diagonal(torch.cdist(anch, py))
        positives = ap + self.margin >= an
        nx = embeddings[nn[:, 0]] #anchors are labels with 0
        ny = embeddings[nn[:, 1]]
        
        labs = labels[nn[:,0]] 
        anch = torch.stack([func(x) for x in labs]) 
        #n_euclidean_dist = CosineSimilarity() # (nx-ny).pow(2).sum(1) #.sqrt()
        # n_euclidean_dist2 = pdist(torch.CosineSimilarityst[nx, ny])
        # print(n_euclidean_dist2)
        
        ap = torch.diagonal(torch.cdist(anch, nx))
        an = torch.diagonal(torch.cdist(anch, ny))
        negatives = ap + self.margin < an
        svm_score  = self.svm_predict(embeddings, labels)
        #negatives = torch.diagonal(torch.cdist(nx, ny))
        #negatives = negatives > self.margin
        # print("distance ",p_euclidean_dist.data, "negatives ", n_euclidean_dist.data, self.margin) 
        return torch.tensor(0), (positives, negatives, svm_score)
        
    def svm_predict(self, embed, lab):
        embeddings = embed
        labels = lab
        if self.cuda:
            embeddings = embed.cpu().detach().numpy()
            labels = lab.cpu().detach().numpy()
        size = len(labels)
        
        predictions = self.svm.predict(embeddings)
        #print(predictions)
        #print(labels)
        score = np.sum(predictions == labels)

        #if self.cuda:
        #    embeddings.cuda()
        #    labels.cuda()
        return 100 * (score/float(size))
