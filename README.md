# Triplet loss in practice

Implements and compares the original triplet loss function with 2 proposed improvmements on the loss function.;

### Papers
```
Learning local feature descriptors with triplets and shallow convolutional neural networks - Balntas et al.
FaceNet: A Unified Embedding for Face Recognition and Clustering (idea paper) - Schroff et al
Tracking Persons-of-Interest via Adaptive Discriminative Features - Zhang et al.
```
The triplet loss and the variants can be found as a submodule in the folder opensource/siamesetriplet or by going to https://github.com/jcchuks/siamese-triplet/blob/master/losses.py.

Balntas et al is implemented in OnlineTripletLossV3
Zhang et al is implemented in OnlineTripletLossV4
Schroff et al is implemented in OnlineTripletLossV5.

All three implementations and their results on the MNIST dataset and the https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia dataset are given in their respective folders.
<p float="left">
<img src="https://github.com/jcchuks/triplet_loss_in_practice/blob/master/florian_result_mnist/clusters/trainVideo/loop.gif" width=200 height=200/>

<img src="https://github.com/jcchuks/triplet_loss_in_practice/blob/master/balntas_result_xray_resnet/clusters/trainVideo/loop.gif" width=200 height=200/>

<img src="https://github.com/jcchuks/triplet_loss_in_practice/blob/master/zhang_result_xray/clusters/trainVideo/loop.gif" width=200 height=200/>
</p>
