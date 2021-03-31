# Triplet loss in practice

Implements and compares the original triplet loss function with 2 proposed improvmements on the loss function.;

### Papers
```
Learning local feature descriptors with triplets and shallow convolutional neural networks - Balntas et al.
FaceNet: A Unified Embedding for Face Recognition and Clustering (idea paper) - Schroff et al
Tracking Persons-of-Interest via Adaptive Discriminative Features - Zhang et al.
```
The triplet loss and the variants can be found as a submodule in the folder `opensource/siamesetriplet` or by direct link to https://github.com/jcchuks/siamese-triplet/blob/master/losses.py.

Balntas et al is implemented in `OnlineTripletLossV3`

Zhang et al is implemented in `OnlineTripletLossV4`

Schroff et al is implemented in `OnlineTripletLossV5`.

The distance metric used is Euclidean distance.

Results on the three variants were obtained on the MNIST dataset and the https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia dataset. All raw results can be found in folders with `result` as part of filename.

<p float="left">
<img src="https://github.com/jcchuks/triplet_loss_in_practice/blob/master/florian_result_mnist/clusters/trainVideo/loop.gif" width=200 height=200/>

<img src="https://github.com/jcchuks/triplet_loss_in_practice/blob/master/balntas_result_xray_resnet/clusters/trainVideo/loop.gif" width=200 height=200/>

<img src="https://github.com/jcchuks/triplet_loss_in_practice/blob/master/zhang_result_xray/clusters/trainVideo/loop.gif" width=200 height=200/>
</p>

# Acknowledgements:

Thanks to Adam, this work was forked and extended from <a href="https://github.com/adambielski/siamese-triplet">Adam Bielski's</a> siamese-triplet repo.

Experiments were carried out on <a href="https://carleton.ca/scs/tech-support/gpu/scs-gpu-cuda10-ubuntu18-04/"> Carleton University's </a> GPU

This experiment was done as part of the <a href="http://people.scs.carleton.ca/~majidkomeili/Teaching/COMP5900-W21/home.html">Advanced Machine Learning Class </a> at Carleton.
