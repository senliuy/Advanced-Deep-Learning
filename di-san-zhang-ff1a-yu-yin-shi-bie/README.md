# MNIST-SiameseNet

Demo of SiameseNet of MNIST dataset. Given a pair of inputs, the model can tell you which is larger or the same.

RankNet is widely used in Search Engine, Recommendation and other related directions. This demo gives you one SiameseNet on MNIST based on keras. The core motivation of this Demo is to show how Siamese Net was implemented. Furthermore, you and transfer it to any other field that you are research.

## Code and Procedure

#### 1. The models are needed

```py
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint
import numpy.random as rng
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot = False)
import matplotlib.pyplot as plt
```

#### 2. Prepare the MNIST data

To achieve a faster training speed. Only numbers of 0, 1, 2 are used to train the net. You can relief numbers with commenting the codes.

```py
# Using 0-2 temporarily， 0-9 can also be used, but more training time is needed
#//TODO: Using tuple instead of matrix to initialize the dataset
def prepare_mnist(images, labels):
    min_num = 60000
    for i in range(0, 10):
        min_num = min(min_num, len(np.where(labels == i)[0]))
    dataset = np.zeros((10, min_num, 28, 28))
    idx_arr = np.zeros((10,1),dtype=int)
    for idx in range(images.shape[0]):
        if(idx_arr[int(labels[idx])] < min_num):
            dataset[int(labels[idx]), idx_arr[labels[idx]], :, :] = np.reshape(images[idx],(28,28))
            idx_arr[labels[idx]] += 1
    dataset = dataset[0:3,:,:,:] # comment this line to relief all numbers
    return dataset
```

#### 3. Define the structure of the Siamese Net

A simple CNN with convolution and pooling are used to construct the network structure. Other networks such as ResNet or VGG can also be used.

```py
def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input_shape = (28, 28, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(1e-4)))
convnet.add(Conv2D(128,(3,3),activation='relu',
                   kernel_regularizer=l2(1e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(32,(3,3),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(1e-4),bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(Dense(128,activation="sigmoid",kernel_regularizer=l2(1e-4),kernel_initializer=W_init,bias_initializer=b_init))
#encode each of the two inputs into a vector with the convnet
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
```

#### 4. Define the hyperparameter of SiameseNet

Adam is used to train the model.

```py
#merge two encoded inputs with the l1 distance between them
L1_distance = lambda x: (x[0]-x[1])
both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)


siamese_net = Model(input=[left_input,right_input],output=prediction)
# optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)
optimizer = Adam(0.0001)
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

siamese_net.count_params()
```

#### 5. Define a loader for training and testing

There are three functions in Siamese\_Loader.

* Function `get_batch`\(self,n\)  is parparing pairs during traing 

```py
class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,Xtrain,Xval):
        self.Xval = Xval
        self.Xtrain = Xtrain
        self.n_classes,self.n_examples,self.w,self.h = Xtrain.shape
        self.n_val,self.n_ex_val,_,_ = Xval.shape

    def get_batch(self,n):
        """Create batch of n pairs, half same class, half different class"""
        categories = rng.choice(self.n_classes,size=(n,),replace=False)
        pairs=[np.zeros((n, self.h, self.w,1)) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//3:] = 1
        for i in range(n):
            category = categories[i]
            idx_1 = rng.randint(0,self.n_examples)
            pairs[0][i,:,:,:] = self.Xtrain[category,idx_1].reshape(self.w,self.h,1)
            idx_2 = rng.randint(0,self.n_examples)
            #pick images of same class for 1st half, different for 2nd
            category_2 = category if i > n//3 else (category + rng.randint(1,self.n_classes-1)) % self.n_classes
            if category_2 < category:
                targets[i] = 0
            elif category_2 > category:
                targets[i] = 1
            else:
                targets[i] = 0.5
            pairs[1][i,:,:,:] = self.Xtrain[category_2,idx_2].reshape(self.w,self.h,1)
        return pairs, targets
```

* function `make_oneshot_task(self,N)` is used to prepare pairs during testing and evaluation

```py
def make_oneshot_task(self,N):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    categories = rng.choice(self.n_val, size=(N,),replace=False)
    indices = rng.randint(0,self.n_ex_val,size=(N,))
    true_category = categories[0]
    ex1, ex2 = rng.choice(self.n_ex_val,replace=False,size=(2,))
    test_image = np.asarray([self.Xval[true_category,ex1,:,:]]*N).reshape(N,self.w,self.h,1)
    support_set = self.Xval[categories,indices,:,:]
    support_set[0,:,:] = self.Xval[true_category,ex2]
    support_set = support_set.reshape(N,self.w,self.h,1)
    pairs = [test_image,support_set]
    print true_category
    print categories
    targets = np.zeros((N,))
    targets[0] = 1
    return pairs, targets, true_category, categories
```

* function`test_oneshot(self,model,N,k,verbose=0)` is calculating the correct rate of this batch. When predict probs is smaller than TEST\_THRES, it means right input is smaller than left input, and larger means right input bigger than left input and otherwise means two inputs are exactly the same number.

```py
def test_oneshot(self,model,N,k,verbose=0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    TEST_THRES = 0.3
    if verbose:
        print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k,N))
    for i in range(k):
        inputs, targets, true_category, categories = self.make_oneshot_task(N)
        probs = model.predict(inputs)
        print probs
        correct = True
        for idx in range(0, len(probs)):
            if TEST_THRES < probs[idx] and probs[idx] < 1-TEST_THRES and true_category == categories[idx]:
                correct = correct and True
            elif TEST_THRES <= probs[idx] and categories[idx] > true_category:
                correct = correct and True
            elif TEST_THRES >= probs[idx] and categories[idx] < true_category:
                correct = correct and True
            else:
                correct = False
                break
        if correct == True:
            n_correct+=1
    percent_correct = (100.0*n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
    return percent_correct, inputs
```

#### 6. Training

Traing step, pay attention that N\_way and batch\_size no larger than number of classes

```py
# Training and Refine the model
evaluate_every = 100
loss_every=300
batch_size = 3
N_way = 3
n_val = 4
K.get_session().run(tf.global_variables_initializer())
# siamese_net.load_weights("./model/model.hdf5") # Refine the model with an trained model
best = 0.0

train_set = preprocee_mnist(mnist.train.images, mnist.train.labels)
val_set = preprocee_mnist(mnist.validation.images, mnist.validation.labels)
loader = Siamese_Loader(train_set, val_set)

for i in range(1100000):
    (inputs,targets)=loader.get_batch(batch_size)
    loss=siamese_net.train_on_batch(inputs,targets)
    if i % evaluate_every == 0:
        val_acc, _ = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
        if val_acc >= best:
            print("saving")
            siamese_net.save('./model/model.hdf5')
            best=val_acc

    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i,loss))
```

#### 7. Testing

```py
# Testing Phrase
TEST_THRES = 0.3
train_set = prepare_mnist(mnist.train.images, mnist.train.labels)
val_set = prepare_mnist(mnist.validation.images, mnist.validation.labels)
siamese_net.load_weights('./model/model.hdf5')
loader = Siamese_Loader(train_set, train_set)
correct = 0
num_pairs = 1
for i in range(num_pairs):
    res, inputs = loader.test_oneshot(model = siamese_net, N = 3, k = 1,verbose=0)
    print inputs[0].shape
    if res == 100:
        correct += 1
    plt.subplot(3,2,1)
    plt.imshow(inputs[0][0][:,:,0])
    plt.subplot(3,2,2)
    plt.imshow(inputs[1][0][:,:,0])
    plt.subplot(3,2,3)
    plt.imshow(inputs[0][1][:,:,0])
    plt.subplot(3,2,4)
    plt.imshow(inputs[1][1][:,:,0])
    plt.subplot(3,2,5)
    plt.imshow(inputs[0][2][:,:,0])
    plt.subplot(3,2,6)
    plt.imshow(inputs[1][2][:,:,0])
    plt.show()
print("Got an accuracy of {}%".format(100.0 * correct/num_pairs))
```

## Result

```
[[ 0.61299264]
 [ 0.12821619]
 [ 0.93877953]]
```

![](data:image/png;base64,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%0AgtGAIyLB/H/+q+dpb6t5XAAAAABJRU5ErkJggg==)

```
Got an accuracy of 100.0%
```



