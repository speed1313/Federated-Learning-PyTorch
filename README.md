# Federated-Learning (PyTorch)

## 考察
no-iidの時, local_epを増やして局所最適解に落とした上でFedAvgをしたらglobalのlossは全然減らないのではないか.
なお, MNISTのnon-iidではlabelごとにソートしてからclientに分配している
#### 実験設定
1. local_ep=20, iid=0, frac=0.03
```
 Results after 10 global rounds of training:
|---- Avg Train Accuracy: 100.00%
|---- Test Accuracy: 34.47%
Total Delay: 67.0
 Total Run Time: 440.8513
```
各々のclientに最適化され, 全体の精度が低いことがわかる.
- local_ep=10, iid=0, frac=0.03
local_epを小さくして過学習しないようにしてみる
```

```

2. local_ep=20, iid=0, frac=0.1
fracを増やしてより平均化してみる.
```
 Results after 10 global rounds of training:
|---- Avg Train Accuracy: 98.33%
|---- Test Accuracy: 81.86%
Total Delay: 73.0

 Total Run Time: 1660.9639
```
過学習が防げている

3. local_ep=10, iid=0, frac=0.03で実験
epochを減らしてclientを極小値に落とさない
```
 Results after 10 global rounds of training:
|---- Avg Train Accuracy: 93.33%
|---- Test Accuracy: 65.21%
Total Delay: 70.0

 Total Run Time: 254.0224
(.venv)
```
計算コストが減っているにも関わらず, 1.より精度が良い.

4. local_ep=5, iid=0, frac=0.03で実験
3.からさらにepochを減らしてみる.
```
Results after 10 global rounds of training:
|---- Avg Train Accuracy: 96.67%
|---- Test Accuracy: 50.75%
Total Delay: 57.0

 Total Run Time: 169.5783
```
3.より精度が落ちてしまった.
バランスが重要であることがわかる.

----

Implementation of the vanilla federated learning paper : [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).


Experiments are produced on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.

Since the purpose of these experiments are to illustrate the effectiveness of the federated learning paradigm, only simple models such as MLP and CNN are used.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):
```
python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under non-IID condition:
```
python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.

## Results on MNIST
#### Baseline Experiment:
The experiment involves training a single model in the conventional way.

Parameters: <br />
* ```Optimizer:```    : SGD
* ```Learning Rate:``` 0.01

```Table 1:``` Test accuracy after training for 10 epochs:

| Model | Test Acc |
| ----- | -----    |
|  MLP  |  92.71%  |
|  CNN  |  98.42%  |

----

#### Federated Experiment:
The experiment involves training a global model in the federated setting.

Federated parameters (default values):
* ```Fraction of users (C)```: 0.1
* ```Local Batch size  (B)```: 10
* ```Local Epochs      (E)```: 10
* ```Optimizer            ```: SGD
* ```Learning Rate        ```: 0.01 <br />

```Table 2:``` Test accuracy after training for 10 global epochs with:

| Model |    IID   | Non-IID (equal)|
| ----- | -----    |----            |
|  MLP  |  88.38%  |     73.49%     |
|  CNN  |  97.28%  |     75.94%     |


## Further Readings
### Papers:
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Blog Posts:
* [CMU MLD Blog Post: Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)
* [Leaf: A Benchmark for Federated Settings (CMU)](https://leaf.cmu.edu/)
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)


