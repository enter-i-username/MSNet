# MSNet

## Abstract

Hyperspectral anomaly detection (HAD) has attracted increasing attention due to its economical and efficient applications. The main challenge lies in the data-starved problem of hyperspectral images (HSIs) and the costliness of manual annotation, making it heavily reliant on the model's adaptability and robustness to unseen scenes under limited samples. Self-supervised learning offers a solution to this urgency via mining meaningful representations from the data itself. One promising paradigm is leveraging untrained neural networks to reconstruct the background component for revealing anomalous information. Its capability stems from the network architecture and the training process rather than learning from expensive and strongly domain-dependent data, which is naturally applicable to HAD. In this paper, to handle the urgent requirement for self-supervised learning in HAD, we propose a multi-scale network (termed MSNet) that detects anomalies with enhanced separation training. The network architecture consists of several multi-scale convolutional encoder-decoder (CED) layers, considering the spatial characteristics of the anomalies. To suppress the anomalies during background reconstruction, we adopt a new separation training strategy by introducing a soft separator for better practicality on larger datasets. Extensive experiments conducted on 5 commonly used datasets and the HAD100 dataset, demonstrate the superiority of our method over its counterparts.

## Workflow

<img src="msnet.png" alt="msnet" width="60%" height="60%">

### 1. Band Selection
We utilize a fast and efficient band selection algorithm [OPBS](https://ieeexplore.ieee.org/document/8320544) to eliminate the redundant information among bands, while reducing time costs for training.

### 2. Network Training
We train the multi-scale network using the enhanced separation training loss and the multi-scale reconstruction loss.

### 3. Anomaly Detection
The detection map is obtained by computing reconstruction errors between the input and output of the trained network.

## Getting Started

### Installing Dependencies
To get started, please install the following packages in Python 3.8 environment:
- matplotlib (version 3.5.2)
- numpy (version 1.24.3)
- scikit_learn (version 1.2.2)
- scipy (version 1.10)
- torch (version 1.13.1)
- tqdm (version 4.65.0)

by running the command:
```
pip install -r requirements.txt
```

### Starting an Experiment

We have prepared a demo program to start a simple experiment by running the command:
```
python main.py
```

In this program, we evaluate the network using the Coast dataset in `dataset/coast`. You can also include other datasets in the `dataset` directory using a format similar to "Coast".
