# Paper

Code for the paper "**Rethinking Ambiguity Attack and Fingerprinting Defense for Model Intellectual Property Protection: A Frequency Perspective**".

## File and Folder Descriptions

### Root Directory

- **`README.md`**: The main documentation file providing an overview of the project, setup instructions, and usage guidelines.
- **`requirements.txt`**: Lists all the Python dependencies required for the project, which can be installed via `pip install -r requirements.txt`.

- **`create_original_model.py`**: Used to train the original independent model.
- **`1D.py`**: Modify the frequency band coefficients of the model's convolutional kernel using 1D-DCT.
- **`2D.py`**: Modify the frequency band coefficients of the model's convolutional kernel using 2D-DCT.
- **`3D.py`**: Modify the frequency band coefficients of the model's convolutional kernel using 3D-DCT.
- **`3D-scale.py`**: Modify the different high-frequency  modification scales of the model's convolutional kernel using 3D-DCT.

### `utils/` Folder

- **`data.py`**: Contains functions to load and preprocess the datasets.
- **`model.py`**: Contains the architectures of various neural network models.
- **`gren_dct.py`**: 3D-DCT transform for each convolutional kernel in the convolutional layer.
- **`compre_cos.py`**: Extraction of frequency domain coefficients as fingerprints and comparison of cosine similarity.

### `E1/` Folder

- **`WP.py`**: Pruning the weights of the model.
- **`FP.py`**: Pruning the model's convolution kernel.
- **`fine_tuning_retrain.py`**: Fine-tuning using the same dataset.
- **`fine_tuning_trans.py`**: Fine-tuning using the different dataset.
- **`freq_ambighity.py`**: Frequency-based Ambiguity attack.
- **`attack.py`**: Tampering with fingerprint frequency domain coefficients.

### `E2` Folder

- **`compare.py`**: Fingerprint comparison of the two models.
  
### `E3` Folder

- **`compare.py`**: Fingerprint comparison of the two models.

## Getting started

Download the data and trained models for each dataset: [MNIST], [Fashion MNIST], [CIFAR-10], [CIFAR-100]

The MNIST, Fashion MNIST, CIFAR-10, and CIFAR-100 datasets can be automatically downloaded in the code.

### Install the dependencies

We tested using Python 3.8. We use PyTorch and you can install it based on your own cuda version.

```conda create -n highfreq python=3.8```
```conda activate highfreq```
```conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia```
```conda install scipy```

### Evaluation

### Step 1: Training the original models

First using ```create_original_model.py``` to train the original Independent model, the --model_style and --dataset can be passed in via command line arguments (args).
The types of```--dataset``` are cifar10,cifar100,mnist,fmnist
The types of```--model_style``` are cnn,alexnet,resnet18,vgg16

e.g. to train an CNN model with the CIFAR-10 dataset the following command line can be used:
```python create_original_model.py --dataset cifar10 --model_style cnn --epochs 50```

Please train multiple models for subsequent use and be careful with file naming when using the same dataset and model but with different initializations!

### Step 2: Evaluating

#### Table1

The path is  named according to 'Start Modification Ratio - End Modification Ratio - Modified Value', where '0-1/3' corresponds to the low-frequency, '1/3-2/3' corresponds to the mid-frequency, '2/3-1' corresponds to the high frequency.

``` python 1D.py ```
``` python 2D.py ```
``` python 3D.py ```

View model accuracy under ```test_accuracy.txt``` file

#### Table2

``` python 3D-scale.py ```

#### Table3 and Table 4

Other fingerprinting methods will not be described too much here, Simply use multiple ambiguity attacks models as suspect models and compare fingerprint similarity with original models.

#### Table5 and Table 6

Modify the command line incoming parameters according to the corresponding dataset and model.The corresponding fingerprint similarity is stored in the path ```compare/FS.txt```  and the model accuracy is stored in the path ```ACC.txt```

WP:
``` python E1/WP.py --dataset cifar10 --model_style cnn --epochs 50 ```

FP:
``` python E1/FP.py --dataset cifar10 --model_style cnn --epochs 50 ```

Fine-tuning Retrain:
```python  E1/fine_tuning_retrain.py --ora_dataset cifar10 --model_style cnn --epochs 50 --ft_dataset cifar10 --ft_epochs 20```

Fine-tuning  Tansfer:
```python  E1/fine_tuning_trans.py --ora_dataset cifar10 --model_style cnn --epochs 50 --ft_dataset cifar100 --ft_epochs 20```

Frequency-based Ambiguity:
```python  E1/frequency_based_ambiguity.py --dataset cifar10 --model_style cnn --epochs 50```

#### Table7

Modify the last parameter (rate) of the compare_cossim_dct_filters_long( ) function in the ```utils/compare_cos.py``` file, and subsequently run it as instructed in table5 and table6，Remember to change the save path.

#### Figure2

``` python  E1/adaptive_attack.py --dataset cifar10 --model_style cnn --epochs 50 ```

#### Table8

``` python E2/compare.py --dataset1 cifar10 --model_style1 alexnet --epochs1 50 --dataset2 cifar10 --model_style2 alexnet --epochs2 50 ```

#### Table9

``` python E3/time_use.py --dataset1 cifar10 --model_style1 alexnet --epochs1 50 --dataset2 cifar10 --model_style2 alexnet --epochs2 50 ```
The time to run can be stored in ``` time_results.txt ``` after executing the commands in Table 8 by running the following.
