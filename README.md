# HLCV-Project
HLCV SS 2021 Project


## 1. Training model on source and Noisy label generation

### Setup

1. Download datasets from follwing links:
    1. SVHN http://ufldl.stanford.edu/housenumbers/
    2. MNIST https://drive.google.com/file/d/1cZ4vSIS-IKoyKWPfcgxFMugw0LtMiqPf/view?usp=sharing
    3. USPS (in data folder) https://github.com/mil-tokyo/MCD_DA/tree/master/classification/data

2. Create a folder name 'data' and put all downloaded datasets in it


Pairs of dataset to be used:
* MNIST -> USPS
* USPS -> MNIST
* SVHN -> MNIST​

** use `--help` token with python file to know more about the parameters.

Training:
```
python train_model.py --source usps --target mnist
```


Command for inference (noisy label generation):
```
python eval_model.py --source usps --target mnist --load_epoch 190 --save_infer
```

## 2. SimCLR on Target data

Training:
```
conda activate simclr
python run.py -data ./datasets -dataset_name svhn --log-every-n-steps 2 --epochs 10
```

Change ```dataset_name``` flag as ```mnist```, ```svhn```, or ```usps``` for different datasets. 


## 3. C2D
