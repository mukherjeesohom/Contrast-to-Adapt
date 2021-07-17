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
python train_model.py --source usps --target mnist --all_use yes 
```


Command for inference (noisy label generation):
```
python eval_model.py --source usps --target mnist --load_epoch 190 --save_json --all_use yes
```

## 2. SimCLR on Target data

Training:
```
conda activate simclr
python run.py -data ./datasets -dataset_name svhn --log-every-n-steps 2 --epochs 10
```

Change ```dataset_name``` flag as ```mnist```, ```svhn```, or ```usps``` for different datasets. 


## 3. C2D

<!-- 1. Save dataset under ```data/cifar-10``` -->
1. Place noisy label `.json` file in `noisy_labels` directory.
2. Save SimCLR model under ```pretrained``` folder
3. Training:
```
python3 main_cifar.py --num_epochs 1 --batch_size 4  --r 0.8 --lambda_u 500 --dataset mnist --p_threshold 0.03 --data_path ./noisy_labels --experiment-name simclr_resnet18 --method selfsup --net resnet50
```
