# HLCV-Project
HLCV SS 2021 Project

Pairs of dataset to be used:
* MNIST -> USPS, 
* USPS -> MNIST,
* SVHN -> MNIST​

** use `--help` token with python file to know more about the parameters.

Training:
```
python train_model.py --source usps --target mnist --max_epoch 200
```


Command for inference:
```
python eval_model.py --source usps --target mnist --load_epoch 190 --save_infer
```
