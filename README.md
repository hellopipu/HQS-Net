## HQS-Net
pytorch implementation of the paper **Learned Half-Quadratic Splitting Network for Magnetic Resonance Image Reconstruction** (https://arxiv.org/abs/2112.09760)

### Install
python>=3.7.11 is required with all requirements.txt installed including pytorch>=1.10.0
```shell
git clone https://github.com/hellopipu/HQS-Net.git
cd HQS-Net
pip install -r requirements.txt
```

### Prepare dataset
you can find more information about OCMR dataset at https://ocmr.info/
```shell
## download dataset
wget https://ocmr.s3.amazonaws.com/data/ocmr_cine.tar.gz -P data/
## download dataset attributes csv file
wget https://raw.githubusercontent.com/MRIOSU/OCMR/master/ocmr_data_attributes.csv -P data/
## untar dataset 
tar -xzvf data/ocmr_cine.tar.gz -C data/
## preprocess and split dataset, it takes several hours
python preprocess_ocmr.py
```

### Training
Training and testing Scripts for all experiments in the paper can be found in folder `run_sh`. For example, if you want to train HQS-Net on accleration factor of 5x, you can run:
```shell
sh run_sh/acc_5/train/train_hqs_5.sh
```
or if you want to train Unet based HQS-Net on accleration factors 10x, you can run:
```shell
sh run_sh/acc_10/train/train_hqs_unet_10.sh
```
### Testing
For example, if you want to test HQS-Net on accleration factor of 5x, you can run:
```shell
sh run_sh/acc_5/test/test_hqs_5.sh
```

### Tensorboard
tensorboard for checking the curves while training
```shell
tensorboard --logdir log
```