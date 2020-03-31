# LFSSR-SAS-PyTorch
PyTorch implementation of **TIP 2018** paper: "**Light Field Spatial Super-resolution Using Deep Efficient Spatial-Angular Separable Convolution**". You can find the original MATLAB code from [here](https://github.com/spatialsr/DeepLightFieldSSR).

## Usage
### Dependencies
- Python 3.6
- PyTorch 1.0
### Dataset
We provide MATLAB code for preparing the training and testing data. Please first download light field datasets, and put them into corresponding folders in `LFData`.

### Demo
```
usage: demo.py [-h] [--model_path MODEL_PATH] [--scale SCALE]
               [--test_dataset TEST_DATASET] [--angular_num ANGULAR_NUM]
               [--layer_num LAYER_NUM] [--save_img SAVE_IMG] 
             
optional arguments:  
  -h, --help          Show this help message and exit  
  --model_path        Model path. Default=pretrained_models/model_2x.pth  
  --scale SCALE       SR factor  
  --test_dataset      Dataset for test  
  --angular_num A     Size of one angular dim. Default=7.  
  --layer_num         Number of SAS layers. Default=6.  
  --save_img          Save image or not  
```
An example of usage is shown as follows:  

```
python demo.py --model_path pretrained_models/model_2x.pth --test_dataset HCI --scale 2  --save_img 1
```  
    
**Note:**  We provide 2 pre-trained models for 2x and 4x SR, respectively. There are some differences from the original MATLAB pre-trained models:  
- new models were trained on a hybrid dataset containing both synthetic and real-world light field images, while the original ones were trained only on real-world images captured by a Lytro Illum camera;
- and new models were trained for light fields with the angular resolution of 7x7, while the original ones were trained for 8x8 light fields. 

### Training 
An example of training your own model is shown as follows:
```
python train.py --dataset all --scale 2 --layer_num 6 --angular_num 7 --lr 1e-4
```
### Testing
An example of testing one epoch of your trained model is shown as follows:
```
python test.py --train_dataset all --test_dataset HCI --scale 2 --layer_num 6 --angular_num 7 --lr 1e-4 --epoch 500 --save_img 1
```
