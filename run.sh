# demo: test the pretrained model
python demo.py --model_path pretrained_models/model_2x.pth --test_dataset HCI --scale 2  --save_img 1

# train your 2x model
python train.py --dataset all --scale 2 --layer_num 6 --angular_num 7 --lr 1e-4
# train your 4x model
python train.py --dataset all --scale 4 --layer_num 6 --angular_num 7 --lr 1e-4
# test one epoch of your models
python test.py --train_dataset all --test_dataset HCI --scale 2 --layer_num 6 --angular_num 7 --lr 1e-4 --epoch 500 --save_img 1



