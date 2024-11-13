# PF-SODNet

## Install
Preparation of the training environment
```
pip install -r requirements.txt  
```
```
install pytorch_wavelets git clone https://github.com/fbcotter/pytorch_wavelets cd pytorch_wavelets pip install . 
Please refer for more details: [https://github.com/fbcotter/pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets)
```

## Datasets
```
VisDrone:https://github.com/VisDrone/VisDrone-Dataset
DIOR:http://www.escience.cn/people/gongcheng/DIOR.html  or https://pan.baidu.com/s/1Sxo5rWq7F3sq49mjDqZhtg extraction code：RSAI
```
## Testing

```
python test.py --data data/VisDrone.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights '' --name PF_640_val
```

## Training
Single GPU training
```
python train.py --workers 8 --device 0 --batch-size 32 --data data/VisDrone.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name PF_SODNet --hyp data/hyp.scratch.p5.yaml
```
Multiple GPU training
```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/VisDrone.yaml --img 640 640 --cfg cfg/training/VisDrone.yaml --weights '' --name PF_SODNet --hyp data/hyp.scratch.p5.yaml
```




