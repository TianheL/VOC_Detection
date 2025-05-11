# VOC_Detection

## 1. 环境配置
### mmdetection安装
请参考[mmdetection repo](https://github.com/open-mmlab/mmdetection)

### 其他依赖
```bash
pip install -r requirements.txt
```

### 数据准备
下载VOC数据集，解压后放在data/VOCdevkit文件夹下，VOCdevkit文件夹下应该有VOC2007和VOC2012两个文件夹。
```bash
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

## 2. 模型训练
训练Sparse R-CNN模型，命令如下：
```bash
python ./tools/train.py ./configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py
```

查看训练日志，命令如下：
```bash
tensorboard --logdir=./work_dirs/sparse-rcnn_r50_fpn_1x_coco/20250509_141113/vis_data
```

## 3. 模型测试
测试Sparse R-CNN模型，命令如下：
```bash
python tools/test.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py  work_dirs/sparse-rcnn_r50_fpn_1x_coco/epoch_9.pth --out "./infer/out_9.pkl" --work-dir="./infer"
```

单张图片检测，命令如下：
```bash
python ./demo/image_demo.py ./demo/demo.jpg configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py --weights work_dirs/sparse-rcnn_r50_fpn_1x_coco/epoch_9.pth --out-dir infer
```
