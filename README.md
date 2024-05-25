# 图像分类问题

### 必要的库：
```
    pytorch, torchvision, numpy, pandas, matplotlib, PIL, sklearn， tqdm
```

### 使用方法：
##### 训练
```
    python train.py --mode train \
                --data_dir /home/xinhuang/Desktop/arapat/Scene_Classify/dataset \
                --epochs 10 \
                --batch_size 8 \
```

#### 测试：
```
python train.py --mode test \
                --data_dir /home/xinhuang/Desktop/arapat/Scene_Classify/dataset \
                --batch_size 8 \
                --pretrained_model /home/xinhuang/Desktop/arapat/Scene_Classify/output/model_epochs_20.pth 
```

### 文件夹结构如下：
```
 - Scene_Classify
    - README.md
    - train.py # 训练过程
    - dataset.py # 数据集处理
    - VggNet.py # VGG网络
    - data_augmentation.py # 数据增强
    - grad_cam.py # Grad-CAM
    
    - train_run.sh # 训练脚本
    - test_run.sh # 测试脚本

    - utils
        - expand.py # 图片扩大
        - modify_csv.py # 修改csv文件
    - dataset # 数据集
        - imgs
            - train
            - val
            - test
        - train_data.csv
        - val_data.csv
        - test_data.csv
    - output
        - cam_map # Grad-CAM结果
        *.pth # 模型文件
```
