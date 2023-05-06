# 基于局部AUROC优化的端到端长尾识别框架

## 1. 算法描述
针对一些跨模态场景下（如涉黄涉爆识别、违禁物品识别等）如何更好的对难样本进行挖掘和优化，提出一种基于局部AUROC优化的端到端长尾识别框架，使得模型同时关注ROC曲线下的高真阳性率（TPR）和低假阳性率（FPR）部分的面积，从而保证模型具有更好的泛化能力，显著提升长尾识别任务的性能。

## 2. 环境依赖及安装
该框架所需的环境依赖如下：
- easydict==1.9
- lmdb==0.9.24
- numpy==1.19.2
- pytorch==1.8.1
- scikit-image==0.18.1
- scikit-learn==0.24.1
- torchaudio==0.8.1
- torchvision==0.2.2
- tqdm==4.59.0

建议使用anaconda或pip配置环境。例如：
```
pip install easydict==1.9
pip install lmdb==0.9.24
pip install numpy==1.19.2
pip install pytorch==1.8.1
pip install scikit-image==0.18.1
pip install scikit-learn==0.24.1
pip install torchaudio==0.8.1
pip install torchvision==0.2.2
pip install tqdm==4.59.0
```

## 3. 运行示例

### 模型训练
模型训练需预先在params文件夹下对应数据集的json文件中配置相应的"class2id", 正样本（少数类）为1，其余默认为0，并运行如下命令：
```
python3 train.py dataset-name loss-type
```
“dataset-name”为数据集，“loss-type”为模型优化的损失函数。

本框架主要提供以下三个数据集，所有数据均已公开，具体如下：
- CIFAR-10-LT: 可通过[此链接下载](https://github.com/statusrank/XCurve/tree/master/example/data)
- CIFAR-100-LT: 可通过[此链接下载](https://github.com/statusrank/XCurve/tree/master/example/data)
- Tiny-ImageNet-200-LT: 可通过[此链接下载](https://drive.google.com/file/d/1WYoQrDIDK-E2aK8Rj_Vph_MBXIDjusHs/view)

可选两种训练损失函数，传统AUROC优化“SquareAUCLoss”和局部AUROC长尾识别损失“TPAUCLoss”，例如：
```
python3 train.py cifar-10-long-tail TPAUCLoss
```
此外，该算法已整合至XCurve通用框架中且兼容pytorch训练模式。
1. 执行如下命令安装XCurve
```
pip install XCurve
```
2. 通过如下方式定制训练：
```python3
# 基于局部AUROC优化的端到端长尾识别损失
from XCurve.AUROC.losses import TPAUCLoss

model = {"your pytorch model"}
optimizer = {"your optimizer"}

# 训练损失
criterion = TPAUCLoss(gamma=1.0, epoch_to_paced=10, re_scheme='Exp') # 详细参数参见losses.py

# create Dataset (train_set, val_set, test_set) and dataloader (trainloader)
# You can construct your own dataset/dataloader 
# but must ensure that there at least one sample for every class in each mini-batch 

train_set, val_set, test_set = your_datasets(...)
trainloader, valloader, testloader = your_dataloaders(...)


# forward of model
for x, target in trainloader:
    x, target  = x.cuda(), target.cuda()
    # target.shape => [batch_size, ]
    # Note that we ask for the prediction of the model among [0,1] 
    # for any binary (i.e., sigmoid) or multi-class (i.e., softmax) AUROC optimization.
    pred = model(x) # [batch_size, num_classess] when num_classes > 2, o.w. output [batch_size, ] 

    loss = criterion(pred, target)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 4. 论文/专利成果
> Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang. [When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC.](https://github.com/statusrank/A-Generic-Framework-for-Optimizing-Two-way-Partial-AUC/blob/main/TPAUC.pdf) ICML 2021 (Long talk, 3%)