# %% import
import os
import torch

import config
from MFModel import MFModel
from util import setup_seed, print_config
from SampleData import get_trainloader
from DataSet import DataSet
from SampleData import SampleData, get_trainloader
# %% 打印全局超参数
print("pid =", os.getpid())
print_config()

# %% 设置随机数种子
setup_seed(config.seed)
# %% 选gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %% 初始化原始训练数据
origin_train_data = DataSet(config.user_item_info_path, config.items_info_path)
# %% 模型与优化器
model = MFModel(config.load_weight)
if config.load_weight:
    model.load_state_dict(torch.load(config.load_path))
    print(f"Load model {config.load_path}")
model = model.to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
# %% 训练
for epoch in range(1+config.load_epoch, config.epochs+config.load_epoch+1):
    losses = []
    train_loader = get_trainloader(origin_train_data,config.batch_size)
    for data in train_loader:
        user, item, item_cat, item_brand, pos_item,pos_item_cat,pos_item_brand,score = data
        loss = model.bpr_loss(user.to(device), item.to(device), item_cat.to(device), item_brand.to(device),pos_item.to(device),pos_item_cat.to(device),pos_item_brand.to(device),score.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    print('Epoch {} finished, average loss {}'.format(
        epoch, sum(losses) / len(losses)))
    if epoch % 5 == 0:
        torch.save(model.state_dict(), './model/{}_{}_{}_{}.pth'.format(
            epoch, config.user_dim, config.lr, config.weight_decay))
