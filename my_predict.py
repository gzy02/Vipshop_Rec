import torch
from DataSet import DataSet
from MFModel import MFModel
import config
from util import get_test_set
topk = 20
# batch_size = 1

epoch = 200
weight_decay =0
lr = 0.001

load_path = f"./model/{epoch}_{config.user_dim}_{lr}_{weight_decay}.pth"
submission_path = f'./submit/{epoch}_{config.user_dim}_{config.cat_dim}_{config.brand_dim}_{lr}_{weight_decay}.txt'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
origin_train_data = DataSet(config.user_item_info_path, config.items_info_path)

model = MFModel(load_model=True)
model.load_state_dict(torch.load(load_path))
model = model.to(device)

user_for_test = list(get_test_set(config.test_users_path))
user_num = len(user_for_test)
item_for_test = list(get_test_set(config.test_items_path))
item_cat_for_test = list(
    map(lambda x: origin_train_data.items_info[x][0], item_for_test))
item_brand_for_test = list(
    map(lambda x: origin_train_data.items_info[x][1], item_for_test))

item_for_test_gpu = torch.LongTensor(item_for_test).to(device)
item_cat_for_test_gpu = torch.LongTensor(item_cat_for_test).to(device)
item_brand_for_test_gpu = torch.LongTensor(item_brand_for_test).to(device)

model.eval()
with torch.no_grad():
    with open(submission_path, "w", encoding="utf8") as fp:
        for user_id in user_for_test:
            rating = model.getUsersRating(
                torch.LongTensor([user_id]).to(device), item_for_test_gpu, item_cat_for_test_gpu, item_brand_for_test_gpu)
            _, topk_indices = torch.topk(
                rating, k=topk)  # 降序排列的top-k个待推荐物品矩阵
            rec_list = topk_indices.tolist()  # 模型的推荐列表
            for item_index in rec_list:
                item_id = item_for_test[item_index]
                fp.write(f"{user_id},{item_id}\n")
