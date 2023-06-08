import torch
from torch import Tensor
from typing import Dict, Set, Tuple
from DataSet import DataSet
from typing import Optional, Union
import config


def Test_batch(model, origin_train_data: DataSet, test_user_item_map: Dict[int, Set[int]], device: str) -> Tuple[float, float, float, float]:
    """测试当前模型"""
    user_num = origin_train_data.user_num
    model.eval()
    with torch.no_grad():
        precision = 0
        recall = 0
        for start_user_id in range(0, user_num, config.batch_size):
            end_user_id = min(origin_train_data.user_num,
                              start_user_id+config.batch_size)
            batch_users_gpu = torch.arange(
                start_user_id, end_user_id, dtype=torch.int32).to(device)

            # 测试开始
            rating = model.getUsersRating(batch_users_gpu)  # 模型预测的用户-物品交互矩阵
            exclude_index = []
            exclude_items = []
            for user_id in range(start_user_id, end_user_id):
                items = origin_train_data.user_item_map[user_id]
                exclude_index.extend([user_id-start_user_id] * len(items))
                exclude_items.extend(items)

            # 去除用户交互过的物品
            rating[exclude_index, exclude_items] = 0.0  # 因为sigmoid输出是(0,1)
            _, topk_indices = torch.topk(
                rating, k=config.topk)  # 降序排列的top-k个待推荐物品矩阵
            for user_id in range(start_user_id, end_user_id):
                items = test_user_item_map[user_id]
                rec_list = topk_indices[user_id -
                                        start_user_id].tolist()  # 模型的推荐列表
                intersect = items.intersection(set(rec_list))  # 模型推荐与真实交互的交集
                if len(intersect) == 0:  # 此时precision, recall, nDCG均为0
                    continue
                precision += len(intersect)/config.topk
                recall += len(intersect)/len(items)

        precision /= len(test_user_item_map)
        recall /= len(test_user_item_map)
        F1 = 0.0 if precision == 0 or recall == 0 else 2 * \
            precision*recall/(precision+recall)
        return precision, recall, F1
