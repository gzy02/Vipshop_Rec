from typing import Dict, Set, List, Tuple
import config


def get_user_item_info(user_item_info_path: str, users_num: int) -> List[List[Tuple[int, int, int, int, int]]]:
    user_item_info = [[] for _ in range(users_num)]  # 邻接表

    with open(user_item_info_path, "r") as fp:
        fp.readline()
        for line in fp.readlines():
            line_list = line.rstrip().split(',')
            user_id = int(line_list[0])
            user_item_info[user_id].append(tuple(map(int, line_list[1:])))
    return user_item_info


def get_items_info(items_info_path: str, items_num: int) -> List[Tuple[int, int]]:
    items_info = [() for _ in range(items_num)]
    with open(items_info_path) as fp:
        fp.readline()
        for line in fp.readlines():
            line_list = line.rstrip().split(',')
            items_info[int(line_list[0])] = (
                int(line_list[1]), int(line_list[2]))
    return items_info


class DataSet():
    def __init__(self, user_item_info_path: str, items_info_path: str):
        self.user_item_info = get_user_item_info(
            user_item_info_path, config.users_num)  # 邻接表
        self.items_info = get_items_info(items_info_path, config.items_num)

        self.items_num: int = config.items_num
        self.users_num: int = config.users_num
        self.items_cat_num: int = config.items_cat_num
        self.items_brand_num: int = config.items_brand_num

        self.items_degree: List[int] = [0] * self.items_num  # 物品热门度
        self.users_degree: List[int] = [0] * self.users_num  # 用户交互数
        self.ordered_set:List[Set[int]]=[set() for _ in range(self.users_num)]
        self.interact_set:List[Set[int]]=[set() for _ in range(self.users_num)]
        for user_id, user_history in enumerate(self.user_item_info):
            for seq in user_history:
                item_id = seq[0]
                self.items_degree[item_id]+=1
                clk=seq[1]
                like=seq[2]
                addcart=seq[3]
                order=seq[4]
                self.interact_set[user_id].add(item_id)
                if order:
                    self.ordered_set[user_id].add(item_id)
            self.users_degree[user_id]+=len(self.interact_set[user_id])