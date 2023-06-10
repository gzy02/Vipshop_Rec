import config
from DataSet import DataSet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

class SampleData(Dataset):
    def __init__(self, origin_data: DataSet):
        super().__init__()
        self.data=origin_data
        self.Xs = []
        self.__init_dataset()

    def __init_dataset(self):
        for user_id in range(self.data.users_num):
            for item_id in self.data.ordered_set[user_id]:
                neg_item_id=random.randint(0,self.data.items_num-1)
                while neg_item_id in self.data.interact_set[user_id]:
                    neg_item_id=random.randint(0,self.data.items_num-1)
                self.Xs.append(
                    (user_id, item_id, self.data.items_info[item_id][0], self.data.items_info[item_id][1], neg_item_id, self.data.items_info[neg_item_id][0], self.data.items_info[neg_item_id][1]))

    def __getitem__(self, index):
        return self.Xs[index]

    def __len__(self):
        return len(self.Xs)


def get_trainloader(origin_train_data: DataSet,batch_size:int) -> DataLoader:
    sample_train_data = SampleData(origin_train_data)
    trainloader = DataLoader(sample_train_data,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=False,
                             num_workers=0)
    return trainloader
