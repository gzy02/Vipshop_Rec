import config
from DataSet import DataSet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SampleData(Dataset):
    def __init__(self, origin_data: DataSet):
        super().__init__()
        self.user_item_info = origin_data.user_item_info
        self.items_info = origin_data.items_info
        self.Xs = []
        self.__init_dataset()

    def __init_dataset(self):
        for user_id, user_history in enumerate(self.user_item_info):
            for seq in user_history:
                item_id = seq[0]
                rating = config.clk_score * \
                    seq[1]+config.like_score*seq[2]+config.addcart_score * \
                    seq[3]+config.order_score*seq[4]
                self.Xs.append(
                    (user_id, item_id, self.items_info[item_id][0], self.items_info[item_id][1], rating))

    def __getitem__(self, index):
        # user_id, item_id, item_cat, item_brand, rating
        return self.Xs[index]

    def __len__(self):
        return len(self.Xs)


def get_trainloader(origin_train_data: DataSet) -> DataLoader:
    sample_train_data = SampleData(origin_train_data)
    trainloader = DataLoader(sample_train_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=False,
                             num_workers=0)
    return trainloader
