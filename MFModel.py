import torch
import torch.nn as nn
import config


class MFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.user_dim = config.user_dim
        self.item_dim = config.item_dim
        self.users_num = config.users_num
        self.items_num = config.items_num
        self.items_cat_num = config.items_cat_num
        self.cat_dim = config.cat_dim
        self.items_brand_num = config.items_brand_num
        self.brand_dim = config.brand_dim

        self.user_embedding = torch.nn.Embedding(self.users_num, self.user_dim)
        self.item_embedding = torch.nn.Embedding(self.items_num, self.item_dim)
        self.item_cat_embedding = torch.nn.Embedding(
            self.items_cat_num, self.cat_dim)
        self.item_brand_embedding = torch.nn.Embedding(
            self.items_brand_num, self.brand_dim)
        self.user_bias_embedding = nn.Embedding(config.users_num, 1)
        self.item_bias_embedding = nn.Embedding(config.items_num, 1)
        self.global_bias = nn.Parameter(torch.tensor(1.0))

        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_cat_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_brand_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.user_bias_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_bias_embedding.weight, gain=1)
        print('use xavier initilizer')

    def forward(self, user, item, item_cat, item_brand):
        self.train()
        user_embedding = self.user_embedding(user)
        item_embedding = torch.concat((self.item_embedding(item), self.item_cat_embedding(
            item_cat), self.item_brand_embedding(item_brand)), dim=1)
        item_bias = self.item_bias_embedding(item).squeeze()
        user_bias = self.user_bias_embedding(user).squeeze()
        dot = torch.sum(torch.mul(user_embedding, item_embedding), dim=1)
        y_ = dot+self.global_bias+item_bias+user_bias
        L2Loss = torch.norm(user_embedding)**2 + torch.norm(item_embedding)**2
        return y_, L2Loss/user.shape[0]/2
        # MSELoss = nn.MSELoss()(y_, rating)
        # return MSELoss, L2Loss/user.shape[0]/2

    def getUsersRating(self, users):
        self.eval()
        with torch.no_grad():  # 不求导
            items_emb = self.item_embedding.weight
            users_emb = self.user_embedding(users)
            item_bias = self.item_bias_embedding.weight
            user_bias = self.user_bias_embedding(users)
            scores = torch.matmul(users_emb,
                                  items_emb.t()) + item_bias.t() + user_bias + self.global_bias
            return scores
