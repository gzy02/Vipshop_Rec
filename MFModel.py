import torch
import torch.nn as nn
import config


class MFModel(torch.nn.Module):
    def __init__(self, load_model):
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
        if load_model == False:
            nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
            nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)
            nn.init.xavier_uniform_(self.item_cat_embedding.weight, gain=1)
            nn.init.xavier_uniform_(self.item_brand_embedding.weight, gain=1)
            print('use xavier initilizer')

    def bpr_loss(self, user, item, item_cat, item_brand,neg_item,neg_item_cat,neg_item_brand):
        self.train()
        user_embedding = self.user_embedding(user)
        item_embedding = torch.concat((self.item_embedding(item), self.item_cat_embedding(
            item_cat), self.item_brand_embedding(item_brand)), dim=1)
        
        neg_item_embedding = torch.concat((self.item_embedding(neg_item), self.item_cat_embedding(
            neg_item_cat), self.item_brand_embedding(neg_item_brand)), dim=1)
 
        pos_scores = torch.sum(torch.mul(user_embedding, item_embedding), dim=1)
        neg_scores = torch.sum(torch.mul(user_embedding, neg_item_embedding), dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss

    def getUsersRating(self, user, item, item_cat, item_brand):
        self.eval()
        with torch.no_grad():  # 不求导
            user_embedding = self.user_embedding(user)
            item_embedding = torch.concat((self.item_embedding(item), self.item_cat_embedding(
                item_cat), self.item_brand_embedding(item_brand)), dim=1)
            dot = torch.sum(torch.mul(user_embedding, item_embedding), dim=1)
        return dot