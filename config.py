seed = 2023

batch_size = 1024*2048

# 训练
epochs = 2000
lr = 1e-3
weight_decay = 1e-7

user_dim = 256
cat_dim = 8
brand_dim = 24
item_dim = user_dim-cat_dim-brand_dim

# 测试
topk = 10

# %% 是否使用预训练好的模型
load_weight = False
load_epoch = 30 if load_weight == True else 0
load_path = f"./model/{load_epoch}_64_0.01_0.pth"

items_num = 3465659     # 物品数量 3465659
items_cat_num = 1855    # 物品种类数量 1804+51
items_brand_num = 5925  # 品牌数量 5874+51
users_num = 51602       # 用户数量 51602

items_names = ['items_id', 'cat_id', 'brandsn']
user_item_names = ['user_id', 'items_id', 'is_clk', 'is_like',
                   'is_addcart', 'is_order', 'expose_start_time', 'dt']

test_set_dir = "./test_set_a/"
test_items_path = test_set_dir+'test_a_items.txt'
test_users_path = test_set_dir+'test_a_users.txt'

train_set_dir = './train_set/'
user_item_info_path = train_set_dir+'user_item_info_rank.csv'
items_info_path = train_set_dir+'items_info.csv'
