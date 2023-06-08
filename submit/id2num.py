users_path = '../users_list.txt'
items_path = '../items_list.txt'
users_list = []
items_list = []
with open(users_path) as users_fp, open(items_path) as items_fp:
    for line in users_fp.readlines():
        users_list.append(line.rstrip())
    for line in items_fp.readlines():
        items_list.append(line.rstrip())

users_dict = {user_id: index for index, user_id in enumerate(users_list)}
items_dict = {item_id: index for index, item_id in enumerate(items_list)}

test_users_path = '../test_a_users.txt'
test_items_path = '../test_a_items.txt'
target_test_users_path = './test_a_users.txt'
target_test_items_path = './test_a_items.txt'
with open(test_users_path) as users_fp, open(test_items_path) as items_fp, open(target_test_users_path, "w") as target_users_fp, open(target_test_items_path, "w") as target_items_fp:
    for line in users_fp.readlines():
        user_id = line.rstrip()
        target_users_fp.write(str(users_dict[user_id])+'\n')
    for line in items_fp.readlines():
        item_id = line.rstrip()
        try:
            target_items_fp.write(str(items_dict[item_id])+'\n')
        except:
            print(item_id)  # 存在34个商品，它们在候选集里但无任何信息
            pass
