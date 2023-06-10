
import config
import torch
import numpy as np
import random


def print_config():
    config_order = [
        "seed",
        "batch_size",
        "epochs",
        "lr",
        "weight_decay",
        "user_dim",
        "cat_dim",
        "brand_dim",
        "item_dim",
        "topk"
    ]

    print("-------------------------------")
    for name in config_order:
        print(f"{name}: {getattr(config, name)}")
    print("-------------------------------")


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def get_test_set(path: str) -> set:
    test_set = set()
    with open(path) as fp:
        for line in fp.readlines():
            test_set.add(int(line.rstrip()))
    return test_set


if __name__ == "__main__":
    test_items_set, test_users_set = get_test_set(
        config.test_items_path), get_test_set(config.test_users_path)
    print(len(test_users_set))
