import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)                       # Python 随机模块
    np.random.seed(seed)                    # Numpy 随机模块
    torch.manual_seed(seed)                 # CPU 随机种子
    torch.cuda.manual_seed(seed)            # 当前GPU随机种子
    torch.cuda.manual_seed_all(seed)        # 所有GPU随机种子

    torch.backends.cudnn.deterministic = True   # 确保每次卷积结果一致
    torch.backends.cudnn.benchmark = False      # 禁止cudnn自动优化（否则可能不确定）