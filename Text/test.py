import random
import numpy as np
import torch
def set_seed(seed_value):
    print("1. 开始设置 Python 随机种子", flush=True)
    random.seed(seed_value)

    print("2. 开始设置 NumPy 随机种子", flush=True)
    np.random.seed(seed_value)

    print("3. 开始设置 PyTorch 随机种子", flush=True)
    torch.manual_seed(seed_value)

    print("4. 准备检查 CUDA", flush=True)
    cuda_available = torch.cuda.is_available()
    print("5. CUDA 是否可用：", cuda_available, flush=True)

    if cuda_available:
        print("6. 开始设置 CUDA 随机种子", flush=True)
        torch.cuda.manual_seed_all(seed_value)
        print("7. CUDA 随机种子设置完成", flush=True)

    print("8. 设置 cuDNN 配置", flush=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("9. 全部完成", flush=True)


set_seed(42)