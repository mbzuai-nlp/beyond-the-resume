def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    import polars as pl
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.set_random_seed(seed)