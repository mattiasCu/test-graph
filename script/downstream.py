import os
import sys
import math
import pprint
import random

import numpy as np

import torch
from torch.optim import lr_scheduler

from torchdrug import core, models, tasks, datasets, utils
from torchdrug.utils import comm

#将当前文件的父目录的父目录添加到模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import util
from DGMGearnet import model


def train_and_validate(cfg, solver, scheduler):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 50)              #每step个epoch保存一次模型
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)     
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)            #保存模型，"model_epoch_%d.pth" % solver.epoch 是一个字符串格式化操作
        metric = solver.evaluate("valid")
        solver.evaluate("test")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result)

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver


def test(cfg, solver):
    solver.evaluate("valid")
    return solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()                                          #解析命令行参数
    cfg = util.load_config(args.config, context=vars)                       #加载配置文件
    working_dir = util.create_working_directory(cfg)                        #创建工作目录

    """设置随机种子"""
    seed = args.seed                                                
    torch.manual_seed(seed + comm.get_rank())                       
    os.environ['PYTHONHASHSEED'] = str(seed)                        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()                                         #获取日志记录器
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)               #加载数据集
    solver, scheduler = util.build_downstream_solver(cfg, dataset)          #构建下游任务求解器

    train_and_validate(cfg, solver, scheduler)
    test(cfg, solver)