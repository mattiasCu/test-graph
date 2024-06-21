import os
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch.optim import lr_scheduler

from torchdrug import core, utils, datasets, models, tasks
from torchdrug.utils import comm


logger = logging.getLogger(__file__)                                        #获取当前文件的日志记录器

"""获取根日志记录器"""
def get_root_logger(file=True):
    logger = logging.getLogger("")                                          #获取根日志记录器
    logger.setLevel(logging.INFO)                                           #设置日志记录器的日志级别
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")    #设置日志记录器的格式

    if file:
        handler = logging.FileHandler("log.txt")                            #创建一个文件处理器
        handler.setFormatter(format)                                        #设置文件处理器的格式
        logger.addHandler(handler)                                          #将文件处理器添加到日志记录器中

    return logger


"""创建工作目录
        构建一个带有特定结构的工作目录路径。
        在分布式计算环境中，同步所有进程使用相同的工作目录路径。
        创建工作目录（仅由主进程执行），其他进程从临时文件中读取路径。
        切换到创建的工作目录并返回路径
"""
def create_working_directory(cfg):
    file_name = "working_dir.tmp"                                           #临时文件名
    world_size = comm.get_world_size()                                      #获取当前进程组中的进程数量

    #如果 world_size 大于 1 且分布式环境尚未初始化，则初始化分布式进程组
    if world_size > 1 and not dist.is_initialized():                        
        comm.init_process_group("nccl", init_method="env://")               # nccl 是一种高效的通信库，通常用于 GPU 间通信
                                                                            # init_method="env://" 指定使用环境变量进行初始化
      
    #创建工作目录路径
    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),                                  #将用户路径转换为绝对路径（home目录）
                               cfg.task["class"], cfg.dataset["class"], cfg.task.model["class"],
                               time.strftime("%Y-%m-%d-%H-%M-%S"))

    # 分级创建工作目录
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:                                  #打开一个名为file_name的文件以写入模式，fout是这个打开文件的引用
            fout.write(working_dir)                                         #将工作目录路径写入名为fout的文件
        os.makedirs(working_dir)                                            #创建工作目录
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()                                        #这行代码读取了 fin 所指向的文件的全部内容，并将其赋值给 working_dir
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)                                                   #将当前工作目录更改为 working_dir 所指向的目录
    return working_dir


"""解析命令行参数
    返回未声明的变量"""
def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()                                              #创建一个 Jinja2 模板引擎的环境 env，它提供了解析和渲染模板的功能                                
    ast = env.parse(raw)                                                    #将读取的原始内容 raw 解析成抽象语法树（AST）
    vars = meta.find_undeclared_variables(ast)                              #meta.find_undeclared_variables(ast) 是 Jinja2 提供的一个工具函数
                                                                            #用于从解析的 AST 中提取所有未声明的变量 ---这里是gpus
    return vars


"""加载配置文件"""
def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)                                         #创建一个 Jinja2 模板对象，使用读取的文件内容 raw 作为模板
    instance = template.render(context)                                     #使用提供的 context 字典渲染模板，将模板中的变量替换为字典中相应的值
    cfg = yaml.safe_load(instance)                                          #将渲染后的模板内容 instance 解析为一个 Python 字典 cfg
    cfg = easydict.EasyDict(cfg)                                            #将 Python 字典 cfg 转换为一个易于访问的 EasyDict 对象

    return cfg


"""解析命令行参数
    从一个 YAML 配置文件中提取未声明的变量，
    然后将这些变量也作为命令行参数来解析。
    最终，返回解析的参数和从命令行传递的动态变量
"""
def parse_args():
    parser = argparse.ArgumentParser()                                                              #创建一个命令行参数解析器
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)            #添加一个参数
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)     #添加一个参数

    args, unparsed = parser.parse_known_args()                                              #解析命令行参数，将解析的已知参数存储在 args 中  
                                                                                            #将未知的剩余参数存储在 unparsed 中
    # 获取在配置文件中定义的动态参数
    vars = detect_variables(args.config)                                                    #检测并返回 YAML 配置文件中未声明的变量
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, default="null")
    vars = parser.parse_known_args(unparsed)[0]                                             #新的解析器解析 unparsed 参数，提取动态变量                                             
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}                        #将解析的动态变量转换为字典形式，并对每个变量的值进行字面量求值

    return args, vars


def build_downstream_solver(cfg, dataset):

    #分割数据集
    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    #将任务设置为多标签分类任务
    if cfg.task['class'] == 'MultipleBinaryClassification':
        cfg.task.task = [_ for _ in range(len(dataset.tasks))]              
                                                                            #[_ for _ in range(len(dataset.tasks))] 是一个列表推导式，
                                                                            #它遍历了 range(len(dataset.tasks)) 中的每一个元素,
                                                                            #然后将这些元素添加到一个新的列表中。
    else:
        cfg.task.task = dataset.tasks

    #加载任务配置
    task = core.Configurable.load_config_dict(cfg.task)                     
                                                                            #首先检查 cfg.task 中是否有 "class" 这个键。如果有，那么它会在注册表中查找对应的类。
                                                                            #如果找到了，那么它会创建这个类的一个新的实例。如果没有找到，那么它会创建一个 core.Configurable 的实例
                                                                            #然后，它会将 cfg.task 中的所有键值对传递给这个实例的构造函数。

    #配置schedule的optimizer
    cfg.optimizer.params = task.parameters()        
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    #配置schedule
    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")                                                  #将 class 键从 cfg.scheduler 中移除
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)      #创建一个 ReduceLROnPlateau 类的实例
                                                                                    #使用剩余的 cfg.scheduler 配置参数来初始化 ReduceLROnPlateau 调度器
                                                                                    #**表示将一个字典解包为关键字参数，传递给函数
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        cfg.engine.scheduler = scheduler

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if "lr_ratio" in cfg:
        cfg.optimizer.params = [
            {'params': solver.model.model.parameters(), 'lr': cfg.optimizer.lr * cfg.lr_ratio},
            {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer
    elif "sequence_model_lr_ratio" in cfg:
        assert cfg.task.model["class"] == "FusionNetwork"
        cfg.optimizer.params = [
            {'params': solver.model.model.sequence_model.parameters(), 'lr': cfg.optimizer.lr * cfg.sequence_model_lr_ratio},
            {'params': solver.model.model.structure_model.parameters(), 'lr': cfg.optimizer.lr},
            {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer

    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    elif scheduler is not None:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        solver.scheduler = scheduler

    if cfg.get("checkpoint") is not None:
        solver.load(cfg.checkpoint)

    #如果有模型检查点
    if cfg.get("model_checkpoint") is not None:
        #记录日志
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))         #加载模型检查点文件。
                                                                                                #map_location=torch.device('cpu') 表示将模型加载到 CPU 上
                                                                                                #这在多 GPU 环境下可以确保模型在主进程中正确加载
                                                                                                
        task.model.load_state_dict(model_dict)                                                  #将加载的模型状态字典 model_dict 应用到当前的模型 task.model 中
                                                                                                #这个函数会将模型的参数加载到模型中，但不会加载优化器的状态
    
    return solver, scheduler


def build_pretrain_solver(cfg, dataset):
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#dataset: %d" % (len(dataset)))

    task = core.Configurable.load_config_dict(cfg.task)
    if "fix_sequence_model" in cfg:
        if cfg.task["class"] == "Unsupervised":
            model_dict = cfg.task.model.model
        else:
            model_dict = cfg.task.model 
        assert model_dict["class"] == "FusionNetwork"
        for p in task.model.model.sequence_model.parameters():
            p.requires_grad = False
    cfg.optimizer.params = [p for p in task.parameters() if p.requires_grad]
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    solver = core.Engine(task, dataset, None, None, optimizer, **cfg.engine)
    
    return solver
