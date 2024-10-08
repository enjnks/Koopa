import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Koopa for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status') #启动训练
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Koopa',
                    help='model name, options: [Koopa]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh2', help='dataset type') #数据集类型
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate M：多变量预测多变量，S：单变量预测单变量，MS：多变量预测单变量')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h') #时间特征编码的频率选项：[s:每秒, t:每分钟, h:每小时, d:每天, b:工作日, w:每周, m:每月]，还可以使用更详细的频率，如15分钟或3小时。
parser.add_argument('--checkpoints', type=str, default='/home/featurize/work/checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times') #实验运行次数 1
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data') #批大小，dataloader
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--seed', type=int, default=2023, help='random seed')

# Koopa
parser.add_argument('--dynamic_dim', type=int, default=128, help='latent dimension of koopman embedding')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of en/decoder')
parser.add_argument('--hidden_layers', type=int, default=2, help='number of hidden layers of en/decoder')
parser.add_argument('--seg_len', type=int, default=48, help='segment length of time series')
parser.add_argument('--num_blocks', type=int, default=3, help='number of Koopa blocks')
parser.add_argument('--alpha', type=float, default=0.2, help='spectrum filter ratio')
parser.add_argument('--multistep', action='store_true', help='whether to use approximation for multistep K', default=False)


args = parser.parse_args() #解析参数

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

fix_seed = args.seed
random.seed(fix_seed) #设置Python标准库random模块的随机种子
torch.manual_seed(fix_seed) #设置PyTorch的全局随机种子
np.random.seed(fix_seed) #设置NumPy的随机种子

if args.use_gpu:
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        torch.cuda.set_device(args.gpu)

print('Args in experiment:')
print(args)

Exp = Exp_Main #将 Exp_Main 类赋值给 Exp 这个变量，而非初始化对象

if args.is_training: 
    #启动实验
    for ii in range(args.itr): #实验运行次数 1次
        #实验参数设置
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_segl{}_dyna{}_h{}_l{}_nb{}_a{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.seg_len,
            args.dynamic_dim,
            args.hidden_dim,
            args.hidden_layers,
            args.num_blocks,
            args.alpha,
            args.des, ii)

        exp = Exp(args)  # Exp 引用 Exp_Main，初始化对象，将参数容器arg传入
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting)) #输出参数设置
        exp.train(setting) #训练

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting) #测试

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True) #验证，默认不开启

        torch.cuda.empty_cache() #清理CUDA缓存
else:
    #关闭实验
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_segl{}_dyna{}_h{}_l{}_nb{}_a{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.seg_len,
        args.dynamic_dim,
        args.hidden_dim,
        args.hidden_layers,
        args.num_blocks,
        args.alpha,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
