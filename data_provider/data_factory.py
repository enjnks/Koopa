from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data] #字典 不同的数据集类型，对应不同的dataset类
    timeenc = 0 if args.embed != 'timeF' else 1

    #按照不同的数据集，设置不同的dataloader参数
    if flag == 'test':
        shuffle_flag = False #乱序
        drop_last = False #	是否丢弃最后一个样本数量不足batch_size批次数据
        batch_size = 1 #批大小
        freq = args.freq #
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        # flag == 'train'，flag == 'val'
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

# Dataset	一个数据集抽象类，所有自定义的Dataset都需要继承它，并且重写__getitem__()或__get_sample__()这个类方法
# DataLoader	一个可迭代的数据装载器。在训练的时候，每一个for循环迭代，就从DataLoader中获取一个batch_sieze大小的数据。
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers, #使用多进程读取数据，设置的进程数
        drop_last=drop_last
    )
    return data_set, data_loader
