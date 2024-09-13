from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Koopa
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    #计算并返回频谱掩码，用于识别时间序列中的时不变成分
    def _get_mask_spectrum(self):
        """
        get shared frequency spectrums
        """
        train_data, train_loader = self._get_data(flag='train')
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)

        mask_spectrum = amps.topk(int(amps.shape[0]*self.args.alpha)).indices
        return mask_spectrum # as the spectrums of time-invariant component

    #构建模型
    def _build_model(self):
        #model_dict 是一个字典，它将模型名称（'Koopa'）映射到相应的模型类。
        model_dict = {
            'Koopa': Koopa,
        }
        self.args.mask_spectrum = self._get_mask_spectrum() #调用 _get_mask_spectrum 方法来计算并设置频谱掩码。这个掩码可能用于模型中，以识别或处理时间序列数据中的时不变成分。
        model = model_dict[self.args.model].Model(self.args).float() #根据 self.args.model 从 model_dict 中选择相应的模型类；使用 self.args（包含模型配置的参数）来实例化模型。；.float() 确保模型参数的数据类型为浮点数，这是进行梯度计算和训练所必需的。

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        #如果 self.args.use_multi_gpu 为 True 且 self.args.use_gpu 也为 True，则使用 torch.nn.DataParallel 来包装模型。
        # DataParallel 是 PyTorch 提供的一个工具，它可以自动分配数据到多个 GPU 上，并收集各个 GPU 的结果，从而实现模型在多个 GPU 上的并行训练。
        # device_ids=self.args.device_ids 指定了哪些 GPU 将被用于训练。

        return model #返回构建好的模型实例

    #获取数据
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    #选择优化器
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    #选择损失函数
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    #验证
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    #训练
    def train(self, setting):
        #数据加载 dataset，dataloader
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        #加载cheakpoint路径， default '/home/featurize/work/checkpoints/'
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        #记录当前时间的时间戳
        time_now = time.time()
        #获取训练数据加载器（train_loader）中批次（batch）的数量
        #train_loader 是一个迭代器，它按照设定的批次大小（batch size）将训练数据集分批加载。
        train_steps = len(train_loader)
        #早停机制
        #patience 允许在性能停止提升之前的最大周期数（epochs）。
        #verbose 是否在控制台打印早停相关信息。
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        #选择优化器 adm 结合了动量和自适应学习率
        model_optim = self._select_optimizer()
        #选择损失函数，评价函数
        criterion = self._select_criterion()
        #使用自动混合精度Automatic Mixed Precision
        #创建一个 GradScaler 实例。
        # GradScaler 需要在每次反向传播之前调用 scaler.scale(loss) 来缩放损失，
        # 然后在反向传播后调用 scaler.step(optimizer) 来更新优化器，
        # 最后调用 scaler.update() 来准备下一次迭代。
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        #开始训练
        for epoch in range(self.args.train_epochs):
            # epoch = iteration * batch_size
            iter_count = 0 #iter_count 用于记录在当前epoch中已经处理的迭代次数（即已经遍历的数据批次数）。在每个epoch开始时将其重置为0，然后在每次迭代（即处理一个批次的数据）时增加1。
            train_loss = [] #初始化一个空列表 train_loss，用于存储当前epoch中每个批次的损失值。在每个epoch的开始，这个列表被清空，以便只记录当前epoch的损失值。在每个批次的数据被处理后，该批次的损失值被添加到这个列表中。

            self.model.train() #这行代码将模型设置为训练模式。在PyTorch中，模型有两种模式：训练模式和评估模式。训练模式是默认模式，模型在这种模式下会进行正常的前向传播和反向传播，包括所有的参数更新。
            #model.eval() 将模型设置为评估模式
            epoch_time = time.time() #记录了当前epoch开始的时间。time.time() 函数返回当前时间的时间戳（以秒为单位），这个时间戳可以用来计算整个epoch的运行时间。
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader): #将数据解包，batch_x 输入数据 batch_y 标签数据
                iter_count += 1 #记录一个epoch中迭代次数
                model_optim.zero_grad() #梯度清零
                #数据类型转换：.float() 方法将数据转换为浮点数类型，进行梯度计算和反向传播
                #数据设备转移：.to(self.device) 方法将数据移动到指定的设备上，可以是 CPU 或 GPU
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                #batch_y[:, -self.args.pred_len:, :] ：
                #从 batch_y 张量中选取每个样本的最后 self.args.pred_len 个时间步的所有特征。
                # : 在第一个维度（批次维度，即样本数量）上选取所有的元素
                # -self.args.pred_len: 第二个维度（时间步维度）,负数索引表示从序列的末尾开始计数。这里 -self.args.pred_len 表示从最后一个时间步开始，向前数 pred_len 个时间步。
                # : 在第三个维度上（通常是特征维度）。这意味着选取每个选定时间步的所有特征
                #例子
                # 假设你有一个数据集，其中包含1000个样本，每个样本是一个包含连续100天的股票价格数据，每天记录了3种不同的股票价格。那么，一个样本的数据张量可能如下所示：

                # 批次维度：1000（样本数量）
                # 时间步维度：100（天数）
                # 特征维度：3（股票种类）
                # 在 PyTorch 中，这样的张量可能被表示为一个形状为 [1000, 100, 3] 的三维张量。
                #在时间序列预测中，切片操作用于分离出需要预测的部分。例如，如果你有一个时间序列数据集，其中每个样本包含100个时间步的数据，而你只对未来的10个时间步感兴趣，那么 self.args.pred_len 可能会被设置为10，这个表达式将会从每个样本中提取最后10个时间步的数据，用于训练或评估模型的预测能力。
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float() #创建一个与 batch_y 的最后 self.args.pred_len 个时间步相同形状的零张量，并将其转换为浮点数类型。
                #在序列生成模型（如某些类型的循环神经网络、Transformer等）中，解码器通常需要一个初始输入来启动序列生成过程。在训练阶段，解码器的输入通常是部分已知的目标序列或者一个特殊的起始标记，这里使用全零张量作为起点。
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                #batch_y[:, :self.args.label_len, :]：这部分代码从目标张量 batch_y 中切片，获取每个样本的前 self.args.label_len 个时间步。这些时间步包含了序列的部分已知信息，通常用作模型训练时的起点。
                #torch.cat 是 PyTorch 中的函数，用于沿着指定的维度拼接张量。这里，它沿着第二个维度（dim=1，时间步维度）拼接了两个张量：从 batch_y 中切片得到的部分已知序列和之前创建的全零张量 dec_inp。这种拼接操作通常用于构建序列生成任务中的初始解码器输入，将已知的历史信息和要生成的序列部分结合起来。
                #将张量转移到指定的设备上，这个设备可以是 CPU 或 GPU

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast(): #自动混合精度（Automatic Mixed Precision, AMP）训练的一个上下文管理器，用于在该代码块内自动将浮点运算转换为混合精度运算。它对模型的计算过程进行优化，使用更低的精度（如从 float32 到 float16）来减少计算资源的使用，同时尽量减少对模型精度的影响。
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) #在 PyTorch 中，这种调用方式是模型对象调用的简化形式，它自动将模型实例的 forward 方法作为被调用的方法。等价于self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0 #据模型的特征设置来确定输出的特征维度，如果特征设置为 'MS'（多变量预测单变量），则 f_dim 被设置为 -1，表示输出的最后一个维度；否则，它被设置为 0，表示输出的第一个维度
                        outputs = outputs[:, -self.args.pred_len:, f_dim:] #从模型输出中切片出最后 self.args.pred_len 个时间步的数据，这些数据是模型预测的未来序列。
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) #从目标数据 batch_y 中切片出最后 self.args.pred_len 个时间步的数据，并将它们转移到正确的设备（如 GPU）上。
                        # outputs 预测序列， batch_y 目标序列
                        loss = criterion(outputs, batch_y) #计算模型输出和目标数据之间的损失
                        train_loss.append(loss.item()) #将计算得到的损失值添加到 train_loss 列表中，用于后续的监控和日志记录。
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                #可视化
                if (i + 1) % 100 == 0: #检查的是每100次迭代。如果条件为真，执行下面的代码块。
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())) #打印当前的迭代次数（i + 1），当前的epoch（epoch + 1），以及当前批次的损失值（loss.item()）
                    speed = (time.time() - time_now) / iter_count #speed = (time.time() - time_now) / iter_count：计算从上一次打印信息以来，每次迭代的平均时间。这里 time.time() 返回当前时间的时间戳，time_now 是上一次打印信息时的时间戳，iter_count 是自上次打印以来的迭代次数。
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i) #预测训练剩余时间。这里，self.args.train_epochs 是总的训练轮数，epoch 是当前的轮数，train_steps 是每个epoch中的迭代次数。speed 是每次迭代的平均时间，整个表达式计算剩余所有迭代的预计总时间。
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time)) #打印每次迭代的平均速度和预计的剩余时间。
                    iter_count = 0 #重置计数器和时间戳
                    time_now = time.time()

                if self.args.use_amp:
                    #使用AMP
                    scaler.scale(loss).backward() #首先通过scaler.scale()方法对损失值进行缩放，然后调用.backward()方法来计算梯度
                    scaler.step(model_optim) #在计算完梯度后，使用scaler.step()方法来更新模型的参数。这个方法会结合缩放的梯度和优化器来更新模型的权重。
                    scaler.update() #scaler.update() 来准备下一次迭代 梯度缩放器的内部状态更新，以便于下一次反向传播。
                else:
                    #标准的反向传播步骤，
                    loss.backward() #反向传播 计算梯度
                    model_optim.step() #更新模型权重

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)) #打印训练周期耗时
            train_loss = np.average(train_loss) #计算平均训练损失
            vali_loss = self.vali(vali_data, vali_loader, criterion) #计算验证损失
            test_loss = self.vali(test_data, test_loader, criterion) #计算测试损失
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss)) #打印训练、验证和测试损失
            
            early_stopping(vali_loss, self.model, path) #早停
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            #根据训练的进度和预设的策略来动态调整学习率
            #这种动态调整学习率的策略可以帮助模型在训练的不同阶段以不同的速度收敛，通常在训练初期使用较大的学习率以快速收敛，而在训练后期减小学习率以细化模型的权重并提高模型的性能。
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        #在训练过程中，通常会在每个epoch后在验证集上评估模型的性能。如果发现模型的性能有所提高（例如，验证损失降低），则保存该模型的状态作为最佳模型。
        best_model_path = path + '/' + 'checkpoint.pth' #构建了保存最佳模型状态的文件路径。
        #def save_checkpoint(self, val_loss, model, path): #验证损失有改善时，在EarlyStopping类中的该方法会torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')保存模型的状态字典。状态字典包含了模型的所有参数（权重和偏差）。
        self.model.load_state_dict(torch.load(best_model_path))
        #torch.load() 函数读取保存在文件中的模型状态，并将其反序列化为 Python 对象。这个对象是一个状态字典，包含了模型中所有参数的名称和对应的值（权重和偏差）。
        #load_state_dict() 方法将这些参数值赋给模型的相应参数，这样模型就恢复到了保存时的状态。
        return self.model

    #测试
    def test(self, setting, test=0):
        #test 方法定义，接受 setting 和 test 两个参数。
        # self._get_data(flag='test') 调用数据获取方法，获取测试数据集和对应的数据加载器。
        test_data, test_loader = self._get_data(flag='test')
        #如果 test 参数为真，则打印加载模型的消息。
        # 使用 torch.load 加载保存在检查点路径的模型状态字典，并使用 load_state_dict 方法将其应用到模型。
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('/home/featurize/work/checkpoints/' + setting, 'checkpoint.pth')))

        #初始化两个空列表 preds 和 trues，用于存储预测结果和真实值。
        # 构建存储测试结果的文件夹路径。
        # 如果文件夹不存在，则创建它。
        preds = []
        trues = []
        folder_path = '/home/featurize/work/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        #将模型设置为评估模式，关闭dropout等仅在训练时使用的层。
        self.model.eval()
        #使用 torch.no_grad 上下文管理器，以减少内存消耗并加速推理过程，因为在推理过程中不需要计算梯度。
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                #对每个批次的数据进行预处理，包括类型转换和设备转移。
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input 同train
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                #使用AMP
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                #传统前向计算
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0 
                #f_dim 变量，它用于确定输出数据的特征维度索引。self.args.features 指定了模型的特征类型：

                # 如果 self.args.features 等于 'MS'（表示多变量预测单变量），则 f_dim 被设置为 -1，这意味着在处理输出数据时将选择最后一个特征。
                # 如果不是 'MS'，则 f_dim 被设置为 0，意味着选择第一个特征。这通常用于单变量预测任务。

                outputs = outputs[:, -self.args.pred_len:, f_dim:] #从模型的输出中切片出最后 self.args.pred_len 个时间步的数据，这些数据是模型预测的未来序列
                #从模型输出 outputs 中，选取所有批次（第一个冒号表示批次维度），最后 self.args.pred_len 个时间步（负数索引表示从末尾开始计数），以及从 f_dim 指定的特征维度开始到结束的所有特征。
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) #转移至GPU
                #将模型的输出和目标数据从 PyTorch 张量转换为 NumPy 数组，以便进行性能评估或其他处理
                #.detach()：从当前计算图中分离出张量，这样对其的修改不会影响到梯度计算。
                # .cpu()：将张量移动到 CPU，这是必要的步骤，因为 NumPy 无法直接处理 GPU 上的张量。
                # .numpy()：将张量转换为 NumPy 数组。
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                #将预测结果和真实值添加到 preds 和 trues 列表中
                preds.append(pred)
                trues.append(true)
                #如果是每20个批次batch，使用 visual 函数可视化预测结果和真实值。
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    #是 NumPy 库中的一个函数，用于将两个或多个数组沿着指定的轴连接起来。
                    #input[0, :, -1] 提取第一个样本（通常是一批数据中的一个）的所有时间步的最后一个特征
                    #true[0, :, -1] 和 pred[0, :, -1] 分别提取真实值和预测值的相同部分。
                    # 将两个数组沿着指定的轴（这里是轴0，即行）拼接起来。这样，gt 包含了输入特征和真实值的对比，而 pd 包含了输入特征和预测值的对比。
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        #将预测结果和真实值的列表转换为 NumPy 数组。
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        #调整数组形状，使其与原始数据的形状一致。
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #维度的大小应该自动计算，以便保持数据的总元素数量不变。在这里，使用 -1 通常是为了保持数据的总元素数量不变，同时明确指定其他维度的大小
        #这些表达式获取 preds 数组的倒数第二个和最后一个维度的大小。在 Python 中，shape[-2] 通常表示倒数第二个维度（例如，在三维数组中，这可能是批次大小或特征数量），而 shape[-1] 表示最后一个维度（通常是特征数量或时间步长）。这些表达式获取 preds 数组的倒数第二个和最后一个维度的大小。在 Python 中，shape[-2] 通常表示倒数第二个维度（例如，在三维数组中，这可能是批次大小或特征数量），而 shape[-1] 表示最后一个维度（通常是特征数量或时间步长）。
        print('test shape:', preds.shape, trues.shape)

        # result save
        #构建存储测试指标的文件夹路径。
        # 如果文件夹不存在，则创建它。
        folder_path = '/home/featurize/work/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues) #使用 metric 函数计算预测结果的多个指标，如平均绝对误差（MAE）、均方误差（MSE）、均方根误差（RMSE）、平均百分比误差（MAPE）和平均绝对百分比误差（MSPE）。
        print('mse:{}, mae:{}'.format(mse, mae))
        #打开（或创建）一个名为 result.txt 的文件，以追加模式。
        # 将设置和 MSE、MAE 写入文件。
        # 关闭文件。
        f = open("/home/featurize/work/result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        #将计算的指标、预测结果和真实值保存为 .npy 文件

        return

    #预测
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return