import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        #如果选择的是 type1，则使用指数衰减策略。学习率随着每个epoch减少到原来的一半。这里的指数衰减基数是0.5，意味着每个epoch学习率都会乘以0.5。
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        #如果选择的是 type2，则使用预设的固定周期学习率调整策略。在这个策略中，学习率在特定的epoch被手动设置为特定的值。
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys(): #如果当前的epoch在 lr_adjust 字典的键中，那么根据 lr_adjust 字典中的值更新学习率。
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups: #遍历优化器中的所有参数组。
            param_group['lr'] = lr #将每个参数组的学习率更新为新的学习率 lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience #耐心阈值
        self.verbose = verbose #是否打印详细信息
        self.counter = 0 #用于跟踪验证损失没有改善的连续周期数
        self.best_score = None #在训练过程中观察到的最佳分数（这里使用负验证损失，因此分数越高越好）
        self.early_stop = False #一个标志，指示是否应该提前停止训练。
        self.val_loss_min = np.Inf #迄今为止观察到的最小验证损失。
        self.delta = delta #确定性能提升的最小阈值。

    #该方法在每个训练周期结束时被调用，传入当前的验证损失。
    #val_loss：当前周期的验证损失。
    # model：当前模型。
    # path：保存模型检查点的路径。
    def __call__(self, val_loss, model, path):
        score = -val_loss #取负值
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta: #验证损失没有改善
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: #超过阈值
                self.early_stop = True #停止
        else: #验证损失有改善
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        #输出
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend() #plt.legend()：显示图例，包括 "GroundTruth" 和 "Prediction" 两个标签。
    plt.savefig(name, bbox_inches='tight') #将图表保存为PDF文件。bbox_inches='tight' 参数确保保存的图像尽可能紧凑，没有多余的边界。
