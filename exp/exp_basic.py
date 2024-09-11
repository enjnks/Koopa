import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device) 
        #self._build_model() 调用了一个抽象方法，该方法应该在子类中被重写以构建具体的神经网络模型。这个方法返回一个模型实例。
        #将模型移动到 self.device 指定的设备上。这个设备可以是 CPU 或 GPU，由 _acquire_device 方法根据传入的参数决定。

    def _build_model(self):
        #NotImplementedError 是一个内置异常，用于指出一个方法是一个抽象方法，也就是说它是一个接口，需要在子类中具体实现。
        #强制子类实现：它强制任何继承 Exp_Basic 类的子类都必须实现自己的 _build_model 方法。这是一种确保子类提供特定功能的方法，即构建模型的具体实现。
        raise NotImplementedError 

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
