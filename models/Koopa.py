import math
import torch
import torch.nn as nn


#将时间序列分解为（周期性）时变（time-variant）和(非周期性)时不变（time-invariant）部分
class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """
    def __init__(self, mask_spectrum): 
        #参数mask_spectrum, 是一个布尔数组或者索引数组，用于指定在傅里叶变换后的频谱中哪些频率成分应该被保留，哪些应该被置零。
        #在时不变成分的提取中，通常将低频成分保留，而将高频成分置零。
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum
        
    def forward(self, x):
        #函数接收一个输入x，这是一个三维张量，其形状为(B, L, C)，其中B是批大小，L是序列长度，C是特征数量。
        xf = torch.fft.rfft(x, dim=1) #这个函数对输入x的每个序列（沿着维度1）进行快速傅里叶变换（FFT），将时域信号转换到频域。rfft是实数输入的FFT，它返回的是每个序列的傅里叶变换的振幅和相位信息。
        mask = torch.ones_like(xf) #创建一个与xf形状相同的全一张量，用于构建掩码。
        mask[:, self.mask_spectrum, :] = 0 #根据mask_spectrum参数，将掩码中的特定频率成分置零。在逆变换时，这些频率成分将不会被考虑，从而可以从原始信号中去除这些成分。
        # 第一个:表示对所有批次（batch）进行操作。
        # self.mask_spectrum表示在频谱的第二维（通常是频率维度）中选择特定的索引，这些索引对应于需要被置零的频率成分。
        # 第三个:表示对所有特征进行操作。
        x_var = torch.fft.irfft(xf*mask, dim=1) #使用修改后的频谱（原始频谱与掩码相乘）进行逆傅里叶变换，得到时变成分。这是因为在频域中置零的频率成分在时域中对应于原信号的时变部分。
        #这是逆变换的结果，它包含了时变成分的时域表示。由于在频域中已经通过掩码去除了某些频率成分，x_var只包含了原始信号中对应于未被置零频率成分的部分。换句话说，x_var是原始信号中变化部分的表示，这些变化部分可能对应于信号中的周期性或非平稳成分。
        #dim=1意味着对每个批次中的序列（即每个时间序列）单独进行逆变换。这是因为输入x的形状是(B, L, C)，其中B是批次大小，L是序列长度，C是特征数量。dim=1确保了对每个序列独立处理。
        x_inv = x - x_var #计算时不变成分，即原始信号减去时变成分。
        
        return x_var, x_inv
    
#多层感知机（Multilayer Perceptron）类，用于编码和解码序列数据的高维表示。它包含输入层、隐藏层和输出层，以及激活函数和dropout。
class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=128, 
                 hidden_layers=2, 
                 dropout=0.05,
                 activation='tanh'): 
        #初始化
        super(MLP, self).__init__()
        self.f_in = f_in #输入特征的维度
        self.f_out = f_out #输出特征的维度
        self.hidden_dim = hidden_dim #隐藏层神经元的数量，传参为128
        self.hidden_layers = hidden_layers #隐藏层的数量，传参为2
        self.dropout = dropout #为了防止过拟合，设置的dropout比率，传参为0.05。
        #激活函数的类型，传参为'tanh'
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError #如果是其他值，则抛出NotImplementedError异常，表示不支持的激活函数

        #构建网络层
        #layers列表用于存储网络中的所有层
        #首先，创建一个nn.Linear层，将输入特征从f_in维度映射到hidden_dim维度，然后添加激活函数和dropout层。
        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        #self.hidden_layers是构造函数中定义的隐藏层总数。减去2是因为第一个隐藏层已经在循环之前被添加到了layers列表中，而循环的最后一次迭代将添加最后一个隐藏层（如果存在）。例如，如果self.hidden_layers是2，那么这个循环不会执行，因为range(0)是空的；如果self.hidden_layers是3，循环将执行一次。
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        #最后，添加一个nn.Linear层，将最后一个隐藏层的特征从hidden_dim映射到输出特征的维度f_out。
        layers += [nn.Linear(hidden_dim, f_out)]
        #使用nn.Sequential容器将所有层按顺序包装起来，这样可以通过简单地调用self.layers(x)来执行前向传播。
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x) #x：输入张量，形状为(B, S, f_in)，其中B是批大小，S是序列长度，f_in是输入特征的维度。
        return y #y：输出张量，通过self.layers(x)计算得到，形状为(B, S, f_out)，表示经过多层感知机处理后的特征。
    
#用于通过动态模态分解（DMD）迭代地找到线性系统的一步转换
class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """
    def __init__(self): 
        super(KPLayer, self).__init__()
        
        self.K = None # B E E

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E
        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred
    
    def forward(self, z, pred_len=1):
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred= self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds

#用于通过多步K近似来找到线性系统的库普曼转换
class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """
    def __init__(self): 
        super(KPLayerApprox, self).__init__()
        
        self.K = None # B E E
        self.K_step = None # B E E

    def forward(self, z, pred_len=1):
        # z:       B L E, koopman invariance space representation
        # z_rec:   B L E, reconstructed representation
        # z_pred:  B S E, forecasting representation
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1) # B L E
        
        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]

        return z_rec, z_pred
    
#用于利用滑动窗口内的局部变化来预测时间变化项的未来
class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """
    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=24,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None,
                 multistep=False,
                ):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder            
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.pred_len / self.seg_len)   # segment number of output
        self.padding_len = self.seg_len * self.freq - self.input_len
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer() 

    def forward(self, x):
        # x: B L C
        B, L, C = x.shape

        res = torch.cat((x[:, L-self.padding_len:, :], x) ,dim=1)

        res = res.chunk(self.freq, dim=1)     # F x B P C, P means seg_len
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)   # B F PC

        res = self.encoder(res) # B F H
        x_rec, x_pred = self.dynamics(res, self.step) # B F H, B S H

        x_rec = self.decoder(x_rec) # B F PC
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C
        
        x_pred = self.decoder(x_pred)     # B S PC
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :] # B S C

        return x_rec, x_pred

#用于利用回顾和预测窗口快照来预测时间不变项的未来
class TimeInvKP(nn.Module):
    """
    Koopman Predictor with learnable Koopman operator
    Utilize lookback and forecast window snapshots to predict the future of time-invariant term
    """
    def __init__(self,
                 input_len=96,
                 pred_len=96,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None):
        super(TimeInvKP, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder

        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init) # stable initialization
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())
    
    def forward(self, x):
        # x: B L C
        res = x.transpose(1, 2) # B C L
        res = self.encoder(res) # B C H
        res = self.K(res) # B C H
        res = self.decoder(res) # B C S
        res = res.transpose(1, 2) # B S C

        return res

#主模型类，完整的库普曼预测模型
class Model(nn.Module):
    '''
    Koopman Forecasting Model
    '''
    def __init__(self, configs):
        super(Model, self).__init__() #继承父类的初始化方法
        self.mask_spectrum = configs.mask_spectrum
        self.enc_in = configs.enc_in
        self.input_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.num_blocks = configs.num_blocks
        self.dynamic_dim = configs.dynamic_dim
        self.hidden_dim = configs.hidden_dim
        self.hidden_layers = configs.hidden_layers
        self.multistep = configs.multistep

        self.disentanglement = FourierFilter(self.mask_spectrum) #解耦时变和时不变

        # shared encoder/decoder to make koopman embedding consistent 共享编码器/解码器以保持库普曼嵌入的一致性
        self.time_inv_encoder = MLP(f_in=self.input_len, f_out=self.dynamic_dim, activation='relu',
                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_kps = self.time_var_kps = nn.ModuleList([ #？？？
                                TimeInvKP(input_len=self.input_len,
                                    pred_len=self.pred_len, 
                                    dynamic_dim=self.dynamic_dim,
                                    encoder=self.time_inv_encoder, 
                                    decoder=self.time_inv_decoder)
                                for _ in range(self.num_blocks)]) #占位符，表示循环变量在循环体内不会被使用

        # shared encoder/decoder to make koopman embedding consistent
        self.time_var_encoder = MLP(f_in=self.seg_len*self.enc_in, f_out=self.dynamic_dim, activation='tanh',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len*self.enc_in, activation='tanh',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_kps = nn.ModuleList([
                    TimeVarKP(enc_in=configs.enc_in,
                        input_len=self.input_len,
                        pred_len=self.pred_len,
                        seg_len=self.seg_len,
                        dynamic_dim=self.dynamic_dim,
                        encoder=self.time_var_encoder,
                        decoder=self.time_var_decoder,
                        multistep=self.multistep)
                    for _ in range(self.num_blocks)])
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: B L C

        # Series Stationarization adopted from NSformer 从NSformer中采用的序列平稳化处理
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Koopman Forecasting
        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = (time_inv_output + time_var_output)
            else:
                forecast += (time_inv_output + time_var_output)

        # Series Stationarization adopted from NSformer
        res = forecast * std_enc + mean_enc

        return res
    


    #代码的主要流程如下：

    # 输入时间序列数据x_enc，首先进行平稳化处理。
    # 使用FourierFilter将序列分解为时变和时不变部分。
    # 对于每个分解部分，分别使用TimeInvKP和TimeVarKP进行预测。
    # 将预测结果相加，得到最终的预测序列。
    # 对预测序列进行反平稳化处理，得到最终的预测结果。
