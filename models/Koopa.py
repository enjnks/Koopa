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
    
#计算单步转换矩阵self.K
class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """
    def __init__(self): 
        super(KPLayer, self).__init__()
        
        self.K = None # B E E 存储从数据中学习到的系统转换矩阵

    #执行单步预测
    def one_step_forward(self, z, return_rec=False, return_K=False): #是否返回重构的数据 #是否返回系统转换矩阵K
        B, input_len, E = z.shape #输入数据，形状为(B, input_len, E)，其中B是批大小，input_len是时间步长的数量，E是特征维度
        assert input_len > 1, 'snapshots number should be larger than 1' #断言 确保input_len大于1，否则抛出异常
        
        #数据分割
        x, y = z[:, :-1], z[:, 1:] #将输入数据z分割为x和y，分别对应于当前和下一个时间步的数据 
        #选取从第一个时间步到最后一个时间步之前的所有时间步，结果是一个形状为(B, input_len-1, E)的张量，将其赋值给变量x。x代表了每个序列中用于预测下一步的当前状态
        #选取从第二个时间步到最后一个时间步的所有时间步（1:表示从索引1开始到结束）。结果也是一个形状为(B, input_len-1, E)的张量，将其赋值给变量y。y代表了每个序列中紧随x之后的下一个状态
        
        # solve linear system 求解线性系统
        self.K = torch.linalg.lstsq(x, y).solution # B E E #使用torch.linalg.lstsq函数解决线性系统，找到最佳拟合的转换矩阵K。这个矩阵描述了系统从当前状态到下一个状态的转换
        if torch.isnan(self.K).any(): #如果转换矩阵K中存在NaN值，则将其替换为单位矩阵，以避免计算错误
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        #单步预测
        #torch.bmm：这是批量矩阵乘法（batch matrix-matrix product）的函数，用于执行两个张量的批量矩阵乘法。
        #z[:, -1:]：这个表达式选取z中每个批次的最后一个时间步的数据
        z_pred = torch.bmm(z[:, -1:], self.K) #利用转换矩阵K进行一步预测，计算z_pred
        if return_rec:
            #z[:, :1]：这个表达式选取z中每个批次的第一个时间步的数据，形状为(B, 1, E)。
            #torch.bmm(x, self.K)：这是对x（当前状态）和转换矩阵self.K进行批量矩阵乘法，以得到预测的下一个状态。结果的形状为(B, input_len-1, E)
            #这是重构的数据，将原始序列的第一个时间步和通过转换矩阵预测的下一个状态连接起来。结果的形状为(B, input_len, E)，与原始输入z的形状相同。
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred
    
    #执行多步预测
    #与one_step_forward函数相同的输入数据
    #pred_len：预测长度，即需要预测的未来时间步数
    def forward(self, z, pred_len=1):
        #检查预测长度：确保pred_len大于等于1，否则抛出异常
        assert pred_len >= 1, 'prediction length should not be less than 1'
        #调用one_step_forward函数进行一步预测，并获取重构的数据z_rec和预测的数据z_pred。
        z_rec, z_pred= self.one_step_forward(z, return_rec=True)
        #多步预测：通过循环，使用转换矩阵K对预测的数据进行进一步的多步预测，直到达到指定的预测长度pred_len。
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K) #再已经预测出的数据上继续往后预测
            z_preds.append(z_pred)
        #合并预测结果：使用torch.cat将所有步骤的预测结果合并到一个张量z_preds中
        z_preds = torch.cat(z_preds, dim=1)
        #返回结果：返回重构的数据z_rec和预测的数据z_preds
        return z_rec, z_preds

#计算多步转换矩阵self.K_step
class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """
    def __init__(self): 
        super(KPLayerApprox, self).__init__()
        
        self.K = None # B E E 用于存储从数据中学习到的系统单步转换矩阵。初始值为None，表示在前向传播中计算。
        self.K_step = None # B E E 用于存储多步转换矩阵。初始值也为None

    def forward(self, z, pred_len=1): 
        #输入数据，形状为(B, input_len, E)，其中B是批大小，input_len是时间步长的数量，E是特征维度
        #预测长度，即需要预测的未来时间步数
        # z:       B L E, koopman invariance space representation
        # z_rec:   B L E, reconstructed representation
        # z_pred:  B S E, forecasting representation
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:] #x 第一个时间步到倒数第二个时间步 #y 第二个时间步到最后一个时间步

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E 使用torch.linalg.lstsq函数解决线性系统，找到最佳拟合的单步转换矩阵K
        if torch.isnan(self.K).any(): #如果转换矩阵K中存在NaN值，则将其替换为单位矩阵，以避免计算错误
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1) # B L E  #这是重构的数据，将原始序列的第一个时间步和通过转换矩阵预测的下一个状态连接起来。
        
        if pred_len <= input_len: #如果pred_len小于等于input_len，则直接计算self.K的pred_len次幂作为多步转换矩阵self.K_step。
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any(): #如果多步转换矩阵self.K_step中存在NaN值，则将其替换为单位矩阵
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)

            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step) 
            #z[:, -pred_len:, :]：这个表达式用于从输入张量z中选取最后pred_len个时间步的数据。pred_len是预测长度，即我们想要预测未来多少时间步
            #z_pred：这是预测结果，它保存了使用多步转换矩阵self.K_step对最后pred_len个时间步的数据进行转换后得到的预测状态。结果的形状为(B, pred_len, E)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():  #如果多步转换矩阵self.K_step中存在NaN值，则将其替换为单位矩阵
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)

            temp_z_pred, all_pred = z, [] #初始化temp_z_pred为输入数据z的最后input_len步，all_pred为空列表，用于存储每一步的预测结果。
            for _ in range(math.ceil(pred_len / input_len)): #次数为pred_len除以input_len的上限
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step) #在每一循环中，使用多步转换矩阵self.K_step更新temp_z_pred，即计算下一个input_len步的预测
                all_pred.append(temp_z_pred) #将每一步的预测结果添加到all_pred列表中
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]
            #torch.cat(all_pred, dim=1)：将all_pred列表中的所有预测结果沿着时间维度（第二维）拼接起来。这样，如果pred_len大于input_len，就会包含多次应用多步转换矩阵的预测结果。
            #[:, :pred_len, :]：从拼接的结果中选取前pred_len个时间步，以确保输出的预测长度与所需的预测长度一致。

        return z_rec, z_pred
    
#用于利用滑动窗口内的局部变化来预测未来时变分量
class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """
    def __init__(self,
                 enc_in=8, #编码器的输入维度
                 input_len=96, #输入序列的长度
                 pred_len=96, #预测序列的长度
                 seg_len=24, #过去/未来滑动窗口的长度
                 dynamic_dim=128, #动态系统的状态维度
                 encoder=None, #在主模型类中构建，并传参
                 decoder=None,
                 multistep=False, #是否使用多步预测
                ):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder            
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input 输入序列被分割成多少段
        self.step = math.ceil(self.pred_len / self.seg_len)   # segment number of output 输出序列被分割成多少段
        self.padding_len = self.seg_len * self.freq - self.input_len #为了确保输入序列可以被整除，可能需要填充的长度
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer() #根据 multistep 参数选择使用 KPLayerApprox 或 KPLayer

    def forward(self, x):
        # x: B L C
        B, L, C = x.shape #输入数据，形状为 (B, L, C)，其中 B 是批次大小，L 是序列长度，C 是特征数量

        #填充
        #使用 torch.cat 将输入数据 x 与自身在末端填充，以确保长度为 seg_len * freq
        res = torch.cat((x[:, L-self.padding_len:, :], x) ,dim=1)
        #分割
        ##使用 chunk 方法将数据分割成 freq 个部分，每部分长度为 seg_len
        res = res.chunk(self.freq, dim=1)     # F x B P C, P means seg_len
        #重塑
        #将分割后的数据重塑为 (B, freq, -1)
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)   # B F PC
        #编码
        # 使用 encoder 将数据编码到动态维度
        res = self.encoder(res) # B F H
        ###动态预测###
        #使用 dynamics 模块（KPLayerApprox 或 KPLayer）进行一步或多步预测
        x_rec, x_pred = self.dynamics(res, self.step) # B F H, B S H

        #解码+#重塑预测结果
        x_rec = self.decoder(x_rec) # B F PC  #使用 decoder 将预测结果解码回原始维度
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C
        
        x_pred = self.decoder(x_pred)     # B S PC
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :] # B S C  S 是预测序列的长度

        return x_rec, x_pred

#用于利用回顾和预测窗口快照来预测未来是不变分量
class TimeInvKP(nn.Module):
    """
    Koopman Predictor with learnable Koopman operator
    Utilize lookback and forecast window snapshots to predict the future of time-invariant term
    """
    def __init__(self,
                 input_len=96, #输入序列的长度，过去窗口大小
                 pred_len=96, #预测序列的长度， 未来窗口大小
                 dynamic_dim=128, #动态系统的状态维度或特征维度
                 encoder=None,
                 decoder=None):
        super(TimeInvKP, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder #编码器和解码器网络 在主模型类中搭建
        self.decoder = decoder

        #库普曼算子
        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim) #初始化一个随机矩阵，行，列，用于后续的奇异值分解（SVD）
        U, _, V = torch.svd(K_init) # stable initialization #对K_init进行奇异值分解，得到U和V矩阵。这一步是为了得到一个稳定的库普曼算子初始化
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False) #创建一个线性层作为库普曼算子，它的权重初始化为从SVD得到的U和V矩阵的乘积，从而得到一个正交矩阵。由于bias=False，这个线性层没有偏置项。
        self.K.weight.data = torch.mm(U, V.t()) #将库普曼算子的权重设置为U和V的转置矩阵乘积，这是一种正交初始化方法，有助于保持数值稳定性。
    
    def forward(self, x):
        # x: B L C 输入数据，形状为(B, L, C)，其中B是批大小，L是序列长度，C是特征数量。
        res = x.transpose(1, 2) # B C L  首先，将输入数据从(B, L, C)转置为(B, C, L)，这样每个批次的特征维度和序列维度就分开了
        res = self.encoder(res) # B C H  使用编码器将输入数据编码为库普曼空间的高维表示
        res = self.K(res) # B C H 应用库普曼算子（线性层）来模拟系统的动态行为
        res = self.decoder(res) # B C S 使用解码器将库普曼空间的表示解码回原始数据空间
        res = res.transpose(1, 2) # B S C 将数据从(B, C, S)转置回(B, S, C)，其中S是预测序列的长度，以匹配输出格式。

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
        # self.num_blocks = configs.num_blocks
        self.dynamic_dim = configs.dynamic_dim
        self.hidden_dim = configs.hidden_dim
        self.hidden_layers = configs.hidden_layers
        self.multistep = configs.multistep

        self.disentanglement = FourierFilter(self.mask_spectrum) #解耦时变和时不变

        # shared encoder/decoder to make koopman embedding consistent 时不变特征的编码器和解码器，使用MLP类实现
        self.time_inv_encoder = MLP(f_in=self.input_len, f_out=self.dynamic_dim, activation='relu',
                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        #self.time_inv_kps 和 self.time_var_kps：分别存储时不变和时变特征的库普曼预测器。这些预测器被包装在nn.ModuleList中，允许它们作为模块的一部分进行管理和更新。
        self.time_inv_kps = self.time_var_kps = nn.ModuleList([ #？？？
                                TimeInvKP(input_len=self.input_len,
                                    pred_len=self.pred_len, 
                                    dynamic_dim=self.dynamic_dim,
                                    encoder=self.time_inv_encoder, 
                                    decoder=self.time_inv_decoder)
                                for _ in range(self.num_blocks)]) #for _ in range(self.num_blocks)循环用于创建多个库普曼预测器（TimeInvKP和TimeVarKP）实例，并将它们分别存储在self.time_inv_kps和self.time_var_kps这两个nn.ModuleList容器中。这里的self.num_blocks表示模型中包含的预测器块的数量。

        # shared encoder/decoder to make koopman embedding consistent 时变特征的编码器和解码器，也使用MLP类实现
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

        # Series Stationarization adopted from NSformer 计算输入序列的均值和标准差，并进行标准化处理，以确保序列的平稳性。
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        #x_enc.mean(1, keepdim=True)：计算x_enc在第二个维度（时间序列维度）上的均值。keepdim=True参数保持输出的维度与输入相同，这样可以直接用于后续的广播操作。
        #detach()：从当前计算图中分离出这个张量，这意味着后续的操作不会在这个张量上构建梯度，通常用于防止梯度回传到预处理步骤。
        x_enc = x_enc - mean_enc #这个操作将原始数据x_enc中的每个时间序列减去其均值，目的是移除数据的均值，使数据的均值为零。
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc #这个操作将去均值后的数据x_enc除以其标准差，进行标准化处理，使数据的方差接近于1。

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
        res = forecast * std_enc + mean_enc #用于对时间序列数据进行平稳化处理的逆操作，即将经过模型预测的数据转换回其原始的尺度。

        return res
    


    #代码的主要流程如下：

    # 输入时间序列数据x_enc，首先进行平稳化处理。
    # 使用FourierFilter将序列分解为时变和时不变部分。
    # 对于每个分解部分，分别使用TimeInvKP和TimeVarKP进行预测。
    # 将预测结果相加，得到最终的预测序列。
    # 对预测序列进行反平稳化处理，得到最终的预测结果。
