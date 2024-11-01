# 原始代码
# import torch
# import torch.nn as nn
# from layers.Embed import PositionalEmbedding

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()

#         # get parameters
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.period_len = configs.period_len

#         self.seg_num_x = self.seq_len // self.period_len
#         self.seg_num_y = self.pred_len // self.period_len

#         self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
#                                 stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

#         self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)


#     def forward(self, x):
#         batch_size = x.shape[0]
#         # normalization and permute     b,s,c -> b,c,s
#         seq_mean = torch.mean(x, dim=1).unsqueeze(1)
#         x = (x - seq_mean).permute(0, 2, 1)

#         # 1D convolution aggregation
#         x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

#         # downsampling: b,c,s -> bc,n,w -> bc,w,n
#         x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

#         # sparse forecasting
#         y = self.linear(x)  # bc,w,m

#         # upsampling: bc,w,m -> bc,m,w -> b,c,s
#         y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

#         # permute and denorm
#         y = y.permute(0, 2, 1) + seq_mean

#         return y

# 多周期+周期预测
# import torch
# import torch.nn as nn


# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()

#         # 获取参数
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.period_len = configs.period_len

#         # 三个不同周期长度
#         self.period_len1 = self.period_len 
#         self.period_len2 = self.period_len // 2
#         self.period_len3 = self.period_len // 4

#         # 对应的分段数量
#         self.seg_num_x1 = self.seq_len // self.period_len1
#         self.seg_num_y1 = self.pred_len // self.period_len1
#         self.seg_num_x2 = self.seq_len // self.period_len2
#         self.seg_num_y2 = self.pred_len // self.period_len2
#         self.seg_num_x3 = self.seq_len // self.period_len3
#         self.seg_num_y3 = self.pred_len // self.period_len3

#         # 三个不同周期长度的卷积核
#         self.conv1d1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len1 // 2),
#                                  stride=1, padding=self.period_len1 // 2, padding_mode="zeros", bias=False)
#         self.conv1d2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len2 // 2),
#                                  stride=1, padding=self.period_len2 // 2, padding_mode="zeros", bias=False)
#         self.conv1d3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len3 // 2),
#                                  stride=1, padding=self.period_len3 // 2, padding_mode="zeros", bias=False)

#         # 用于加权求和的可学习参数
#         self.weight1 = nn.Parameter(torch.ones(1))
#         self.weight2 = nn.Parameter(torch.ones(1))
#         self.weight3 = nn.Parameter(torch.ones(1))

#         # 三个周期的全连接层
#         self.linear1 = nn.Linear(self.seg_num_x1, self.seg_num_y1, bias=False)
#         self.linear2 = nn.Linear(self.seg_num_x2, self.seg_num_y2, bias=False)
#         self.linear3 = nn.Linear(self.seg_num_x3, self.seg_num_y3, bias=False)

#     def forward(self, x):
#         batch_size = x.shape[0]

#         # 归一化和维度交换 b,s,c -> b,c,s
#         seq_mean = torch.mean(x, dim=1).unsqueeze(1)
#         x = (x - seq_mean).permute(0, 2, 1)

#         # 三个卷积核的特征提取
#         # filter1 = torch.tanh(self.conv1d1(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len))
#         # gate1 = torch.sigmoid(self.conv1d1(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len))
#         # x1 = filter1 * gate1 + x

#         # filter2 = torch.tanh(self.conv1d2(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len))
#         # gate2 = torch.sigmoid(self.conv1d2(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len))
#         # x2 = filter2 * gate2 + x

#         # filter3 = torch.tanh(self.conv1d3(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len))
#         # gate3 = torch.sigmoid(self.conv1d3(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len))
#         # x3 = filter3 * gate3 + x
        
#         x1 = self.conv1d1(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
#         x2 = self.conv1d2(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
#         x3 = self.conv1d3(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
        
#         # 下采样并调整维度
#         x1 = x1.reshape(-1, self.seg_num_x1, self.period_len1).permute(0, 2, 1)
#         x2 = x2.reshape(-1, self.seg_num_x2, self.period_len2).permute(0, 2, 1)
#         x3 = x3.reshape(-1, self.seg_num_x3, self.period_len3).permute(0, 2, 1)

#         # 稀疏预测
#         y1 = self.linear1(x1)
#         y2 = self.linear2(x2)
#         y3 = self.linear3(x3)

#         # 将维度统一，进行reshape
#         y1 = y1.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
#         y2 = y2.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
#         y3 = y3.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

#         # 进行加权求和
#         weight_all = self.weight1 + self.weight2 + self.weight3
#         y = self.weight1 * y1 + self.weight2 * y2 + self.weight3 * y3
#         y = y / weight_all


#         # 维度交换并反归一化
#         y = y.permute(0, 2, 1) + seq_mean
#         return y


# 单周期+线性预测
# import torch
# import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()

#         # 参数初始化
#         self.seq_len = configs.seq_len  # 历史序列长度
#         self.pred_len = configs.pred_len  # 预测序列长度
#         self.enc_in = configs.enc_in  # 输入特征维度
#         self.period_len = configs.period_len  # 周期长度

#         self.seg_num_x = self.seq_len // self.period_len  # 历史分段数量
#         self.seg_num_y = self.pred_len // self.period_len  # 预测分段数量

#         # 1D卷积层用于周期性聚合
#         self.conv1d = nn.Conv1d(
#             in_channels=1, out_channels=1, 
#             kernel_size=1 + 2 * (self.period_len // 2), 
#             stride=1, padding=self.period_len // 2, 
#             padding_mode="zeros", bias=False
#         )

#         # 稀疏预测的线性层
#         self.sparse_linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

#         # DLinear 线性模型（直接基于历史数据的预测）
#         self.direct_linear = nn.Linear(self.seq_len, self.pred_len, bias=False)

#         # 加权参数
#         self.weight_sparse = nn.Parameter(torch.tensor(0.5))  # 可学习参数，用于加权求和
#         self.weight_direct = nn.Parameter(torch.tensor(0.5))

#     def forward(self, x):
#         batch_size = x.shape[0]

#         # 归一化并调整维度: (b, s, c) -> (b, c, s)
#         seq_mean = torch.mean(x, dim=1).unsqueeze(1)  # 计算每个序列的均值
#         x_norm = (x - seq_mean).permute(0, 2, 1)  # 标准化后的输入 (b, c, s)

#         # 1D卷积聚合，用于周期性预测
#         x_conv = self.conv1d(x_norm.reshape(-1, 1, self.seq_len)).reshape(
#             -1, self.enc_in, self.seq_len
#         ) + x_norm

#         # 下采样: (b, c, s) -> (b*c, w, n) -> (b*c, n, w)
#         x_periodic = x_conv.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

#         # 稀疏预测部分
#         y_sparse = self.sparse_linear(x_periodic)  # (b*c, w, m)
#         y_sparse = y_sparse.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

#         # **直接预测部分** (每个特征单独线性预测)
#         y_direct = torch.zeros((batch_size, self.enc_in, self.pred_len), device=x.device)  # 预分配

#         for i in range(self.enc_in):
#             # 对每一特征分别预测：输入为 (batch_size, seq_len)
#             y_direct[:, i, :] = self.direct_linear(x_norm[:, i, :])

#         # **合并两部分预测** (加权求和)
#         y = (
#             self.weight_sparse * y_sparse + 
#             self.weight_direct * y_direct
#         )

#         # 调整维度并反归一化
#         y = y.permute(0, 2, 1) + seq_mean

#         return y

# dropout + 多周期 + 线性 +多尺度
# import torch
# import torch.nn as nn


# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()

#         # 参数初始化
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.period_len = configs.period_len
#         self.dropout_prob = configs.dropout_ratio

#         # 三个不同周期长度
#         self.periods = [self.period_len, self.period_len // 2, self.period_len // 4]

#         # 初始化卷积、线性层和 Dropout
#         self.convs = nn.ModuleList([
#             nn.Conv1d(1, 1, 1 + 2 * (p // 2), stride=1, padding=p // 2, bias=False)
#             for p in self.periods
#         ])
#         self.linears = nn.ModuleList([
#             nn.Linear(self.seq_len // p, self.pred_len // p, bias=False)
#             for p in self.periods
#         ])
#         self.dropouts = nn.ModuleList([
#             nn.Dropout(self.dropout_prob) for _ in self.periods  # 每个周期对应一个 Dropout
#         ])

#         # 加权求和的可学习参数
#         self.weights = nn.Parameter(torch.ones(len(self.periods)))

#         # 直接预测模型
#         self.direct_linear = nn.Linear(self.seq_len, self.pred_len, bias=False)
#         self.direct_linear1 = nn.Linear(self.seq_len // 2, self.pred_len, bias=False)
#         self.direct_linear2 = nn.Linear(self.seq_len // 3, self.pred_len, bias=False)  # 新增的线性层
#         self.direct_dropout = nn.Dropout(self.dropout_prob)  # 用于直接预测部分

#         # 稀疏和直接预测部分的可学习权重
#         self.weight_sparse = nn.Parameter(torch.tensor(0.3))
#         self.weight_direct = nn.Parameter(torch.tensor(0.7))
#         self.weight_direct1 = nn.Parameter(torch.tensor(0.7))  # 可学习的权重

#     def forward(self, x):
#         batch_size = x.shape[0]

#         # 归一化并交换维度 (b, s, c) -> (b, c, s)
#         x, seq_mean = self.normalize(x)

#         # 周期性预测部分：卷积和线性层输出的结果求和
#         y_periodic = sum(
#             self.weights[i] * self.periodic_predict(x, conv, linear, dropout)
#             for i, (conv, linear, dropout) in enumerate(zip(self.convs, self.linears, self.dropouts))
#         ) / self.weights.sum()

#         # 直接预测部分
#         y_direct = self.direct_predict(x, batch_size)

#         # 合并稀疏预测和直接预测部分
#         y = self.combine_predictions(y_periodic, y_direct)

#         # 调整维度并反归一化
#         return y.permute(0, 2, 1) + seq_mean

#     def normalize(self, x):
#         """归一化输入序列并调整维度 (b, s, c) -> (b, c, s)。"""
#         seq_mean = torch.mean(x, dim=1, keepdim=True)
#         return (x - seq_mean).permute(0, 2, 1), seq_mean

#     def periodic_predict(self, x, conv, linear, dropout):
#         """周期性预测：卷积 + Dropout + 线性层处理每个周期长度。"""
#         x = conv(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
#         x = x.reshape(-1, self.seq_len // linear.in_features, linear.in_features)
#         x = dropout(x)  # Dropout 应用在卷积输出之后
#         return linear(x).permute(0, 2, 1).reshape(x.size(0) // self.enc_in, self.enc_in, -1)

#     def direct_predict(self, x, batch_size):
#         """直接预测部分，避免for循环提升性能。"""
#         x = x.reshape(batch_size * self.enc_in, -1)  
#         x = self.direct_dropout(x)  # Dropout 应用在直接预测部分

#         # 使用seq_len和seq_len // 2进行切片
#         y_direct = self.direct_linear(x).reshape(batch_size, self.enc_in, self.pred_len)
#         y_direct1 = self.direct_linear1(x[:, self.seq_len // 2:]).reshape(batch_size, self.enc_in, self.pred_len)
#         y_direct2 = self.direct_linear2(x[:, self.seq_len - (self.seq_len // 3):]).reshape(batch_size, self.enc_in, self.pred_len)  # 新的预测
        
#         # 使用可学习的权重进行加权求和
#         y_combined = (y_direct * (1 - self.weight_direct1) + 
#                       y_direct1 * self.weight_direct1 * 0.5 + 
#                       y_direct2 * self.weight_direct1 * 0.5)
#         # y_combined = y_direct * (1 - self.weight_direct1) + y_direct1 * self.weight_direct1
#         return y_combined

#     def combine_predictions(self, y_periodic, y_direct):
#         """合并稀疏和直接预测结果。"""
#         weights = torch.softmax(torch.stack([self.weight_sparse, self.weight_direct]), dim=0)
#         return weights[0] * y_periodic + weights[1] * y_direct

# dropout + 单周期 + 线性
# import torch
# import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()

#         # 参数初始化
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.period_len = configs.period_len
#         self.dropout_prob = 0.4

#         # 只保留一个周期长度
#         self.periods = [self.period_len]  # 修改为只保留一个周期

#         # 初始化卷积、线性层和 Dropout
#         self.convs = nn.ModuleList([
#             nn.Conv1d(1, 1, 1 + 2 * (p // 2), stride=1, padding=p // 2, bias=False)
#             for p in self.periods
#         ])
#         self.linears = nn.ModuleList([
#             nn.Linear(self.seq_len // p, self.pred_len // p, bias=False)
#             for p in self.periods
#         ])
#         self.dropouts = nn.ModuleList([
#             nn.Dropout(self.dropout_prob) for _ in self.periods  # 每个周期对应一个 Dropout
#         ])

#         # 加权求和的可学习参数
#         self.weights = nn.Parameter(torch.ones(len(self.periods)))

#         # 直接预测模型
#         self.direct_linear = nn.Linear(self.seq_len, self.pred_len, bias=False)
#         self.direct_linear1 = nn.Linear(self.seq_len // 2, self.pred_len, bias=False)
#         self.direct_linear2 = nn.Linear(self.seq_len // 3, self.pred_len, bias=False)  # 新增的线性层
#         self.direct_dropout = nn.Dropout(self.dropout_prob)  # 用于直接预测部分

#         # 稀疏和直接预测部分的可学习权重
#         self.weight_sparse = nn.Parameter(torch.tensor(0.5))
#         self.weight_direct = nn.Parameter(torch.tensor(0.5))
#         self.weight_direct1 = nn.Parameter(torch.tensor(0.7))  # 可学习的权重

#     def forward(self, x):
#         batch_size = x.shape[0]

#         # 归一化并交换维度 (b, s, c) -> (b, c, s)
#         x, seq_mean = self.normalize(x)

#         # 周期性预测部分：卷积和线性层输出的结果求和
#         y_periodic = sum(
#             self.weights[i] * self.periodic_predict(x, conv, linear, dropout)
#             for i, (conv, linear, dropout) in enumerate(zip(self.convs, self.linears, self.dropouts))
#         ) / self.weights.sum()

#         # 直接预测部分
#         y_direct = self.direct_predict(x, batch_size)

#         # 合并稀疏预测和直接预测部分
#         y = self.combine_predictions(y_periodic, y_direct)

#         # 调整维度并反归一化
#         return y.permute(0, 2, 1) + seq_mean

#     def normalize(self, x):
#         """归一化输入序列并调整维度 (b, s, c) -> (b, c, s)。"""
#         seq_mean = torch.mean(x, dim=1, keepdim=True)
#         return (x - seq_mean).permute(0, 2, 1), seq_mean

#     def periodic_predict(self, x, conv, linear, dropout):
#         """周期性预测：卷积 + Dropout + 线性层处理每个周期长度。"""
#         x = conv(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
#         x = x.reshape(-1, self.seq_len // linear.in_features, linear.in_features)
#         x = dropout(x)  # Dropout 应用在卷积输出之后
#         return linear(x).permute(0, 2, 1).reshape(x.size(0) // self.enc_in, self.enc_in, -1)
    
#     def direct_predict(self, x, batch_size):
#         """直接预测部分，避免for循环提升性能。"""
#         x = x.reshape(batch_size * self.enc_in, -1)  
#         x = self.direct_dropout(x)  # Dropout 应用在直接预测部分

#         # 使用seq_len和seq_len // 2进行切片
#         y_direct = self.direct_linear(x).reshape(batch_size, self.enc_in, self.pred_len)
#         y_direct1 = self.direct_linear1(x[:, self.seq_len // 2:]).reshape(batch_size, self.enc_in, self.pred_len)
#         y_direct2 = self.direct_linear2(x[:, self.seq_len - (self.seq_len // 3):]).reshape(batch_size, self.enc_in, self.pred_len)  # 新的预测
        
#         # 使用可学习的权重进行加权求和
#         y_combined = (y_direct * (1 - self.weight_direct1) + 
#                       y_direct1 * self.weight_direct1 * 0.5 + 
#                       y_direct2 * self.weight_direct1 * 0.5)
#         # y_combined = y_direct * (1 - self.weight_direct1) + y_direct1 * self.weight_direct1
#         return y_combined

#     def combine_predictions(self, y_periodic, y_direct):
#         """合并稀疏和直接预测结果。"""
#         weights = torch.softmax(torch.stack([self.weight_sparse, self.weight_direct]), dim=0)
#         return weights[0] * y_periodic + weights[1] * y_direct


# 单周期 + dropout + 线性 + 可学习周期
# import torch
# import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()

#         # 参数初始化
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.dropout_prob = 0.4

#         # 将周期长度定义为可学习的参数
#         self.period_len = nn.Parameter(torch.tensor(configs.period_len, dtype=torch.float32))

#         # 只保留一个周期长度
#         self.periods = [self.period_len]  # 这里只使用一个周期

#         # 初始化卷积、线性层和 Dropout
#         self.convs = nn.ModuleList([
#             nn.Conv1d(1, 1, 1 + 2 * (int(p) // 2), stride=1, padding=int(p) // 2, bias=False)
#             for p in self.periods
#         ])
#         self.linears = nn.ModuleList([
#             nn.Linear(self.seq_len // int(p), self.pred_len // int(p), bias=False)
#             for p in self.periods
#         ])
#         self.dropouts = nn.ModuleList([
#             nn.Dropout(self.dropout_prob) for _ in self.periods  # 每个周期对应一个 Dropout
#         ])

#         # 加权求和的可学习参数
#         self.weights = nn.Parameter(torch.ones(len(self.periods)))

#         # 直接预测模型
#         self.direct_linear = nn.Linear(self.seq_len, self.pred_len, bias=False)
#         self.direct_dropout = nn.Dropout(self.dropout_prob)  # 用于直接预测部分

#         # 稀疏和直接预测部分的可学习权重
#         self.weight_sparse = nn.Parameter(torch.tensor(0.3))
#         self.weight_direct = nn.Parameter(torch.tensor(0.7))

#     def forward(self, x):
#         batch_size = x.shape[0]

#         # 归一化并交换维度 (b, s, c) -> (b, c, s)
#         x, seq_mean = self.normalize(x)

#         # 周期性预测部分：卷积和线性层输出的结果求和
#         y_periodic = sum(
#             self.weights[i] * self.periodic_predict(x, conv, linear, dropout)
#             for i, (conv, linear, dropout) in enumerate(zip(self.convs, self.linears, self.dropouts))
#         ) / self.weights.sum()

#         # 直接预测部分
#         y_direct = self.direct_predict(x, batch_size)

#         # 合并稀疏预测和直接预测部分
#         y = self.combine_predictions(y_periodic, y_direct)

#         # 调整维度并反归一化
#         return y.permute(0, 2, 1) + seq_mean

#     def normalize(self, x):
#         """归一化输入序列并调整维度 (b, s, c) -> (b, c, s)。"""
#         seq_mean = torch.mean(x, dim=1, keepdim=True)
#         return (x - seq_mean).permute(0, 2, 1), seq_mean

#     def periodic_predict(self, x, conv, linear, dropout):
#         """周期性预测：卷积 + Dropout + 线性层处理每个周期长度。"""
#         x = conv(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
#         x = x.reshape(-1, self.seq_len // linear.in_features, linear.in_features)
#         x = dropout(x)  # Dropout 应用在卷积输出之后
#         return linear(x).permute(0, 2, 1).reshape(x.size(0) // self.enc_in, self.enc_in, -1)

#     def direct_predict(self, x, batch_size):
#         """直接预测部分，避免for循环提升性能。"""
#         x = x.reshape(batch_size * self.enc_in, -1)  
#         x = self.direct_dropout(x)  # Dropout 应用在直接预测部分
#         y_direct = self.direct_linear(x).reshape(batch_size, self.enc_in, self.pred_len)
#         return y_direct

#     def combine_predictions(self, y_periodic, y_direct):
#         """合并稀疏和直接预测结果。"""
#         weights = torch.softmax(torch.stack([self.weight_sparse, self.weight_direct]), dim=0)
#         return weights[0] * y_periodic + weights[1] * y_direct





# import torch
# import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()

#         # get parameters
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.period_len = configs.period_len
#         self.d_model = configs.d_model
#         self.model_type = configs.model_type
#         assert self.model_type in ['linear', 'mlp']

#         self.seg_num_x = self.seq_len // self.period_len
#         self.seg_num_y = self.pred_len // self.period_len

#         self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
#                                 stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

#         if self.model_type == 'linear':
#             self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
#         elif self.model_type == 'mlp':
#             self.mlp = nn.Sequential(
#                 nn.Linear(self.seg_num_x, self.d_model),
#                 nn.ReLU(),
#                 nn.Linear(self.d_model, self.seg_num_y)
#             )


#     def forward(self, x):
#         batch_size = x.shape[0]
#         # normalization and permute     b,s,c -> b,c,s
#         seq_mean = torch.mean(x, dim=1).unsqueeze(1)
#         x = (x - seq_mean).permute(0, 2, 1)

#         # 1D convolution aggregation
#         x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

#         # downsampling: b,c,s -> bc,n,w -> bc,w,n
#         x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

#         # sparse forecasting
#         if self.model_type == 'linear':
#             y = self.linear(x)  # bc,w,m
#         elif self.model_type == 'mlp':
#             y = self.mlp(x)

#         # upsampling: bc,w,m -> bc,m,w -> b,c,s
#         y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

#         # permute and denorm
#         y = y.permute(0, 2, 1) + seq_mean

#         return y


import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # 参数初始化
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.dropout_prob = configs.dropout_ratio
        self.model_type = configs.model_type 
        self.d_model = configs.d_model
        # 三个不同周期长度
        self.periods = [self.period_len, self.period_len // 2, self.period_len // 4]

        # 初始化卷积、线性层/MLP和Dropout
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, 1 + 2 * (p // 2), stride=1, padding=p // 2, bias=False)
            for p in self.periods
        ])
        
        if self.model_type == 'linear':
            self.linears = nn.ModuleList([
                nn.Linear(self.seq_len // p, self.pred_len // p, bias=False)
                for p in self.periods
            ])
        elif self.model_type == 'mlp':
            self.linears = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.seq_len // p, self.d_model),  # 中间层
                    nn.ReLU(),
                    nn.Linear(self.d_model, self.pred_len // p)
                )
                for p in self.periods
            ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(self.dropout_prob) for _ in self.periods  # 每个周期对应一个 Dropout
        ])

        # 加权求和的可学习参数
        self.weights = nn.Parameter(torch.ones(len(self.periods)))

        # 直接预测模型
        if self.model_type == 'linear':
            self.direct_linear = nn.Linear(self.seq_len, self.pred_len, bias=False)
            self.direct_linear1 = nn.Linear(self.seq_len // 2, self.pred_len, bias=False)
            self.direct_linear2 = nn.Linear(self.seq_len // 3, self.pred_len, bias=False)
        elif self.model_type == 'mlp':
            self.direct_linear = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),  # 中间层
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )
            self.direct_linear1 = nn.Sequential(
                nn.Linear(self.seq_len // 2, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )
            self.direct_linear2 = nn.Sequential(
                nn.Linear(self.seq_len // 3, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

        self.direct_dropout = nn.Dropout(self.dropout_prob)  # 用于直接预测部分

        # 稀疏和直接预测部分的可学习权重
        self.weight_sparse = nn.Parameter(torch.tensor(0.3))
        self.weight_direct = nn.Parameter(torch.tensor(0.7))
        self.weight_direct1 = nn.Parameter(torch.tensor(0.7))  # 可学习的权重

    def forward(self, x):
        batch_size = x.shape[0]

        # 归一化并交换维度 (b, s, c) -> (b, c, s)
        x, seq_mean = self.normalize(x)

        # 周期性预测部分：卷积和线性层/MLP输出的结果求和
        y_periodic = sum(
            self.weights[i] * self.periodic_predict(x, conv, linear, dropout)
            for i, (conv, linear, dropout) in enumerate(zip(self.convs, self.linears, self.dropouts))
        ) / self.weights.sum()

        # 直接预测部分
        y_direct = self.direct_predict(x, batch_size)

        # 合并稀疏预测和直接预测部分
        y = self.combine_predictions(y_periodic, y_direct)

        # 调整维度并反归一化
        return y.permute(0, 2, 1) + seq_mean

    def normalize(self, x):
        """归一化输入序列并调整维度 (b, s, c) -> (b, c, s)。"""
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        return (x - seq_mean).permute(0, 2, 1), seq_mean

    def periodic_predict(self, x, conv, linear, dropout):
        """周期性预测：卷积 + Dropout + 线性层/MLP处理每个周期长度。"""
        x = conv(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
        x = x.reshape(-1, self.seq_len // linear[0].in_features, linear[0].in_features)  # 获取输入特征数
        x = dropout(x)  # Dropout 应用在卷积输出之后
        return linear(x).permute(0, 2, 1).reshape(x.size(0) // self.enc_in, self.enc_in, -1)

    def direct_predict(self, x, batch_size):
        """直接预测部分，避免for循环提升性能。"""
        x = x.reshape(batch_size * self.enc_in, -1)  
        x = self.direct_dropout(x)  # Dropout 应用在直接预测部分

        # 使用seq_len和seq_len // 2进行切片
        y_direct = self.direct_linear(x).reshape(batch_size, self.enc_in, self.pred_len)
        y_direct1 = self.direct_linear1(x[:, self.seq_len // 2:]).reshape(batch_size, self.enc_in, self.pred_len)
        y_direct2 = self.direct_linear2(x[:, self.seq_len - (self.seq_len // 3):]).reshape(batch_size, self.enc_in, self.pred_len)  # 新的预测
        
        # 使用可学习的权重进行加权求和
        y_combined = (y_direct * (1 - self.weight_direct1) + 
                      y_direct1 * self.weight_direct1 * 0.5 + 
                      y_direct2 * self.weight_direct1 * 0.5)
        return y_combined

    def combine_predictions(self, y_periodic, y_direct):
        """合并稀疏和直接预测结果。"""
        weights = torch.softmax(torch.stack([self.weight_sparse, self.weight_direct]), dim=0)
        return weights[0] * y_periodic + weights[1] * y_direct
