import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('CUDA available')
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('CUDA not available, using CPU')

# 读取数据
dataframe = pd.read_csv('btc_data.csv', parse_dates=[0], index_col=0, usecols=[0, 5])
data = dataframe.values
# 归一化数据
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data)


# 创建时间序列数据
def create_sequences(data, seq_length, prediction_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - prediction_length + 1):
        x = data[i:i + seq_length]
        y = data[i + seq_length:i + seq_length + prediction_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 10
prediction_length = 5
x, y = create_sequences(data_normalized, seq_length, prediction_length)

# 转换为PyTorch张量并调整形状
x_tensor = torch.FloatTensor(x).view(-1, seq_length, 1)
y_tensor = torch.FloatTensor(y).view(-1, prediction_length)

# 创建数据集和数据加载器
dataset = TensorDataset(x_tensor, y_tensor)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=prediction_length):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        lstm_out_last = lstm_out[:, -1, :]
        predictions = self.linear(lstm_out_last)
        return predictions


# 实例化模型，定义损失函数和优化器
model = LSTMModel().to(device)
loss_function = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 300
for epoch in range(epochs):
    for seq, labels in train_loader:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, seq.size(0), model.hidden_layer_size).to(device),
                             torch.zeros(1, seq.size(0), model.hidden_layer_size).to(device))
        seq = seq.to(device)
        labels = labels.to(device)
        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if epoch % 25 == 0:
        print(f'Epoch {epoch} loss: {single_loss.item()}')

save_dir = "./modle_save"
torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
print(f'Epoch {epochs} loss: {single_loss.item()}')

# 预测
model.eval()
test_predictions = []
test_actuals = []
for seq, labels in test_loader:
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, seq.size(0), model.hidden_layer_size).to(device),
                             torch.zeros(1, seq.size(0), model.hidden_layer_size).to(device))
        seq = seq.to(device)
        labels = labels.to(device)
        y_pred = model(seq)
        test_predictions.extend(y_pred.cpu().numpy())
        test_actuals.extend(labels.cpu().numpy())

# 反归一化数据
test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, prediction_length)).reshape(-1)
test_actuals = scaler.inverse_transform(np.array(test_actuals).reshape(-1, prediction_length)).reshape(-1)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.plot(test_actuals, label='Actual Data')
plt.plot(test_predictions, label='Predicted Data')
plt.legend()
plt.show()

# 预测未来5天
model.eval()
future_predictions = []
with torch.no_grad():
    input_seq = torch.FloatTensor(data_normalized[-seq_length:]).view(1, seq_length, 1).to(device)
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                         torch.zeros(1, 1, model.hidden_layer_size).to(device))
    future_predictions = model(input_seq).cpu().numpy()

# 反归一化未来5天的预测数据
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).reshape(-1)

print(f"未来5天的预测值: {future_predictions}")
