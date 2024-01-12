import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, output_dim=29):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.GRU(input_dim, output_dim, batch_first=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        y_pred = self.softmax(rnn_out)
        return y_pred

class RNNModel(nn.Module):
    def __init__(self, input_dim, units, activation, output_dim=29):
        super(RNNModel, self).__init__()
        self.rnn = nn.GRU(input_dim, units, activation=activation, batch_first=True)
        self.bn = nn.BatchNorm1d(units)
        self.time_dense = nn.Linear(units, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        bn_out = self.bn(rnn_out)
        time_dense_out = self.time_dense(bn_out)
        y_pred = self.softmax(time_dense_out)
        return y_pred

class CNNRNNModel(nn.Module):
    def __init__(self, input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
        super(CNNRNNModel, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, filters, kernel_size, stride=conv_stride, padding=conv_border_mode)
        self.bn_cnn = nn.BatchNorm1d(filters)
        self.rnn = nn.RNN(filters, units, batch_first=True)
        self.bn_rnn = nn.BatchNorm1d(units)
        self.time_dense = nn.Linear(units, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        conv_out = F.relu(self.conv1d(x))
        bn_cnn_out = self.bn_cnn(conv_out)
        rnn_out, _ = self.rnn(bn_cnn_out)
        bn_rnn_out = self.bn_rnn(rnn_out)
        time_dense_out = self.time_dense(bn_rnn_out)
        y_pred = self.softmax(time_dense_out)
        return y_pred

class DeepRNNModel(nn.Module):
    def __init__(self, input_dim, units, recur_layers, output_dim=29):
        super(DeepRNNModel, self).__init__()
        # Add your recurrent layers with batch normalization here

    def forward(self, x):
        # Implement the forward pass with recurrent layers and batch normalization
        pass

class BidirectionalRNNModel(nn.Module):
    def __init__(self, input_dim, units, output_dim=29):
        super(BidirectionalRNNModel, self).__init__()
        # Add your bidirectional recurrent layer here

    def forward(self, x):
        # Implement the forward pass with bidirectional recurrent layer
        pass

class FinalModel(nn.Module):
    def __init__(self, input_dim, units, output_dim=29):
        super(FinalModel, self).__init__()
        # Specify the layers in your network here

    def forward(self, x):
        # Implement the forward pass for the final model
        pass
