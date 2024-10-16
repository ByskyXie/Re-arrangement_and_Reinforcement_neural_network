import torch
import torch.nn as nn
import random
import torchcde

class RCNN(nn.Module):

    def __init__(self, input_matrix_num, predict_matrix_num, in_channels, matrix_row, matrix_column) -> None:
        super(RCNN, self).__init__()
        self.input_matrix_num = input_matrix_num
        self.predict_matrix_num = predict_matrix_num
        self.in_channels = in_channels
        self.matrix_row = matrix_row
        self.matrix_column = matrix_column
        self.NUM_FLOWs = matrix_row*matrix_column

        self.INPUT_UNITS = self.NUM_FLOWs * 3
        self.HIDDEN_UNITS = self.NUM_FLOWs
        self.HIDDEN_UNITS1 = 256
        self.HIDDEN_UNITS2 = self.NUM_FLOWs
        self.VECTOR_SIZE = self.NUM_FLOWs * 2

        # self.HIDDEN_UNITS = self.NUM_FLOWs // 8
        # self.HIDDEN_UNITS1 = self.NUM_FLOWs // 2
        # self.HIDDEN_UNITS2 = self.NUM_FLOWs
        # self.VECTOR_SIZE = self.NUM_FLOWs // 2
        # self.INPUT_UNITS = self.NUM_FLOWs + self.VECTOR_SIZE

        self.arranger = Arranger(matrix_row, matrix_column)
        self.MultiCNN = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.ReLU(inplace=True),
        )
        self.flat = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.ReLU(inplace=True),
        )
        self.lin = nn.Linear(self.NUM_FLOWs*64, 1024)
        self.drop = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(1024, self.VECTOR_SIZE)

        self.LSTM1 = torch.nn.LSTM(self.INPUT_UNITS, self.HIDDEN_UNITS, batch_first=True)
        self.LSTM2 = torch.nn.LSTM(self.HIDDEN_UNITS, self.HIDDEN_UNITS1, batch_first=True)
        self.LSTM3 = torch.nn.LSTM(self.HIDDEN_UNITS1, self.HIDDEN_UNITS2, batch_first=True)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.2)


    def forward(self, x):
        out = x

        # # TODO: [
        out = self.arranger(out)

        out=self.MultiCNN(out)
        out=torch.transpose(out, 1, 2)
        out=self.flat(out)

        out=self.lin(out)
        out=self.drop(out)
        out=self.lin2(out)
        shape = out.shape
        out2 = x.view(shape[0], shape[1], -1)
        out = torch.cat([out, out2], dim=-1)

        # LSTM
        out, (hn, cn) = self.LSTM1(out)
        out = self.drop2(out)
        out, (hn, cn) = self.LSTM2(out)
        out = self.drop3(out)
        out, (hn, cn) = self.LSTM3(out)

        pred = out[:,-1,:].view(-1, self.matrix_row, self.matrix_column)

        # rearrange part ]
        inverse_column, inverse_row = self.arranger.relin0.get_inverse_matrix()
        pred = torch.matmul(pred, inverse_column)
        pred = torch.matmul(inverse_row, pred)

        return pred


class Arranger(nn.Module):
    def __init__(self, matrix_row, matrix_column):
        super(Arranger, self).__init__()
        self.matrix_row = matrix_row
        self.matrix_column = matrix_column
        self.relin0 = RearrangeLinearNoConstraint(self.matrix_row, self.matrix_column)

    def forward(self, x):
        x = self.relin0(x)
        return x

    def constraint_loss(self):
        return self.relin0.constraint_loss()



class RearrangeLinearNoConstraint(torch.nn.Module):
    __constants__ = ['matrix_row', 'matrix_column']
    matrix_row: int
    matrix_column: int
    weight_row: torch.Tensor
    weight_column: torch.Tensor

    class Binarize(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.clamp(0, 1).round()

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input > 1] = 0  # 类似STE直通估计器解决导数几乎处处为 0 所产生梯度问题，以及防止参数超出[0,1]
            grad_input[input < 0] = 0  # 类似STE直通估计器解决导数几乎处处为 0 所产生梯度问题

            return grad_input

    binarize = Binarize.apply  # 别名

    def __init__(self, matrix_row: int, matrix_column: int) -> None:
        super(RearrangeLinearNoConstraint, self).__init__()
        self.matrix_row = matrix_row
        self.matrix_column = matrix_column
        self.weight_column = torch.nn.Parameter(torch.eye(matrix_column))
        self.column_rearrange_matrix = None

        self.weight_row = torch.nn.Parameter(torch.eye(matrix_row))
        self.row_rearrange_matrix = None

        # self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.weight_column, a=0, b=1)
        torch.nn.init.uniform_(self.weight_row, a=0, b=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.column_rearrange_matrix = (self.weight_column)
        self.row_rearrange_matrix = (self.weight_row)

        out = torch.matmul(input, self.column_rearrange_matrix)  # rearrange column
        out = torch.matmul(self.row_rearrange_matrix, out)  # rearrange row
        return out

    def get_inverse_matrix(self):
        inverse_column = self.column_rearrange_matrix.inverse()
        inverse_row = self.row_rearrange_matrix.inverse()
        return [inverse_column, inverse_row]

    def print_column_order(self):
        for i in range(len(self.column_rearrange_matrix)):
            print(f'Column{i}=', end='{')
            for j in range(len(self.column_rearrange_matrix[i])):
                if self.column_rearrange_matrix[i][j] > 1e-2:
                    print(f'{j},', end='')
            print('}')

    def print_row_order(self):
        for i in range(len(self.row_rearrange_matrix[0])):
            print(f'Row{i}=', end='{')
            for j in range(len(self.row_rearrange_matrix)):
                if self.row_rearrange_matrix[j][i] > 1e-2:
                    print(f'{j},', end='')
            print('}')




