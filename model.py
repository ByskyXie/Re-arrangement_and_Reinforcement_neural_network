import torch
import torch.nn as nn
import random


class MapInterpolationModel(nn.Module):

    def get_predictor(self):
        self.predictor = nn.Sequential(
            nn.Conv3d(self.in_channels, self.predict_matrix_num
                      , kernel_size=(self.input_matrix_num, 3, 3), padding=[0, 1, 1], bias=False),
            nn.Conv3d(self.predict_matrix_num, self.predict_matrix_num
                      , kernel_size=(1, 3, 3), padding=[0, 1, 1], bias=False),
            nn.ReLU()
        )

    # initialization parameters: Encoder,Decoder,predictor
    def __init__(self, input_matrix_num, predict_matrix_num, in_channels, blocks_num, matrix_row, matrix_column):
        super(MapInterpolationModel, self).__init__()
        assert input_matrix_num % 2 == 1
        self.input_matrix_num = input_matrix_num
        self.predict_matrix_num = predict_matrix_num
        self.in_channels = in_channels

        self.arranger = Arranger(matrix_row, matrix_column)
        self.get_predictor()

        # 4 layers CNN for net
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=(1, 0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=(1, 0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=1, padding=(1, 0, 0),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=1, padding=(1, 1, 1),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=1, padding=(1, 0, 0),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=1, padding=(1, 1, 1),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        # 4 layers CNN for video
        # self.encoder = nn.Sequential(
        #     nn.Conv3d(in_channels, in_channels*2, kernel_size=(3,7,7), padding=(1, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels*2, in_channels*4, kernel_size=(3,7,7), padding=(1, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels*4, in_channels*4, kernel_size=(3,7,7), padding=(1, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels*4, in_channels*8, kernel_size=(3,7,7), padding=(1, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose3d(in_channels*8, in_channels*4, kernel_size=(3,7,7), stride=1, padding=(1, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose3d(in_channels*4, in_channels*4, kernel_size=(3,7,7), stride=1, padding=(1, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose3d(in_channels*4, in_channels*2, kernel_size=(3,7,7), stride=1, padding=(1, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose3d(in_channels*2, in_channels, kernel_size=(3,7,7), stride=1, padding=(1, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, x, time_seq):
        # [ rearrange part
        x = self.arranger(x)

        # TIM part
        res = self.encoder(x)
        out = self.decoder(res)
        # 最后predict
        out = self.predictor(out)
        # print(out.shape)

        # rearrange part2 ] restore original row-column order
        inverse_column, inverse_row = self.arranger.relin0.get_inverse_matrix()
        out = torch.matmul(out, inverse_column)
        out = torch.matmul(inverse_row, out)
        return out




class PredictionModel(nn.Module):

    def get_predictor(self):
        self.predictor = nn.Sequential(
            nn.Conv3d(self.in_channels, self.predict_matrix_num
                      , kernel_size=(1, 3, 3), padding=[0, 1, 1], bias=False),
            nn.Conv3d(self.predict_matrix_num, self.predict_matrix_num
                      , kernel_size=(1, 3, 3), padding=[0, 1, 1], bias=False),
            nn.ReLU()
        )

    # initialization：Encoder,Decoder,predictor
    def __init__(self, input_matrix_num, predict_matrix_num, in_channels, blocks_num, matrix_row, matrix_column):
        super(PredictionModel, self).__init__()
        self.input_matrix_num = input_matrix_num
        self.predict_matrix_num = predict_matrix_num
        self.in_channels = in_channels

        self.arranger = Arranger(matrix_row, matrix_column)
        self.get_predictor()

        # 4 layers CNN for net
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, in_channels*2, kernel_size=(3, 5, 5), padding=(0, 0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels*2, in_channels*2, kernel_size=(3, 5, 5), padding=(0, 2, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels*2, in_channels*4, kernel_size=(3, 5, 5), padding=(0, 0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels*4, in_channels*8, kernel_size=(2, 5, 5), padding=(0, 2, 2), bias=False),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels*8, in_channels*4, kernel_size=(1, 5, 5), stride=1, padding=(0, 0, 0),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels*4, in_channels*2, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels*2, in_channels*2, kernel_size=(1, 5, 5), stride=1, padding=(0, 0, 0),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels*2, in_channels, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        # 4 layers CNN for video
        # self.encoder = nn.Sequential(
        #     nn.Conv3d(in_channels, in_channels * 2, kernel_size=(3, 7, 7), padding=(0, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        #     nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=(3, 7, 7), padding=(0, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels * 4, in_channels * 8, kernel_size=(3, 7, 7), padding=(0, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        #     nn.Conv3d(in_channels * 8, in_channels * 16, kernel_size=(2, 7, 7), padding=(0, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose3d(in_channels * 16, in_channels * 8, kernel_size=(1, 7, 7), stride=1, padding=(0, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     # nn.UpsamplingNearest2d(scale_factor=(1,2,2)),
        #     nn.ConvTranspose3d(in_channels * 8, in_channels * 4, kernel_size=(1, 7, 7), stride=1, padding=(0, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose3d(in_channels * 4, in_channels * 2, kernel_size=(1, 7, 7), stride=1, padding=(0, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingNearest2d(scale_factor=(1,2,2)),
        #     nn.ConvTranspose3d(in_channels * 2, in_channels, kernel_size=(1, 7, 7), stride=1, padding=(0, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     # nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 0, 0),
        #     #                    output_padding=0, bias=True),
        #     # nn.ReLU(inplace=True),
        # )

    def forward(self, x, time_seq):
        # [ rearrange part
        x = self.arranger(x)

        # task part
        out = self.encoder(x)  # torch.Size([8, 8, 8, 120, 152]) or torch.Size([8, 1, 8, 19, 19])
        # decode
        out = self.decoder(out)
        # print(out.shape)
        # predict
        out = self.predictor(out)
        # print(out.shape)

        # rearrange part2 ] restore original row-column order
        inverse_column, inverse_row = self.arranger.relin0.get_inverse_matrix()
        out = torch.matmul(out, inverse_column)
        out = torch.matmul(inverse_row, out)
        return out



class FillModel(nn.Module):

    def get_predictor(self):
        self.predictor = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels
                      , kernel_size=(1, 3, 3), padding=[0, 1, 1], bias=False),
            nn.Conv3d(self.in_channels, self.in_channels
                      , kernel_size=(1, 3, 3), padding=[0, 1, 1], bias=False),
            nn.ReLU()
        )

    # Initialization：Encoder,Decoder,predictor
    def __init__(self, input_matrix_num, in_channels, blocks_num, matrix_row, matrix_column):
        super(FillModel, self).__init__()
        self.input_matrix_num = input_matrix_num
        self.in_channels = in_channels

        self.arranger = Arranger(matrix_row, matrix_column)
        self.get_predictor()

        # 4 layers CNN for net # [8, 1, 4, 23, 23]
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, in_channels*2, kernel_size=(3, 5, 5), padding=(0, 0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels*2, in_channels*2, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels*2, in_channels*4, kernel_size=(3, 5, 5), padding=(0, 0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels*4, in_channels*4, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels*4, in_channels*4, kernel_size=(3, 5, 5), stride=1, padding=(0, 0, 0),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels*4, in_channels*2, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels*2, in_channels*2, kernel_size=(3, 5, 5), stride=1, padding=(0, 0, 0),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels*2, in_channels, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2),
                               output_padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        # 4 layers CNN for video
        # self.encoder = nn.Sequential(
        #     nn.Conv3d(in_channels, in_channels * 2, kernel_size=(3, 7, 7), padding=(0, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        #     nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=(3, 7, 7), padding=(1, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels * 4, in_channels * 4, kernel_size=(3, 7, 7), padding=(0, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        #     nn.Conv3d(in_channels * 4, in_channels * 8, kernel_size=(3, 7, 7), padding=(1, 0, 0), bias=True),
        #     nn.ReLU(inplace=True),
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose3d(in_channels * 8, in_channels * 4, kernel_size=(3, 7, 7), stride=1, padding=(1, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingNearest2d(scale_factor=(1,2,2)),
        #     nn.ConvTranspose3d(in_channels * 4, in_channels * 4, kernel_size=(3, 7, 7), stride=1, padding=(0, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose3d(in_channels * 4, in_channels * 2, kernel_size=(3, 7, 7), stride=1, padding=(1, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingNearest2d(scale_factor=(1,2,2)),
        #     nn.ConvTranspose3d(in_channels * 2, in_channels, kernel_size=(3, 7, 7), stride=1, padding=(0, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 0, 0),
        #                        output_padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=(1, 0, 1), bias=True), # TODO:for firework
        # )

    def forward(self, x, time_seq):
        # [ rearrange part
        x = self.arranger(x)

        # task part
        out = self.encoder(x)  # torch.Size([8, 8, 8, 120, 152]) or torch.Size([8, 1, 8, 19, 19])

        # decode
        out = self.decoder(out)

        # predict
        out = self.predictor(out)
        # print(out.shape)

        # rearrange part2 ] restore original row-column order
        inverse_column, inverse_row = self.arranger.relin0.get_inverse_matrix()
        out = torch.matmul(out, inverse_column)
        out = torch.matmul(inverse_row, out)
        return out


class Arranger(nn.Module):
    def __init__(self, matrix_row, matrix_column):
        super(Arranger, self).__init__()
        self.matrix_row = matrix_row
        self.matrix_column = matrix_column
        self.relin0 = RE_neural_network(self.matrix_row, self.matrix_column)

    def forward(self, x):
        x = self.relin0(x)
        return x

    def constraint_loss(self):
        return self.relin0.constraint_loss()


class RE_neural_network(torch.nn.Module):
    __constants__ = ['matrix_row', 'matrix_column']
    matrix_row: int
    matrix_column: int
    weight_row: torch.Tensor
    weight_column: torch.Tensor

    class Binarize(torch.autograd.Function):
        #  Use for Binary Neural Network, but experiment shows model hard converge.
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.clamp(0, 1).round()

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input > 1] = 0  # defense parameter exceede limit [0,1]
            grad_input[input < 0] = 0  # defense parameter exceede limit [0,1]

            # # scan 1 element amount per column
            # binarize_matrix = input.clamp(0, 1).round()
            # for col_idx in range(1, binarize_matrix.shape[-1]):  # scan per column
            #     indices = binarize_matrix[:, :col_idx].sum(dim=-1)
            #     # [indices > 0] is the set of row numbers where there is a position of 1 on the left side of the column, such as [1,0,col_idx], there is 1 on the left side
            #     grad_input[indices > 0, col_idx] *= 10  # means that the column has been used before, and it will be punished
            return grad_input

    binarize = Binarize.apply

    def __init__(self, matrix_row: int, matrix_column: int) -> None:
        super(RE_neural_network, self).__init__()
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

    def constraint_loss(self):
        one_mat4column = torch.ones(self.matrix_column) if not self.column_rearrange_matrix.is_cuda else torch.ones(
            self.matrix_column).to(torch.device('cuda:0'))
        one_mat4row = torch.ones(self.matrix_row) if not self.column_rearrange_matrix.is_cuda else torch.ones(
            self.matrix_row).to(torch.device('cuda:0'))
        row_loss = torch.mean(
            torch.abs(torch.sum(self.binarize(self.weight_column), dim=1) - one_mat4column)) + torch.mean(
            torch.abs(torch.sum(self.binarize(self.weight_row), dim=1) - one_mat4row))  # each row should have 1
        column_loss = torch.mean(
            torch.abs(torch.sum(self.binarize(self.weight_column), dim=0) - one_mat4column)) + torch.mean(
            torch.abs(torch.sum(self.binarize(self.weight_row), dim=0) - one_mat4row))  # each column should have 1
        return row_loss + column_loss

    def get_inverse_matrix(self):
        inverse_column = self.column_rearrange_matrix.inverse()
        inverse_row = self.row_rearrange_matrix.inverse()
        return [inverse_column, inverse_row]


class RearrangeLinear2(torch.nn.Module):
    __constants__ = ['matrix_row', 'matrix_column']
    matrix_row: int
    matrix_column: int
    weight_row: torch.Tensor
    weight_column: torch.Tensor

    class Binarize(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, upper:torch.Tensor):
            assert len(upper) == 1
            ctx.save_for_backward(input)
            ctx.upper = int(upper)
            return input.clamp(0, ctx.upper).round()

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_upper = None

            grad_input = grad_output.clone()
            if len(input.shape) == 1:  # 1 means it's weight, value should clip to [0, upper].
                grad_input[input > ctx.upper] = 0  # like STE estimator to solve the gradient problem caused by the derivative being almost 0 everywhere, and to prevent the parameters from exceeding [0,1]
                grad_input[input < 0] = 0
            else:  # otherwise it's rearranger matrix
                pass
                # binarize_matrix = input.clamp(0, 1).round()
                # for col_idx in range(1, binarize_matrix.shape[-1]):  # scan per column
                #     indices = binarize_matrix[:, :col_idx].sum(dim=-1)
                #     # [indices > 0] is the set of row numbers where there is a position of 1 on the left side of the column, such as [1,0,col_idx], there is 1 on the left side
                #     grad_input[indices > 0, col_idx] *= 10  # means that the column has been used before, and it will be punished

            return grad_input, grad_upper

    binarize = Binarize.apply

    def __init__(self, matrix_row: int, matrix_column: int) -> None:
        super(RearrangeLinear2, self).__init__()
        self.matrix_row = matrix_row
        self.matrix_column = matrix_column

        self.weight_column = torch.nn.Parameter(torch.Tensor(matrix_column))
        self.identity_column = torch.nn.Parameter(torch.Tensor(
            [[i for j in range(self.matrix_column)] for i in range(self.matrix_column)]), requires_grad=False)

        self.weight_row = torch.nn.Parameter(torch.Tensor(matrix_row))
        self.identity_row = torch.nn.Parameter(torch.Tensor(
            [[j for j in range(self.matrix_row)] for i in range(self.matrix_row)]), requires_grad=False)

        self.column_rearrange_matrix = None
        self.row_rearrange_matrix = None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight_column, a=0, b=self.matrix_column)
        torch.nn.init.uniform_(self.weight_row, a=0, b=self.matrix_row)

    def __update_rearrange_matrix__(self):
        # column
        self.column_rearrange_matrix = self.identity_column - self.binarize(self.weight_column, torch.Tensor([self.matrix_column-1,]))
        self.column_rearrange_matrix = self.binarize(1-torch.abs(self.column_rearrange_matrix), torch.Tensor([1,]))
        # row
        self.row_rearrange_matrix = self.identity_row - self.binarize(self.weight_row, torch.Tensor([self.matrix_row-1,])).unsqueeze(dim=-1)
        self.row_rearrange_matrix = self.binarize(1-torch.abs(self.row_rearrange_matrix), torch.Tensor([1,]))

    def __constraint_rearrange_matrix__(self):
        # collect empty row idx of column_rearrange_matrix
        empty_idx, empty_loss = [], None
        temp = self.column_rearrange_matrix.sum(dim=-1)
        for idx in range(len(temp)):
            if temp[idx] == 0:
                empty_idx.append(idx)
        random.shuffle(empty_idx)
        # modify column weight
        for col_idx in range(1, self.column_rearrange_matrix.shape[-1]):  # scan per column
            indices = self.column_rearrange_matrix[:, :col_idx].sum(dim=-1)
            # [indices > 0] is the set of row numbers where there is a position of 1 on the left side of the column, such as [1,0,col_idx], there is 1 on the left side
            if indices[self.column_rearrange_matrix[:, col_idx] == 1] == 0:
                continue  # Indicates that the selected line (==1) has not been used before (==0)
            # print(f"weight_COLUMN[{col_idx}] from {self.weight_column[col_idx]} to {empty_idx[-1]}")
            self.weight_column.data[col_idx] = empty_idx.pop()

        # collect empty column idx of row_rearrange_matrix
        empty_idx, empty_loss = [], None
        temp = self.row_rearrange_matrix.sum(dim=-2)
        for idx in range(len(temp)):
            if temp[idx] == 0:
                empty_idx.append(idx)
        random.shuffle(empty_idx)
        # modify row weight
        for row_idx in range(1, self.row_rearrange_matrix.shape[-2]):  # scan per row
            indices = self.row_rearrange_matrix[:row_idx, :].sum(dim=-2)
            # [indices > 0] is the set of column numbers where there is a position of 1 on the top side of the row, such as [1,0,row_idx].t(), there is 1 on the left side
            if indices[self.row_rearrange_matrix[row_idx, :] == 1] == 0:
                continue  # Indicates that the selected column (==1) has not been used before (==0)
            # print(f"weight_ROW[{col_idx}] from {self.weight_row[row_idx]} to {empty_idx[-1]}")
            self.weight_row.data[row_idx] = empty_idx.pop()

        # update matrix after modified weight
        self.__update_rearrange_matrix__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.__update_rearrange_matrix__()
        self.__constraint_rearrange_matrix__()

        out = torch.matmul(input, self.column_rearrange_matrix)  # Column rearrange
        out = torch.matmul(self.row_rearrange_matrix, out)  # Row rearrange
        return out

    def constraint_loss(self):
        column_one_mat = torch.ones(self.matrix_column) if not self.column_rearrange_matrix.is_cuda else torch.ones(
            self.matrix_column).to(torch.device('cuda:0'))
        row_one_mat = torch.ones(self.matrix_row) if not self.row_rearrange_matrix.is_cuda else torch.ones(
            self.matrix_row).to(torch.device('cuda:0'))
        row_loss = torch.mean(
            torch.abs(torch.sum(self.column_rearrange_matrix, dim=1) - column_one_mat)) + torch.mean(
            torch.abs(torch.sum(self.row_rearrange_matrix, dim=1) - row_one_mat))  # each row should have 1
        column_loss = torch.mean(
            torch.abs(torch.sum(self.column_rearrange_matrix, dim=0) - column_one_mat)) + torch.mean(
            torch.abs(torch.sum(self.row_rearrange_matrix, dim=0) - row_one_mat))  # each column should have 1

        return row_loss+column_loss

    def get_inverse_matrix(self):
        inverse_column, inverse_row = None, None
        # column inverse
        if self.column_rearrange_matrix.sum(dim=-1).max() != 1 or self.column_rearrange_matrix.sum(dim=-2).max() != 1:
            inverse_column = torch.eye(self.column_rearrange_matrix.shape[-1])
            if self.column_rearrange_matrix.is_cuda:
                inverse_column = inverse_column.to(torch.device('cuda:0'))
        else:
            inverse_column = self.column_rearrange_matrix.inverse()
        # row inverse
        if self.row_rearrange_matrix.sum(dim=-1).max() != 1 or self.row_rearrange_matrix.sum(dim=-2).max() != 1:
            inverse_row = torch.eye(self.row_rearrange_matrix.shape[-1])
            if self.row_rearrange_matrix.is_cuda:
                inverse_row = inverse_row.to(torch.device('cuda:0'))
        else:
            inverse_row = self.row_rearrange_matrix.inverse()
        return [inverse_column, inverse_row]

    def print_column_order(self):
        for i in range(len(self.column_rearrange_matrix)):
            print(f'Column{i}=', end='{')
            for j in range(len(self.column_rearrange_matrix[i])):
                if self.column_rearrange_matrix[i][j] > 1e-1:
                    print(f'{j},', end='')
            print('}')

    def print_row_order(self):
        for i in range(len(self.row_rearrange_matrix[0])):
            print(f'Row{i}=', end='{')
            for j in range(len(self.row_rearrange_matrix)):
                if self.row_rearrange_matrix[j][i] > 1e-1:
                    print(f'{j},', end='')
            print('}')



if __name__ == '__main__':
    # test
    batch_size = 1
    input_matrix_num = 4
    predict_matrix_num = 3
    # model = MapInterpolationModel(input_matrix_num, predict_matrix_num, batch_size, blocks_num=3)


