import torch
import torch.nn as nn
import random
import torchcde


class MapInterpolationModel(nn.Module):

    def get_predictor(self):
        self.predictor = nn.Sequential(
            nn.Conv3d(self.in_channels, self.predict_matrix_num
                      , kernel_size=(self.input_matrix_num, 3, 3), padding=[0, 1, 1], bias=False),
            nn.Conv3d(self.predict_matrix_num, self.predict_matrix_num
                      , kernel_size=(1, 3, 3), padding=[0, 1, 1], bias=False),
            nn.ReLU()
        )

    # 初始化参数：Encoder,Decoder,predictor
    def __init__(self, input_matrix_num, predict_matrix_num, in_channels, blocks_num, matrix_row, matrix_column):
        super(MapInterpolationModel, self).__init__()
        assert input_matrix_num % 2 == 1
        self.input_matrix_num = input_matrix_num
        self.predict_matrix_num = predict_matrix_num
        self.in_channels = in_channels

        self.arranger = Arranger(matrix_row, matrix_column)
        self.encoder = Encoder(input_matrix_num, blocks_num, in_channels=in_channels)
        decoder_in_channels = self.encoder.channel_list[-1]
        self.decoder = Decoder(input_matrix_num, blocks_num, in_channels=decoder_in_channels)
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

        # rearrange part ]
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
        self.encoder = Encoder(input_matrix_num, blocks_num, in_channels=in_channels)
        decoder_in_channels = self.encoder.channel_list[-1]
        self.decoder = Decoder(input_matrix_num, blocks_num, in_channels=decoder_in_channels)
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

        # rearrange part ]
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
        self.encoder = Encoder(input_matrix_num, blocks_num, in_channels=in_channels)
        decoder_in_channels = self.encoder.channel_list[-1]
        self.decoder = Decoder(input_matrix_num, blocks_num, in_channels=decoder_in_channels)
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

        # rearrange part ]
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



if __name__ == '__main__':
    batch_size = 1
    input_matrix_num = 4
    predict_matrix_num = 3
    # model = MapInterpolationModel(input_matrix_num, predict_matrix_num, batch_size, blocks_num=3)


