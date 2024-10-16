from typing import Optional

import torch
import torch.nn as nn
from stresnet.models import ResUnit



class Arranger(torch.nn.Module):
    __constants__ = ['matrix_row', 'matrix_column']
    matrix_row: int
    matrix_column: int
    weight_row: torch.Tensor
    weight_column: torch.Tensor

    def __init__(self, matrix_row: int, matrix_column: int) -> None:
        super(Arranger, self).__init__()
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




class STResNet(nn.Module):
    def __init__(
        self,
        len_closeness: int,
        len_period: int,
        len_trend: int,
        external_dim: Optional[int],
        nb_flow: int,
        map_height: int,
        map_width: int,
        nb_residual_unit: int,
    ) -> None:
        super().__init__()
        self.external_dim = external_dim
        self.map_height = map_height
        self.map_width = map_width
        self.nb_flow = nb_flow
        self.nb_residual_unit = nb_residual_unit

        # models
        self.arranger = Arranger(map_height, map_width)  # R&E
        self.c_net = self._create_timenet(len_closeness, nb_flow)
        self.p_net = self._create_timenet(len_period, nb_flow)
        self.t_net = self._create_timenet(len_trend, nb_flow)
        if self.external_dim:
            # in/out flows * (len_closeness + len_period + len_trend)
            nb_total_flows = self.nb_flow * (len_closeness + len_period + len_trend)
            self.e_net = self._create_extnet(
                self.external_dim, nb_total_flows=nb_total_flows
            )

        # for fusion
        self.W_c = nn.parameter.Parameter(
            torch.randn(self.nb_flow, self.map_width, self.map_height),
            requires_grad=True,
        )
        self.W_p = nn.parameter.Parameter(
            torch.randn(self.nb_flow, self.map_width, self.map_height),
            requires_grad=True,
        )
        self.W_t = nn.parameter.Parameter(
            torch.randn(self.nb_flow, self.map_width, self.map_height),
            requires_grad=True,
        )

    def _create_extnet(self, ext_dim: int, nb_total_flows: int) -> nn.Sequential:
        ext_net = nn.Sequential(
            nn.Linear(ext_dim, nb_total_flows),
            nn.ReLU(inplace=True),
            # flatten in/out flow * grid_height * grid_width
            nn.Linear(nb_total_flows, self.nb_flow * self.map_height * self.map_width),
        )
        return ext_net

    def _create_timenet(self, length: int, nb_flow: int) -> nn.Sequential:
        time_net = nn.Sequential()
        time_net.add_module(
            "Conv1",
            nn.Conv2d(
                in_channels=(length * self.nb_flow),
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

        for i in range(self.nb_residual_unit):
            time_net.add_module(
                f"ResUnit{i + 1}", ResUnit(in_channels=64, out_channels=64)
            )

        time_net.add_module(
            "Conv2",
            nn.Conv2d(
                in_channels=64, out_channels=nb_flow, kernel_size=3, stride=1, padding="same"
            ),
        )
        return time_net

    def forward(
        self,
        xc: torch.Tensor,
        xp: torch.Tensor,
        xt: torch.Tensor,
        ext: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # R&E
        channel_c, channel_p = xc.shape[1], xp.shape[1]
        tmp = self.arranger(torch.concat([xc, xp, xt], dim=1))
        xc, xp, xt = tmp[:, :channel_c], tmp[:, channel_c:channel_c+channel_p], tmp[:, channel_c+channel_p:]

        c_out = self.c_net(xc)
        p_out = self.p_net(xp)
        t_out = self.t_net(xt)

        if self.external_dim:
            e_out = self.e_net(ext).view(
                -1, self.nb_flow, self.map_width, self.map_height
            )
            # fusion with ext data
            res = self.W_c.unsqueeze(0) * c_out
            res += self.W_p.unsqueeze(0) * p_out
            res += self.W_t.unsqueeze(0) * t_out
            res += e_out
        else:
            res = self.W_c.unsqueeze(0) * c_out
            res += self.W_p.unsqueeze(0) * p_out
            res += self.W_t.unsqueeze(0) * t_out

        # rearrange part ]
        inverse_column, inverse_row = self.arranger.get_inverse_matrix()
        res = torch.matmul(res, inverse_column)
        res = torch.matmul(inverse_row, res)

        return torch.tanh(res)
