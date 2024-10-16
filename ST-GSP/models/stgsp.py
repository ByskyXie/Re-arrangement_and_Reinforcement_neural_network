import torch
import torch.nn as nn
from models.layer.sfe import Conv1ResUnitsConv2
from models.ModelConfiguration import ModelConfigurationTaxiBJ, ModelConfigurationAbilene\
    , ModelConfigurationGEANT, ModelConfigurationFirework#, ModelConfigurationBikeNYC


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
        self.column_rearrange_matrix = (self.weight_column)

        self.weight_row = torch.nn.Parameter(torch.eye(matrix_row))
        self.row_rearrange_matrix = (self.weight_row)

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

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print("The training Arranger was successfully loaded.")


class STGSP(nn.Module):
    def __init__(self, data_conf):
        super().__init__()

        # config
        self.dconf = data_conf
        if self.dconf.name == 'BikeNYC':
            print('There is no file: ModelConfigurationBikeNYC')
            #self.mconf = ModelConfigurationBikeNYC()
        elif self.dconf.name == 'TaxiBJ':
            self.mconf = ModelConfigurationTaxiBJ()
        elif self.dconf.name == 'Abilene':
            self.mconf = ModelConfigurationAbilene()
        elif self.dconf.name == 'GEANT':
            self.mconf = ModelConfigurationGEANT()
        elif self.dconf.name == 'Firework':
            self.mconf = ModelConfigurationFirework()
        else:
            raise ValueError('The data set does not exist')
        self.mconf.show()

        self.arranger = Arranger(data_conf.dim_h, data_conf.dim_w)
        self.resnns = nn.ModuleList()
        for i in range(self.dconf.len_seq):
            resnn = Conv1ResUnitsConv2(self.dconf, self.mconf)
            self.resnns.append(resnn)

        self.extnn_inter = nn.Sequential(
            nn.Linear(self.dconf.ext_dim, self.mconf.inter_extnn_inter_channels),
            nn.Dropout(self.mconf.inter_extnn_dropout),
            nn.SELU(),
            nn.Linear(self.mconf.inter_extnn_inter_channels, self.mconf.res_nbfilter*self.dconf.dim_h*self.dconf.dim_w),
            nn.SELU()
        )

        self.pre_token = nn.Parameter(torch.randn(1, self.mconf.transformer_dmodel))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.mconf.transformer_dmodel, nhead=self.mconf.transformer_nhead, dropout=self.mconf.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.mconf.transformer_nlayers)
        
        self.FC = nn.Linear(in_features=self.mconf.transformer_dmodel*(self.dconf.len_seq+1), out_features=self.dconf.dim_flow*self.dconf.dim_h*self.dconf.dim_w)

        self.extnn_last = nn.Sequential(
            nn.Linear(self.dconf.ext_dim, self.mconf.last_extnn_inter_channels),
            nn.SELU(),
            nn.Linear(self.mconf.last_extnn_inter_channels, self.dconf.dim_flow*self.dconf.dim_h*self.dconf.dim_w),
            nn.SELU()
        )

        self.model_name = str(type(self).__name__)

    def forward(self, X, X_ext, Y_ext):  # torch.Size([32, 10, 32, 32]) torch.Size([32, 5, 77]) torch.Size([32, 77])
        # X = self.arranger(X)

        inputs = torch.split(X, self.dconf.dim_flow, 1)
        ext_outputs = self.extnn_inter(X_ext)
        E_ems = torch.split(ext_outputs, 1, 1)
        
        transformer_inputs = []
        pre_tokens = self.pre_token.repeat(inputs[0].shape[0], 1)
        transformer_inputs.append(pre_tokens)
        for i in range(self.dconf.len_seq):
            X_em = self.resnns[i](inputs[i], E_ems[i].squeeze(1))
            transformer_inputs.append(X_em)

        transformer_inputs = torch.stack(transformer_inputs, 0)
        transformer_outputs = self.transformer_encoder(transformer_inputs)
        transformer_outputs = transformer_outputs.transpose(0, 1)
        out = torch.flatten(transformer_outputs, 1)
        out = self.FC(out)

        main_out = out.reshape(-1, self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w)
        ext_out = self.extnn_last(Y_ext)
        ext_out = ext_out.reshape(-1, self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w)
        main_out = main_out + ext_out

        # rearrange part ]
        # inverse_column, inverse_row = self.arranger.get_inverse_matrix()
        # main_out = torch.matmul(main_out, inverse_column)
        # main_out = torch.matmul(inverse_row, main_out)

        main_out = torch.tanh(main_out)
        return main_out

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print("The training model was successfully loaded.")
