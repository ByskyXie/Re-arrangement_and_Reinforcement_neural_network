import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
configs.batch_size_test = 16
configs.batch_size = 8
configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 250
configs.num_epochs = 200
configs.early_stopping = False
configs.load_RE_weight = True
configs.RE_trainable = False
configs.patience = 4  # early stopping Max Waitting.
configs.gradient_clipping = True
configs.clipping_threshold = 1.

# data related
configs.input_dim = 1  # i.e. channels num
configs.output_dim = 1  # i.e. channels num
configs.input_length = 4
configs.output_length = 1
configs.input_gap = 1
configs.pred_shift = 0
configs.matrix_row = 32
configs.matrix_column = 32

# model related
configs.kernel_size = (3, 3)
configs.bias = True
configs.hidden_dim = (64, 64, 64, 64)
configs.d_attn = 32
configs.ssr_decay_rate = 0.8e-4



# import from NMMF-Stream
configs.input_matrix_num = configs.input_length
configs.predict_matrix_num = configs.output_length
configs.in_channels = configs.input_dim
configs.sampling_rate1 = 1  # sampling rate of matrix
configs.sampling_rate2 = 1  # sampling rate of time interval


configs_forBJ = configs
