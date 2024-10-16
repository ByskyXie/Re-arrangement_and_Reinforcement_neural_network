from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_Custom4RE
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'Abilene': Dataset_Custom4RE,
    'GEANT': Dataset_Custom4RE,
    'TaxiBJ': Dataset_Custom4RE,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last, collate_fn=custom_collate_fn)
    return data_set, data_loader


import torch
def custom_collate_fn(batch):
    seq_x_batch, seq_y_batch, seq_x_mark_batch, seq_y_mark_batch = [], [], [], []

    for idx, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(batch):
        seq_x_batch.append(torch.as_tensor(seq_x))
        seq_y_batch.append(torch.as_tensor(seq_y))
        seq_x_mark_batch.append(torch.as_tensor(seq_x_mark))
        seq_y_mark_batch.append(torch.as_tensor(seq_y_mark))

    return torch.stack(seq_x_batch), torch.stack(seq_y_batch), torch.stack(seq_x_mark_batch), torch.stack(
        seq_y_mark_batch)
