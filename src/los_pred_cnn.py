import argparse
import glob
import json
import os.path
import time

from pathlib import Path

from sklearn.model_selection import KFold

import torch
import torchmetrics
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

from torchsampler import ImbalancedDatasetSampler

from utils.tsa import *
from utils.common_processed import *

import warnings
warnings.filterwarnings("ignore")

NP_RANDOM_SEED = 0
np.random.seed(NP_RANDOM_SEED)

ENCODE_ELEM = ['mean', 'var', 'encoded1', 'encoded2', 'encoded3', 'encoded4']


def cnn_clf_config(data_type: str, batch_size: int = 64):
    # model setup
    if data_type == 'raw':
        cnn_in = 38
    elif data_type == 'encoded':
        cnn_in = 228
    cnn_kernel_size = [3, 3]
    cnn_stride_size = [1, 1]
    cnn_group_size = [38, 1]
    config = {
        "n_cnn_layer": 2,
        "n_hidden_dense": [512],
        "n_hidden_cnn": [cnn_in, 380, 256],
        "cnn_kernel_size": cnn_kernel_size,
        "cnn_stride_size": cnn_stride_size,
        "cnn_group_size": cnn_group_size,
        "dropout_cnn": .0,
        "dropout_dense": .5,
        "lr": 1e-5,
        "weight_decay": 0.0,
        "batch_size": batch_size,
    }
    return config


class PatientDataset(Dataset):
    def __init__(self, data: list, los: int):
        self.data = data
        self.los = los
        self.labels = []
        for sid in self.data['sample_id'].unique():
            los = self.data.loc[self.data['sample_id']==sid, 'LOS'].iloc[-1]
            if los > self.los * 24:
                self.labels.append(1)
            else:
                self.labels.append(0)

    def __len__(self):
        return len(self.data['sample_id'].unique())

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        df_sample = self.data[self.data['sample_id']==self.data['sample_id'].unique()[idx]]

        # 1. labels
        x_label = torch.Tensor([self.labels[idx]])

        # 2. static data
        x_static = []
        # 2.1 info numeric
        info_numeric = [df_sample[col].iloc[0] for col in COL_INFO_NUMERIC]
        x_static.append(torch.tensor(info_numeric, dtype=torch.float))
        # 2.2 info text (1 hot)
        for col in COL_INFO_TEXT:
            ohk = F.one_hot(torch.tensor(df_sample[col].iloc[0].astype(int)),
                            num_classes=len(info_text_vals[col]))
            x_static.append(ohk)
        x_static = torch.cat(x_static, dim=0)

        # 3. time series data
        x_timeseries = self._get_timeseries_data(df_sample)

        return {
            'data': {'static': x_static,
                     'time_series': x_timeseries, },
            'label': x_label,
        }

    def _get_timeseries_data(self, df: pd.DataFrame) -> torch.Tensor:
        pass


class PatientDatasetRaw(PatientDataset):
    def __init__(self, data: list, los: int):
        super().__init__(data, los)

    def _get_timeseries_data(self, df: pd.DataFrame) -> torch.Tensor:
        # 3. time series data
        x_timeseries = []
        # 3.1 chart numeric
        chart_numeric = df[COL_CHART_NUMERIC].to_numpy(dtype=float)
        x_timeseries.append(torch.tensor(chart_numeric, dtype=torch.float))
        # 3.2 chart text
        for col in COL_CHART_TEXT:
            non_na_idx = ~pd.isna(df[col])
            num_classes = len(item_text_vals[col])
            ohk = torch.zeros((df.shape[0], num_classes)).long()
            ohk[non_na_idx] = F.one_hot(
                torch.tensor(df[non_na_idx][col].to_numpy(dtype=float),
                             dtype=torch.long),
                num_classes=num_classes
            )
            x_timeseries.append(ohk.float())
        x_timeseries = torch.cat(x_timeseries, dim=1)
        return x_timeseries


class PatientDatasetEncoded(PatientDataset):
    def __init__(self, data: list, los: int):
        super().__init__(data, los)

    def _get_timeseries_data(self, df: pd.DataFrame) -> torch.Tensor:
        # 3. time series data
        x_timeseries = []
        # 3.1 chart numeric
        for col in COL_CHART_NUMERIC:
            col_encoded = [col + '_' + elem for elem in ENCODE_ELEM]
            chart_numeric = df[col_encoded].to_numpy(dtype=float)
            x_timeseries.append(torch.tensor(chart_numeric, dtype=torch.float))
        # 3.2 chart text
        for col in COL_CHART_TEXT:
            non_na_idx = ~pd.isna(df[col])
            num_classes = len(item_text_vals[col])
            ohk = torch.zeros((df.shape[0], num_classes)).long()
            ohk[non_na_idx] = F.one_hot(
                torch.tensor(df[non_na_idx][col].astype(float).to_numpy(),
                             dtype=torch.long),
                num_classes=num_classes
            )
            x_timeseries.append(ohk.float())
        x_timeseries = torch.cat(x_timeseries, dim=1)
        return x_timeseries


class LigthningModuleBase(LightningModule):
    def __init__(self, data, data_type, los,
                 num_workers=16, validation_stats=False):
        super().__init__()

        self.validation_stats = validation_stats

        self.config = None

        self.los = los
        self.num_workers = num_workers

        self.data_type = data_type
        if data_type == 'raw':
            self.dataset = PatientDatasetRaw
        elif data_type == 'encoded':
            self.dataset = PatientDatasetEncoded
        else:
            raise ValueError(f'Unsupported data_type: {data_type}')

        self.data = data

        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        self.metrics = {
            'acc': torchmetrics.Accuracy().cpu(),
            'recall': torchmetrics.Recall().cpu(),
            'precision': torchmetrics.Precision().cpu(),
            'specification': torchmetrics.Specificity().cpu(),
            'roc': torchmetrics.AUROC(pos_label=1).cpu(),
            'f1': torchmetrics.F1().cpu(),
        }

    def forward(self, x):
        pass

    def configure_optimizers(self):
        adam = optim.Adam(self.parameters(),
                          lr=self.config['lr'],
                          weight_decay=self.config['weight_decay'])
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam, 'min')
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     adam, T_0=5, T_mult=2, eta_min=1e-6,)
        # return [adam], [lr_scheduler]
        return adam

    # def training_epoch_end(self, outputs):
    #     sch = self.lr_schedulers()
    #     # If the selected scheduler is a ReduceLROnPlateau scheduler.
    #     if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #         sch.step(self.trainer.callback_metrics["loss"])

    def _step(self, batch, batch_idx, split: str):
        # split: {train, val, test}

        # lr_sch = self.lr_schedulers()
        # lr_sch.step()

        x = batch['data']
        y = batch['label']

        x_logit, pred = self.forward(x)

        loss_ind = self.loss(x_logit, y)
        loss = loss_ind.mean()

        outputs = {f'{split}_loss': loss, f'{split}_loss_ind': loss_ind}
        self.log(f'{split}_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        if self.validation_stats:
            outputs['high_loss_samples'] = [
                (y[i].cpu(), x_logit[i].cpu(), loss_ind[i].cpu())
                for i in range(loss_ind.shape[0])
                if loss_ind[i] > 50
            ]

        torch.cuda.empty_cache()

        for metric in self.metrics:
            outputs[f'{split}_' + metric] = \
                self.metrics[metric](pred.detach().cpu(),
                                     y.int().detach().cpu())
            self.log(f'{split}_' + metric,
                     outputs[f'{split}_' + metric],
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)

        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, outputs = self._step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self._step(batch, batch_idx, 'val')
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))  # noqa
        return outputs

    def validation_epoch_end(self, validation_step_outputs):
        if self.validation_stats:
            for out in validation_step_outputs:
                if len(out["high_loss_samples"]) > 0:
                    print(out["high_loss_samples"])
            loss_ind = [out["val_loss_ind"] for out in validation_step_outputs]
            loss_ind = torch.cat(loss_ind, dim=0)
            hist = torch.histogram(loss_ind.cpu(), bins=10)
            print("Histogram of Val loss:", hist)

    def test_step(self, batch, batch_idx):
        loss, outputs = self._step(batch, batch_idx, 'test')
        return outputs

    def _dataloader(self, split: str):
        if split == 'train':
            return DataLoader(self.dataset(self.data[split], self.los),
                              sampler=ImbalancedDatasetSampler(self.dataset(self.data[split], self.los)),
                              batch_size=self.config['batch_size'],
                              num_workers=self.num_workers)
        else:
            return DataLoader(self.dataset(self.data[split], self.los),
                              batch_size=self.config['batch_size'],
                              num_workers=self.num_workers)

    def train_dataloader(self):
        return self._dataloader('train')

    def val_dataloader(self):
        return self._dataloader('val')

    def test_dataloader(self):
        return self._dataloader('test')


class CNN_CLF(LigthningModuleBase):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.st_linear = nn.LazyLinear(out_features=64)

        self.cnn_layer1 = nn.Conv1d(
            in_channels=config['n_hidden_cnn'][0],
            out_channels=config['n_hidden_cnn'][1],
            kernel_size=config['cnn_kernel_size'][0],
            stride=config['cnn_stride_size'][0],
            groups=config['cnn_group_size'][0],
        )
        self.cnn_layer2 = nn.Conv1d(
            in_channels=config['n_hidden_cnn'][1]+54,
            out_channels=config['n_hidden_cnn'][2],
            kernel_size=config['cnn_kernel_size'][1],
            stride=config['cnn_stride_size'][1],
            groups=config['cnn_group_size'][1],
        )
        for i in range(config['n_cnn_layer']):
            setattr(
                self, f'cnn_relu{i+1}',
                nn.ReLU()
            )
            setattr(
                self, f'cnn_dropout{i+1}',
                nn.Dropout(p=self.config['dropout_cnn'])
            )
        self.avg_pool1 = nn.AvgPool1d(
            kernel_size=config['cnn_kernel_size'][0],
            stride=config['cnn_stride_size'][0],
        )

        for i in range(len(config['n_hidden_dense'])):
            setattr(
                self, f'fc{i+1}',
                nn.LazyLinear(out_features=config['n_hidden_dense'][i])
            )
            setattr(
                self, f'fc_relu{i+1}',
                nn.ReLU()
            )
            setattr(
                self, f'fc_dropout{i+1}',
                nn.Dropout(p=self.config['dropout_dense'])
            )

        self.output_layer = nn.LazyLinear(out_features=1)

    def forward(self, x):
        if 'label' in x:
            x = x['data']
        x_static = x['static'].float()
        x_ts = x['time_series'].float()
        batch_size = x_static.shape[0]
        if self.data_type == 'raw':
            x_ts_num = x_ts[:, :, :38]
            x_ts_cat = x_ts[:, :, 38:]
        elif self.data_type == 'encoded':
            x_ts_num = x_ts[:, :, :228]
            x_ts_cat = x_ts[:, :, 228:]

        x_static = self.st_linear(x_static)

        x_ts_num = x_ts_num.permute(0, 2, 1)
        x_ts_cat = x_ts_cat.permute(0, 2, 1)

        x_ts_num = getattr(self, f'cnn_layer{1}')(x_ts_num)
        x_ts_num = getattr(self, f'cnn_relu{1}')(x_ts_num)
        x_ts_num = getattr(self, f'cnn_dropout{1}')(x_ts_num)
        x_ts_cat = getattr(self, f'avg_pool{1}')(x_ts_cat)
        x_ts = torch.cat((x_ts_num, x_ts_cat), dim=1)
        x_ts = getattr(self, f'cnn_layer{2}')(x_ts)
        x_ts = getattr(self, f'cnn_relu{2}')(x_ts)
        x_ts = getattr(self, f'cnn_dropout{2}')(x_ts)

        x = torch.cat((x_static.reshape(batch_size, -1),
                          x_ts.reshape(batch_size, -1)),
                         dim=1)

        for i in range(1, len(self.config['n_hidden_dense'])+1):
            x = getattr(self, f'fc{i}')(x)
            x = getattr(self, f'fc_relu{i}')(x)
            x = getattr(self, f'fc_dropout{i}')(x)

        logit = self.output_layer(x)
        output = F.sigmoid(logit)

        return logit, output


def train(args: argparse.Namespace):

    # Load data ----------------------------------------------------------------
    print('Data type: ', args.datatype, 'LOS: ', args.los)
    if args.datatype == 'raw':
        df_data = pickle.load(open(args.datafile_raw, 'rb'))
    elif args.datatype == 'encoded':
        df_data = pickle.load(open(args.datafile_encoded, 'rb'))
    else:
        raise ValueError("Unknown data type")
    df_data = df_data[df_data['mortality']==0]

    # DF split
    datasplit = 'subject_id'

    train_val_idx = df_data.loc[df_data[datasplit].isin(PID_TRAIN), datasplit].unique()
    test_idx = df_data.loc[df_data[datasplit].isin(PID_TEST), datasplit].unique()

    kf = KFold(n_splits=args.kf_splits, shuffle=True,
               random_state=NP_RANDOM_SEED)
    kf_splits = [k for k in kf.split(train_val_idx)]

    # Model config -------------------------------------------------------------
    config = cnn_clf_config(args.datatype, args.bs)
    CLF_Model = CNN_CLF


    # Save directory -----------------------------------------------------------
    model_name = \
        f"{args.split}_" +\
        f"{args.datatype}_" +\
        f"{args.model}_" + \
        f"{'-'.join([str(elem) for elem in config['n_hidden_cnn'][1:]])}cnn_" +\
        f"{'-'.join([str(elem) for elem in config['cnn_kernel_size']])}kernel_" + \
        f"{config['dropout_cnn']}dropout_" + \
        f"{'-'.join([str(elem) for elem in config['n_hidden_dense']])}fc_" + \
        f"{config['dropout_dense']}dropout_" + \
        f"{args.los}day"


    model_folder = os.path.join(args.savedir, model_name)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    model_version = os.listdir(model_folder)
    version = 0
    versions = [int(v.split('_')[1]) for v in model_version if 'version_' in v]
    if len(versions) > 0:
        version = sorted(versions)[-1] + 1

    model_folder = os.path.join(model_folder, f'version_{version}')
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # Save configs -------------------------------------------------------------
    with open(os.path.join(model_folder, 'config.json'), 'w') as fp:
        json.dump(config, fp)
    with open(os.path.join(model_folder, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp)

    # MAIN LOOP ----------------------------------------------------------------
    for kfold in range(args.kf_splits):

        train_idx, val_idx = kf_splits[kfold]
        train_idx = [train_val_idx[i] for i in train_idx]
        val_idx = [train_val_idx[i] for i in val_idx]

        data_train = df_data[df_data[datasplit].isin(train_idx)]
        data_val = df_data[df_data[datasplit].isin(val_idx)]
        data_test = df_data[df_data[datasplit].isin(test_idx)]
        data = {
            'train': data_train,
            'val': data_val,
            'test': data_test,
        }

        model_filename = f'model_cv{kfold}'

        callbacks = [
            ModelCheckpoint(
                monitor='val_roc',
                mode='max',
                save_top_k=1,
                dirpath=model_folder,
                filename=model_filename+'-epoch{epoch:02d}-val_roc{val_roc:.8f}',  # noqa
                auto_insert_metric_name=False
            ),
            EarlyStopping(
                monitor='val_roc',
                mode='max',
                patience=args.patience,
                verbose=True
            )
        ]

        Model = CLF_Model(config,
                          data=data,
                          data_type=args.datatype,
                          los=args.los,
                          num_workers=args.num_workers)

        logger = TensorBoardLogger(model_folder, name=model_filename)

        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=args.gpu,
            callbacks=callbacks,
            logger=logger,
        )

        train_time_start = time.time()
        trainer.fit(Model)
        train_time_total = time.time() - train_time_start

        with open(os.path.join(model_folder, 'train_time.txt'), 'a') as train_time_file:  # noqa
            train_time_file.write(f'{model_filename}: {train_time_total}\n')

        summary = ModelSummary(Model, max_depth=-1)
        print(summary)

        ckpt_roc = glob.glob(os.path.join(model_folder, '*val_roc*.ckpt'))
        roc_score = [float(os.path.splitext(ckpt)[0].split('val_roc')[1]) for ckpt in ckpt_roc]
        ckpt_best = ckpt_roc[np.argmax(roc_score)]
        test_result = trainer.test(ckpt_path=ckpt_best)
        with open(os.path.join(model_folder, f'model_performance_test_data_cv{kfold}.json'), 'w') as fp:
            json.dump(test_result[0], fp)

    return


def eval_cnn_los(model_folder, data_test, pred_task='los'):
    model_config = json.load(open(os.path.join(model_folder, 'config.json'), 'rb'))
    args = json.load(open(os.path.join(model_folder, 'args.json'), 'rb'))
    los = args['los']
    ckpt_roc = glob.glob(os.path.join(model_folder, '*val_roc*.ckpt'))

    data_test_ = data_test[pred_task][args['datatype']][args['split']]
    data = {
        'train': None,
        'val': None,
        'test': data_test_,
    }

    if args['datatype'] == 'raw':
        test_dataset = PatientDatasetRaw(data_test_, los=los)
    elif args['datatype'] == 'encoded':
        test_dataset = PatientDatasetEncoded(data_test_, los=los)
    else:
        raise ValueError("Unknown data type")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=48,
        shuffle=False,
    )

    y_true = test_dataset.get_labels()
    y_preds = []
    for i in range(5):
        model = CNN_CLF.load_from_checkpoint(
            ckpt_roc[i],
            config=model_config,
            data_type=args['datatype'] ,
            los=los,
            data=data
        ).eval()

        trainer = pl.Trainer(
            devices=[1],
            accelerator='gpu',
        )

        # test_result = trainer.test(model)
        predictions = trainer.predict(model, test_dataloader)

        y_ = []
        for logit, prob in predictions:
            y_ += list(prob.numpy().flatten())
        y_preds.append(y_)
    y_pred = np.array(y_preds).mean(axis=0)

    return y_true, y_pred


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    # training
    CLI.add_argument("--kf_splits", type=int, default=5)
    CLI.add_argument("--test_size", type=float, default=.2)
    CLI.add_argument("--max_epochs", type=int, default=1000)
    CLI.add_argument("--patience", type=int, default=30)
    CLI.add_argument("--num_workers", type=int, default=1)
    CLI.add_argument("--savedir", type=str, default='models/CNN_LOS_PRED/delme')  # noqa
    # data
    CLI.add_argument("--datafile_raw", type=str, default='data/preprocessed/data_raw.pkl')
    CLI.add_argument("--datafile_encoded", type=str, default='data/preprocessed/data_emb.pkl')
    CLI.add_argument("--datatype", type=str, default='encoded',)
    # model
    CLI.add_argument("--bs", type=int, default=64)
    # general
    CLI.add_argument("--gpu",  nargs="*", type=int, default=[5],)
    CLI.add_argument("--seed", type=int, default=999,)
    CLI.add_argument("--los", type=int, default=10,)
    args = CLI.parse_args()

    torch.manual_seed(args.seed)

    train(args)


    print("Finish")


