import argparse
import json
import time

from pathlib import Path


from xgboost import XGBClassifier
from xgboost import XGBRFClassifier
from xgboost.callback import TrainingCallback
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.tsa import *
from utils.common_processed import *

import warnings
warnings.filterwarnings("ignore")

NP_RANDOM_SEED = 0
np.random.seed(NP_RANDOM_SEED)

ENCODE_ELEM = ['mean', 'var', 'encoded1', 'encoded2', 'encoded3', 'encoded4']


def xgb_cls_binary_config(args, estimators):
    return {
        # MAIN PARAMS
        'verbosity': 1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug).
        'learning_rate': 0.3,
        'objective': 'binary:logistic',
        'n_estimators': estimators,  # default 100
        'max_depth': 6,
        'n_jobs': args.num_workers,
        'gpu_id': args.gpu[0],
        'predictor': 'gpu_predictor',  # [cpu_predictor, gpu_predictor].
        'eval_metric': ['map', 'auc', 'error'],
        # DEFAULTS
        # 'max_leaves': 0,  # no limit
        # 'max_bin': 256,
        # 'grow_policy': 'depthwise',  # depthwise, lossguide
        # 'booster': 'gbtree',  # gbtree, gblinear or dart; gbtree and dart
        'tree_method': 'gpu_hist',  # auto, exact, approx, hist, gpu_hist
        # 'gamma': 0,
        # 'min_child_weight': 1,
        # 'max_delta_step': 0,
        # 'subsample': 1,
        # 'sampling_method': 'uniform',
        # 'colsample_bytree': 1,
        # 'colsample_bylevel': 1,
        # 'colsample_bynode': 1,
        # 'reg_alpha': 0,  # l1 regs
        # 'reg_lambda': 1,  # l2 reg
        # 'scale_pos_weight': 1,
        # 'base_score': NA,
        # 'random_state': np.random.RandomState(seed=NP_RANDOM_SEED),
        # 'missing': np.nan,
        # 'num_parallel_tree': 1,  # Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.  # noqa
        # 'monotone_constraints': NA,
        # 'interaction_constraints': NA,
        # 'importance_type': NA,
        # 'validate_parameters': False,
        # 'enable_categorical': NA,
        # 'feature_types': NA,
        # 'max_cat_to_onehot': NA,
        # 'max_cat_threshold': NA,
        # 'early_stopping_rounds': NA,
        # 'callbacks': [],
    }


def xgbrf_cls_binary_config(args, estimators):
    return {
        # Predefined
        'learning_rate': 1.0,  # default 0.3
        'subsample': 0.8,  # default 1
        'colsample_bynode': 0.8,  # default 1
        'reg_lambda': 1e-5,  # default 1
        # MAIN PARAMS
        'verbosity': 1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug).
        'objective': 'binary:logistic',
        'n_estimators': estimators,  # default 100
        'max_depth': 6,
        'n_jobs': args.num_workers,
        'gpu_id': args.gpu[0],
        'predictor': 'gpu_predictor',  # [cpu_predictor, gpu_predictor].
        'eval_metric': ['map', 'auc', 'error'],
        # DEFAULTS
        # 'max_leaves': 0,  # no limit
        # 'max_bin': 256,
        # 'grow_policy': 'depthwise',  # depthwise, lossguide
        # 'booster': 'gbtree',  # gbtree, gblinear or dart; gbtree and dart
        'tree_method': 'gpu_hist',  # auto, exact, approx, hist, gpu_hist
        # 'gamma': 0,
        # 'min_child_weight': 1,
        # 'max_delta_step': 0,
        # 'sampling_method': 'uniform',
        # 'colsample_bytree': 1,
        # 'colsample_bylevel': 1,
        # 'reg_alpha': 0,  # l1 regs
        # 'scale_pos_weight': 1,
        # 'base_score': NA,
        # 'random_state': np.random.RandomState(seed=NP_RANDOM_SEED),
        # 'missing': np.nan,
        # 'num_parallel_tree': 1,  # Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.  # noqa
        # 'monotone_constraints': NA,
        # 'interaction_constraints': NA,
        # 'importance_type': NA,
        # 'validate_parameters': False,
        # 'enable_categorical': NA,
        # 'feature_types': NA,
        # 'max_cat_to_onehot': NA,
        # 'max_cat_threshold': NA,
        # 'early_stopping_rounds': NA,
        # 'callbacks': [],
    }


# FROM : https://github.com/dmlc/xgboost/issues/5727
class TensorBoardCallback(TrainingCallback):
    def __init__(self, log_dir: str = None, save_metrics: bool = True):
        self.save_metrics = save_metrics
        self.log_dir = os.path.join(log_dir, "log")
        self.val_dir = os.path.join(self.log_dir, "val")
        self.val_writer = SummaryWriter(log_dir=self.val_dir)
        if self.save_metrics:
            self.val_metric_file = os.path.join(self.val_dir, "metrics.txt")
            f = open(self.val_metric_file, "w")
            f.close()

    def after_iteration(
        self, model, epoch: int, evals_log: TrainingCallback.EvalsLog
    ) -> bool:
        if not evals_log:
            return False

        for data, metric in evals_log.items():
            printout = f'{epoch}'
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                printout += f"::{metric_name}:{score}"

                s_metric_name = f"{data}/{metric_name}"
                self.val_writer.add_scalar(s_metric_name, score, epoch)

            if self.save_metrics:
                # https://thispointer.com/how-to-append-text-or-lines-to-a-file-in-python/
                with open(self.val_metric_file, "a+") as f:
                    f.seek(0)
                    if len(f.read(100)) > 0:
                        f.write("\n")
                    f.write(printout)

        return False


class PatientDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 datasplit: str,
                 los: int = 0,
                 mortality: bool = False,
                 mode: str = 'max'):
        self.data = data
        self.datasplit = datasplit
        self.los = los
        self.mortality = mortality
        self.mode = mode

        self.data_processed = None
        self._preprocess_data()

    def __len__(self):
        return len(self.data['sample_id'].unique())

    def __getitem__(self, idx):
        return self.data_processed[0][idx], self.data_processed[1][idx]

    def _get_label(self, df: pd.DataFrame) -> torch.Tensor:
        if self.los > 0:
            if df['LOS'].iloc[-1] > self.los * 24:
                x_label = torch.Tensor([1])
            else:
                x_label = torch.Tensor([0])
        elif self.mortality:
            if df['mortality'].mean() > 0.0:
                x_label = torch.Tensor([1])
            else:
                x_label = torch.Tensor([0])
        else:
            raise ValueError("Wrong mode for label defined...")
        return x_label

    def _get_static_data(self, df: pd.DataFrame) -> torch.Tensor:
        x_static = []
        # 2.1 info numeric
        info_numeric = [df[i].iloc[0] for i in COL_INFO_NUMERIC]
        x_static.append(torch.tensor(info_numeric, dtype=torch.float))
        # 2.2 info text (1 hot)
        for col in COL_INFO_TEXT:
            ohk = F.one_hot(
                torch.tensor(df[col].iloc[0].astype(int)),
                num_classes=len(info_text_vals[col])
            )
            x_static.append(ohk)
        return torch.cat(x_static, dim=0)

    def _get_timeseries_data(self, df: pd.DataFrame) -> torch.Tensor:
        raise NotImplementedError

    def _preprocess_data(self):
        x_all, y_all = [], []
        for idx in range(len(self.data[self.datasplit].unique())):
            split_idx = self.data[self.datasplit].unique()[idx]
            df_sample = self.data[self.data[self.datasplit] == split_idx]

            for sid in df_sample['sample_id'].unique():
                df = df_sample[df_sample['sample_id']==sid]

                # 1. labels
                y = self._get_label(df)

                # 2. static data
                x_static = self._get_static_data(df)

                # 3. time series data
                x_timeseries = self._get_timeseries_data(df)

                # 4. mode
                if self.mode == 'max':
                    x, _ = torch.max(x_timeseries, dim=0)
                    x = torch.cat([x_static, x], dim=-1)
                elif self.mode == 'min':
                    x, _ = torch.min(x_timeseries, dim=0)
                    x = torch.cat([x_static, x], dim=-1)
                elif self.mode == 'avg':
                    x = torch.mean(x_timeseries, dim=0)
                    x = torch.cat([x_static, x], dim=-1)
                elif self.mode == 'trend_10':
                    x = torch.mean(x_timeseries[-10:, :], dim=0) -  \
                        torch.mean(x_timeseries[:10, :], dim=0)
                    x = torch.cat([x_static, x], dim=-1)
                elif self.mode == 'max_avg':
                    x_max, _ = torch.max(x_timeseries, dim=0)
                    x_avg = torch.mean(x_timeseries, dim=0)
                    x = torch.cat([x_static, x_max, x_avg], dim=-1)
                x_all.append(x)
                y_all.append(y)

        x_all = torch.stack(x_all, dim=0).float()
        y_all = torch.stack(y_all, dim=0)

        self.data_processed = (x_all, y_all)


class PatientDatasetRaw(PatientDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


def train(args: argparse.Namespace):

    # Load data ----------------------------------------------------------------
    print('Data type: ', args.datatype)
    if args.datatype == 'raw':
        df_data = pickle.load(open(args.datafile_raw, 'rb'))
        dataset = PatientDatasetRaw
    elif args.datatype == 'encoded':
        df_data = pickle.load(open(args.datafile_encoded, 'rb'))
        dataset = PatientDatasetEncoded


    # DF filter
    if not args.mortality:
        df_data = df_data[df_data['mortality'] == 0]

    # DF split
    datasplit = 'subject_id'
    train_val_idx = df_data.loc[df_data[datasplit].isin(PID_TRAIN), datasplit].unique()

    kf = KFold(n_splits=args.kf_splits, shuffle=True,
               random_state=NP_RANDOM_SEED)
    kf_split_idxs = [k for k in kf.split(train_val_idx)]


    # LOOP THROUGH CRITERIONS FOR MODEL TUNING ---------------------------------
    for estimators in args.estimators:

        # Save directory -------------------------------------------------------
        if args.los > 0:
            ending = f"{args.los}day"
        elif args.mortality:
            ending = f"mortality"
        model_name = f"{args.datatype}_{estimators}estimators_{args.timeseries_mode}ts_{ending}"  # noqa

        model_folder = os.path.join(args.savedir, model_name)
        Path(model_folder).mkdir(parents=True, exist_ok=True)

        model_version = os.listdir(model_folder)
        version = 0
        versions = [int(v.split('_')[1])
                    for v in model_version if 'version_' in v]
        if len(versions) > 0:
            version = sorted(versions)[-1] + 1

        model_folder = os.path.join(model_folder, f'version_{version}')
        Path(model_folder).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(model_folder, 'args.json'), 'w') as fp:
            json.dump(vars(args), fp)

        # MAIN LOOP ------------------------------------------------------------
        for kfold in range(args.kf_splits):
            train_idx, val_idx = kf_split_idxs[kfold]
            train_idx = [train_val_idx[i] for i in train_idx]
            val_idx = [train_val_idx[i] for i in val_idx]

            df_data_train = df_data[df_data[datasplit].isin(train_idx)]
            df_data_val = df_data[df_data[datasplit].isin(val_idx)]


            dl = DataLoader(dataset(data=df_data_train,
                                    datasplit=datasplit,
                                    los=args.los,
                                    mortality=args.mortality,
                                    mode=args.timeseries_mode),
                            batch_size=len(df_data_train['sample_id'].unique()),
                            # batch_size=args.bs,
                            num_workers=args.num_workers)
            data_train = next(iter(dl))

            dl = DataLoader(dataset(data=df_data_val,
                                    datasplit=datasplit,
                                    los=args.los,
                                    mortality=args.mortality,
                                    mode=args.timeseries_mode),
                            batch_size=len(df_data_val['sample_id'].unique()),
                            # batch_size=args.bs,
                            num_workers=args.num_workers)
            data_val = next(iter(dl))

            k_folder = f'cv{kfold}'
            log_dir = os.path.join(model_folder, k_folder)

            if args.model == 'xgboost':
                model_cfg = xgb_cls_binary_config(args, estimators)
                model_cfg['callbacks'] = [
                    TensorBoardCallback(log_dir=log_dir),
                ]
                model = XGBClassifier(**model_cfg)
            elif args.model == 'randomforest':
                model_cfg = xgbrf_cls_binary_config(args, estimators)
                model = XGBRFClassifier(**model_cfg)

            # Train ------------------------------------------------------------
            train_time_start = time.time()
            model.fit(data_train[0], data_train[1],
                      eval_set=[data_train, data_val])
            train_time_total = time.time() - train_time_start
            model.save_model(os.path.join(model_folder, f'model_cv{kfold}.json'))

            with open(os.path.join(model_folder, 'train_time.txt'), 'a') as f:
                f.write(f'{k_folder}: {train_time_total}\n')

            if args.model == 'randomforest':
                callback = TensorBoardCallback(log_dir=log_dir)
                results = model.evals_result()
                callback.after_iteration(None, estimators, results)

            # Validation -------------------------------------------------------
            label = data_val[1]
            y = model.predict(data_val[0])
            y_prob = model.predict_proba(data_val[0])[:, 1]  # class 1 as pos
            # evaluate predictions
            # ap = average_precision_score(label, y_prob) # same as auprc
            accuracy = accuracy_score(label, y)
            f1 = f1_score(label, y)
            precision_50 = precision_score(label, y)
            recall_50 = recall_score(label, y, pos_label=1)
            specificity_50 = recall_score(label, y, pos_label=0)
            precision, recall, thres = precision_recall_curve(label, y_prob)
            auprc = auc(recall, precision)
            auroc = roc_auc_score(label, y_prob)
            printout = ""
            # printout += f"ap:{ap:.3f}::"
            printout += f"acc:{accuracy:.3f}::"
            printout += f"f1:{f1:.3f}::"
            printout += f"precision_50:{precision_50:.3f}::"
            printout += f"recall_50:{recall_50:.3f}::"
            printout += f"specificity_50:{specificity_50:.3f}::"
            printout += f"precision:{precision}::"
            printout += f"recall:{recall}::"
            printout += f"auprc:{auprc:.3f}::"
            printout += f"auroc:{auroc:.3f}"
            # print(printout)

            results = {
                'validation_1': {
                    'score_accuracy': [accuracy],
                    'score_f1': [f1],
                    'score_precision': [precision_50],
                    'score_recall': [recall_50],
                    'score_specifictiy': [specificity_50],
                    'score_auprc': [auprc],
                    'score_auroc': [auroc],
                }
            }

            if args.model == 'xgboost':
                tbc = model_cfg['callbacks'][0]
                tbc.after_iteration(None, estimators, results)
            elif args.model == 'randomforest':
                tbc = callback
                tbc.after_iteration(None, estimators, results)

            with open(os.path.join(model_folder, 'final_metric.txt'), 'a') as f:
                f.write(f'{k_folder}: {printout}\n')

    return


def eval_xgboost(model_folder, data_test, pred_task='los'):
    args = json.load(open(os.path.join(model_folder, 'args.json'), 'rb'))

    data_test_ = data_test[pred_task][args['datatype']][args['datasplit']]

    dl = DataLoader(PatientDatasetRaw(data=data_test_,
                            datasplit=args['datasplit']+'_id',
                            los=args['los'],
                            mortality=args['mortality'],
                            mode=args['timeseries_mode']),
                    batch_size=len(data_test_['sample_id'].unique()),
                    # batch_size=args.bs,
                    num_workers=48)
    data_test = next(iter(dl))
    y_true = data_test[1].numpy().flatten()
    y_probs = []
    y_preds = []
    for i in range(5):
        if args['model'] == 'xgboost':
            model = XGBClassifier()
        elif args['model'] == 'randomforest':
            model = XGBRFClassifier()
        model.load_model(os.path.join(model_folder, f'model_cv{i}.json'))

        y_pred = model.predict(data_test[0])
        y_prob = model.predict_proba(data_test[0])[:, 1]  # class 1 as pos
        y_preds.append(y_pred)
        y_probs.append(y_prob)
    y_prob = np.array(y_probs).mean(axis=0)

    return y_true, y_prob

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    # training
    CLI.add_argument("--kf_splits", type=int, default=5)
    CLI.add_argument("--test_size", type=float, default=.2)
    CLI.add_argument("--num_workers", type=int, default=1)
    CLI.add_argument("--savedir", type=str, default='models/XGBoost/delme')  # noqa
    # data
    CLI.add_argument("--datafile_raw", type=str, default='data/preprocessed/data_raw.pkl')  # noqa
    CLI.add_argument("--datafile_encoded", type=str, default='data/preprocessed/data_emb.pkl')  # noqa
    CLI.add_argument("--datatype", type=str, default='raw',)
    CLI.add_argument("--timeseries_mode", type=str, default='max',)
    # model
    CLI.add_argument("--model", type=str, default='xgboost',)
    CLI.add_argument("--estimators", nargs="*", type=int, default=[100])
    CLI.add_argument("--bs", type=int, default=64)
    # general
    CLI.add_argument("--gpu",  nargs="*", type=int, default=[5],)
    CLI.add_argument("--seed", type=int, default=999,)
    CLI.add_argument("--los", type=int, default=10,)
    CLI.add_argument("--mortality", type=bool, default=False,)
    args = CLI.parse_args()

    print(args)

    torch.manual_seed(args.seed)

    assert args.model in ['xgboost', 'randomforest']
    assert args.datatype in ['raw', 'encoded']
    assert args.timeseries_mode in ['min', 'max', 'avg', 'trend_10', 'max_avg']
    assert args.mortality or args.los > 0

    train(args)


    print("Finish")
