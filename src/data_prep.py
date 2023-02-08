import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath('.')))
import multiprocessing

from utils.tsa import *
from utils.common import *

import tqdm
import pandas as pd
NP_RANDOM_SEED=0
np.random.seed(NP_RANDOM_SEED)

from sklearn.model_selection import train_test_split
import GPy as gpy
from GPy.likelihoods import Gaussian
from GPy.likelihoods.link_functions import Identity


def one_hot_encoding(text, text_dict):
    ohk = [0] * len(text_dict)
    ohk[text_dict[text]] = 1
    return ohk

def  one_hot_encoding_arr(arr, text_dict):
    ohk_arr = []
    for text in arr:
        ohk = one_hot_encoding(text, text_dict)
        ohk_arr.append(ohk)
    return np.array(ohk)


def generate_raw_samples(config):
    adm = config['adm']
    data_raw_path = config['data_raw_path']
    save_path_raw = config['save_path_raw']
    text_ohk_info = config['text_ohk_info']
    text_ohk_chart = config['text_ohk_chart']

    data = pd.read_csv(os.path.join(data_raw_path, f'{adm}.csv'))
    info = pd.read_csv(os.path.join(data_raw_path, f'{adm}_info.csv'))
    data_ = data.copy()
    data_['charttime'] = pd.to_datetime(data_['charttime'])

    #one hot encode + interpolation
    for iid in selected_numeric_iid:
        d = data[data['itemid']==iid]['valuenum']
        data_.loc[data_['itemid']==iid, 'valuenum'] = d
    for iid in selected_text_iid:
        d = data[data['itemid']==iid]['value']
        data_.loc[data_['itemid']==iid, 'valuenum'] = [text_ohk_chart[iid][di] for di in d.values]

    ct_disch = pd.to_datetime(info['dischtime']).item()
    ct_start = data_['charttime'].min()
    ct_start = ct_start.replace(hour=ct_start.hour, minute=0, second=0, microsecond=0)
    ct_end = data_['charttime'].max()
    if (ct_end-ct_start).days * 24 + (ct_end-ct_start).seconds / 3600 < 48:
        print(f'{adm}: chart data time length {ct_end-ct_start} < 48H')
        return 0
    data_ = interpolate_charttime_df(data_, freq='1H', method='fillin', begin=ct_start, end=ct_end)

    # reformulate dataframe in [timestamp] - [data column] format
    columns = ['hadm_id', 'charttime', 'LOS'] + ['age', 'admission_weight', 'gender', 'admission_type'] + COL_CHART_TEXT + COL_CHART_NUMERIC
    df_data = pd.DataFrame(columns=columns)
    df_data['charttime'] = pd.date_range(ct_start, ct_end, freq='1H')
    los = ct_disch - df_data['charttime']
    df_data['LOS'] = [dt.days*24 + dt.seconds//3600 for dt in los]
    df_data['hadm_id'] = adm
    for col in ['age', 'admission_weight']:
        df_data[col] = info[col].item()
    for col in ['gender', 'admission_type']:
        df_data[col] = text_ohk_info[col][info[col].item()]
    for iid in selected_numeric_iid + selected_text_iid:
        try:
            col = item_dict[item_dict['itemid']==iid]['abbreviation'].item()
            df_data[col] = data_.loc[data_['itemid']==iid, 'valuenum'].values
        except:
            pass

    admittime = pd.to_datetime(info['admittime']).item()
    dischtime = pd.to_datetime(info['dischtime']).item()
    subject_id = info['subject_id'].item()
    if not pd.isnull(pd.to_datetime(info['deathtime']).item()):
        mortality = True
    else:
        mortality = False

    sample_raw = df_data[(df_data['charttime']>admittime) & (df_data['charttime']<dischtime)].copy()
    sample_raw = sample_raw[['LOS'] + col_info + COL_CHART_TEXT + COL_CHART_NUMERIC].reset_index(drop=True)
    sample_raw['subject_id'] = subject_id
    sample_raw['hadm_id'] = adm
    if mortality:
        sample_raw['mortality'] = 1
    else:
        sample_raw['mortality'] = 0

    pickle.dump(sample_raw, open(os.path.join(save_path_raw, f'{adm}.pkl'), 'wb'))



def calculate_norm_params(data_path, sample_train):
    info_vals = {
        'age':[],
        'admission_weight':[],
    }
    item_vals = {
        col:[] for col in COL_CHART_NUMERIC
    }
    norm_param_info = {col:{} for col in ['age', 'admission_weight']}
    norm_param_numeric = {
        col:{} for col in COL_CHART_NUMERIC
    }

    for sample_file in tqdm.tqdm(sample_train):
        data = pickle.load(open(os.path.join(data_path, f'{sample_file}.pkl'),'rb'))
        for col in info_vals:
            info_vals[col] += [data[col].iloc[-1]]
        for col in COL_CHART_NUMERIC:
            val = data[col].astype(float).to_list()
            item_vals[col] += val

    for col in norm_param_numeric:
        d = np.array(item_vals[col])
        d = d[~np.isnan(d)]
        lower = np.percentile(d, 1)
        upper = np.percentile(d, 99)
        print(f'{col}: {lower}--{upper}')
        d_1_99 = d[np.argwhere((d >= lower) & (d <= upper))]

        norm_param_numeric[col]['max'] = np.nanmax(d_1_99)
        norm_param_numeric[col]['min'] = np.nanmin(d_1_99)
        norm_param_numeric[col]['mean'] = np.nanmean(d_1_99)
        norm_param_numeric[col]['std'] = np.nanstd(d_1_99)

    for col in norm_param_info:
        d = np.array(info_vals[col])
        d = d[~np.isnan(d)]
        lower = np.percentile(d, 1)
        upper = np.percentile(d, 99)
        d_1_99 = d[np.argwhere((d >= lower) & (d <= upper))]
        norm_param_info[col]['max'] = np.nanmax(d_1_99)
        norm_param_info[col]['min'] = np.nanmin(d_1_99)
        norm_param_info[col]['mean'] = np.nanmean(d_1_99)
        norm_param_info[col]['std'] = np.nanstd(d_1_99)

    pickle.dump(norm_param_numeric, open(os.path.join(data_path, '..', 'norm_param_chart.pkl'), 'wb'))
    pickle.dump(norm_param_info, open(os.path.join(data_path, '..', 'norm_param_info.pkl'), 'wb'))

    return 1


def encode_data(data, plot=False):
    df_data = data.copy()
    df_data = df_data.reindex(columns = df_data.columns.tolist() + [item_dict[item_dict['itemid']==iid]['abbreviation'].item() + '_mean' for iid in selected_numeric_iid])
    df_data = df_data.reindex(columns = df_data.columns.tolist() + [item_dict[item_dict['itemid']==iid]['abbreviation'].item() + '_var' for iid in selected_numeric_iid])
    df_data = df_data.reindex(columns = df_data.columns.tolist() + [item_dict[item_dict['itemid']==iid]['abbreviation'].item() + f'_encoded{i}' for iid in selected_numeric_iid for i in range(1, 5)])
    for iid in selected_numeric_iid:
        col = item_dict[item_dict['itemid']==iid]['abbreviation'].item()

        d_encoded = np.zeros((df_data.shape[0], 4))

        y = df_data[col].to_numpy().astype(float)
        x = np.linspace(0, df_data.shape[0]-1, df_data.shape[0])
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        if x.size == 0:
            df_data[df_data.columns[df_data.columns.str.startswith(col+'_encoded')]] = 0
            continue

        anomaly = detect_sudden_change_anomaly(x, y, margin=.2, plot=False)
        x_ = np.delete(x, anomaly['peaks'])
        y_ = np.delete(y, anomaly['peaks'])

        kern_trend = gpy.kern.RBF(input_dim=1, variance=1., lengthscale=24)
        kern_trend.lengthscale.constrain_bounded(lower=24, upper=48, warning=False)
        kern_periodic_std = gpy.kern.StdPeriodic(input_dim=1, variance=1., lengthscale=3., period=24)
        kern_periodic_std.period.constrain_bounded(22, 26, warning=False)
        kern_periodic_std.lengthscale.constrain_bounded(.5, 10, warning=False)

        model_trend = gpy.models.GPRegression(x_.reshape(-1, 1), y_.reshape(-1, 1), kern_trend)
        model_trend.optimize()

        trend_mean, trend_cov = model_trend.predict(x_.reshape(-1, 1), full_cov=True)
        y_detrend = y_ - trend_mean.ravel()

        model_season = gpy.models.GPRegression(x_.reshape(-1, 1), y_detrend.reshape(-1, 1), kern_periodic_std)
        model_season.optimize()

        trend_mean, trend_var = model_trend.predict(np.arange(0, df_data.shape[0]+1).reshape(-1, 1), full_cov=False)
        season_mean, season_var = model_season.predict(np.arange(0, df_data.shape[0] + 1).reshape(-1, 1), full_cov=False)
        trend_diff = trend_mean[1:] - trend_mean[:-1]
        trend_perc = np.array(model_trend.predict_quantiles(np.arange(0, df_data.shape[0]+1).reshape(-1, 1),
                                                            quantiles=(2.5, 97.5),
                                                            likelihood=Gaussian(Identity(), 1e-9))).reshape(2, df_data.shape[0]+1)
        season_perc = np.array(model_season.predict_quantiles(np.arange(0, df_data.shape[0]+1).reshape(-1, 1),
                                                              quantiles=(2.5, 97.5),
                                                              likelihood=Gaussian(Identity(), 1e-9))).reshape(2, df_data.shape[0]+1)

        mean = trend_mean + season_mean
        perc = trend_perc + season_perc

        lower = perc[0,:]
        upper = perc[1,:]

        for i, idx in enumerate(x_.astype(int)):
            if y_[i] < lower[idx]:
                if trend_diff[idx] < 0:
                    d_encoded[idx, 0] = y_[i] - trend_mean[idx]
                elif trend_diff[idx] >= 0:
                    d_encoded[idx, 1] = y_[i] - trend_mean[idx]
            elif y_[i] > upper[idx]:
                if trend_diff[idx] < 0:
                    d_encoded[idx, 2] = y_[i] - trend_mean[idx]
                elif trend_diff[idx] >= 0:
                    d_encoded[idx, 3] = y_[i] - trend_mean[idx]

        df_data[col+'_mean'] = mean[:-1]
        df_data[col+'_var'] = perc[1,:-1] - perc[0, :-1]
        df_data[df_data.columns[df_data.columns.str.startswith(col+'_encoded')]] = d_encoded

        if plot:
            x_plot = np.arange(-24, df_data.shape[0]+24).reshape(-1, 1)
            trend_mean, trend_var = model_trend.predict(x_plot, full_cov=False)
            season_mean, season_var = model_season.predict(x_plot, full_cov=False)
            trend_perc = np.array(model_trend.predict_quantiles(x_plot, quantiles=(2.5, 97.5),
                                                                likelihood=Gaussian(Identity(), 1e-9))).reshape(2, x_plot.size)
            season_perc = np.array(model_season.predict_quantiles(x_plot, quantiles=(2.5, 97.5),
                                                                  likelihood=Gaussian(Identity(), 1e-9))).reshape(2, x_plot.size)
            mean = trend_mean + season_mean
            perc = trend_perc + season_perc

            fig, ax = plt.subplots(3, 1, figsize=(9, 6))
            plt.tight_layout()
            plt.subplots_adjust(bottom=.1, top=.9, right=.93, hspace=.5)
            model_trend.plot_f(ax=ax[0])
            ax[0].scatter(x_, y_, c='black', marker='x')
            ax[0].scatter(x[anomaly['peaks']], y[anomaly['peaks']], c='r', marker='^')
            ax[0].set_title('Trend')

            model_season.plot_f(ax=ax[1])
            ax[1].scatter(x_, y_detrend, c='black', marker='x')
            ax[1].set_title('Seasonality')

            ax[2].plot(np.arange(-24, df_data.shape[0]+24), mean, label='mean')
            ax[2].fill_between(x_plot.ravel(), perc[0,:].ravel(), perc[1,:].ravel(), label='confidence', alpha=.3, color='b')
            ax[2].scatter(x_, y_, c='black', marker='x')
            ax[2].set_xlabel(f'time [h]')
            ax[2].set_title(f'Gaussian process model of {col}')

            plt.suptitle(col, fontsize=16)

            plt.show()
            print(11)

    return df_data


def generate_48h_encoded_samples(config):
    sample_file = config['sample_file']
    data_processed_path = config['data_processed_path']
    save_path_raw_norm = config['save_path_raw_norm']
    save_path_emb = config['save_path_emb']
    norm_param_info = config['norm_param_info']
    norm_param_numeric = config['norm_param_numeric']
    zero_imp = config['zero_imp']

    sample_raw = pickle.load(open(os.path.join(data_processed_path, 'raw', sample_file), 'rb'))
    for col in norm_param_info:
        sample_raw[col] = (sample_raw[col] - norm_param_info[col]['mean']) /  norm_param_info[col]['std']
    for col in norm_param_numeric:
        sample_raw[col] = (sample_raw[col] - norm_param_numeric[col]['mean']) /  norm_param_numeric[col]['std']

    for i in range(sample_raw.shape[0]//48):
        row_start = 48 * i
        row_end = 48 * (i + 1)

        sample_raw_curr = sample_raw.iloc[row_start:row_end].copy().reset_index(drop=True)
        if (pd.isna(sample_raw_curr[col_chart]).sum().sum() / sample_raw_curr[col_chart].size > .9)\
                or (pd.isna(sample_raw_curr[col_chart]).all(axis=0).sum() / len(col_chart) > .5):
            continue
        sample_encoded = encode_data(sample_raw_curr, plot=False)
        sample_encoded = sample_encoded[['LOS'] + COL_INFO_NUMERIC + col_text + col_numeric_encoded].reset_index(drop=True)

        sample_encoded = sample_encoded.reindex(columns=['subject_id', 'hadm_id', 'mortality']+sample_encoded.columns.to_list())
        sample_encoded['subject_id'] = sample_raw_curr['subject_id']
        sample_encoded['hadm_id'] = sample_raw_curr['hadm_id']
        sample_encoded['mortality'] = sample_raw_curr['mortality']

        sample_raw_curr.fillna(value=zero_imp, inplace=True)
        sample_encoded.fillna(value=zero_imp, inplace=True)

        pickle.dump(sample_raw_curr, open(os.path.join(save_path_raw_norm, f'_{i}'.join(os.path.splitext(sample_file))), 'wb'))
        pickle.dump(sample_encoded, open(os.path.join(save_path_emb,  f'_{i}'.join(os.path.splitext(sample_file))), 'wb'))




if __name__ == '__main__':
    data_path = os.path.join(os.path.abspath(''), 'data')
    data_raw_path = os.path.join(data_path, 'raw')
    data_processed_path = os.path.join(data_path, 'preprocessed')
    save_path_raw = os.path.join(data_processed_path, 'raw')
    save_path_raw_norm = os.path.join(data_processed_path, 'raw_normalized')
    save_path_emb = os.path.join(data_processed_path, 'emb')
    Path(save_path_raw).mkdir(parents=True, exist_ok=True)
    Path(save_path_raw_norm).mkdir(parents=True, exist_ok=True)
    Path(save_path_emb).mkdir(parents=True, exist_ok=True)

    text_ohk_chart = pickle.load(open(os.path.join(data_path, 'text_chart_onehotkey.pkl'), 'rb'))
    text_ohk_info = pickle.load(open(os.path.join(data_path, 'text_info_onehotkey.pkl'), 'rb'))

    # df_check = pickle.load(open(os.path.join(data_path, 'preprocessed', '26892283.pkl'), 'rb'))


    # Generate raw samples per admission
    with multiprocessing.Pool(60) as pool:
        for _ in tqdm.tqdm(
                pool.imap_unordered(
                    generate_raw_samples,
                    [dict(
                        adm=adm,
                        data_raw_path=data_raw_path,
                        save_path_raw=save_path_raw,
                        save_path_emb=save_path_emb,
                        text_ohk_chart=text_ohk_chart,
                        text_ohk_info=text_ohk_info,
                    ) for adm in adm_cohort]
                ), total=len(adm_cohort)
        ):
            pass

    # Train-Test Split
    adm_files = os.listdir(save_path_raw)
    adm_files = [int(os.path.splitext(file)[0]) for file in adm_files]
    pid_files = []
    pid_adm_dict = {}
    for adm in tqdm.tqdm(adm_files):
        info = pd.read_csv(os.path.join(data_raw_path, f'{adm}_info.csv'))
        pid = info['subject_id'].item()
        pid_files.append(pid)
        if pid in pid_adm_dict:
            pid_adm_dict[pid].append(adm)
        else:
            pid_adm_dict[pid] = [adm]
    pid_files = list(set(pid_files))

    pid_train, pid_test = train_test_split(
        pid_files,
        test_size=.2,
        random_state=NP_RANDOM_SEED,
        shuffle=True
    )
    pickle.dump(pid_train, open(os.path.join(data_processed_path, 'pid_train.pkl'), 'wb'))
    pickle.dump(pid_test, open(os.path.join(data_processed_path, 'pid_test.pkl'), 'wb'))

    adm_train = []
    adm_test = []
    for pid in pid_adm_dict:
        if pid in pid_train:
            adm_train += pid_adm_dict[pid]
        elif pid in pid_test:
            adm_test += pid_adm_dict[pid]
    pickle.dump(adm_train, open(os.path.join(data_processed_path, 'adm_train.pkl'), 'wb'))
    pickle.dump(adm_test, open(os.path.join(data_processed_path, 'adm_test.pkl'), 'wb'))
    # adm_train = pickle.load(open(os.path.join(data_processed_path, 'adm_train.pkl'), 'rb'))
    # adm_test = pickle.load(open(os.path.join(data_processed_path, 'adm_test.pkl'), 'rb'))


    # Calculate normalization parameters
    calculate_norm_params(data_path=save_path_raw, sample_train=adm_train)

    from utils.common_processed import *

    # Encode data
    sample_files = os.listdir(save_path_raw)

    norm_param_numeric = pickle.load(open(os.path.join(data_processed_path, 'norm_param_chart.pkl'), 'rb'))
    norm_param_info = pickle.load(open(os.path.join(data_processed_path, 'norm_param_info.pkl'), 'rb'))
    with multiprocessing.Pool(60) as pool:
        for _ in tqdm.tqdm(
                pool.imap_unordered(
                    generate_48h_encoded_samples,
                    [dict(
                        sample_file=sample_file,
                        data_processed_path=data_processed_path,
                        save_path_raw_norm=save_path_raw_norm,
                        save_path_emb=save_path_emb,
                        norm_param_info=norm_param_info,
                        norm_param_numeric=norm_param_numeric,
                        zero_imp=zero_imp,
                    ) for sample_file in sample_files]
                ), total=len(sample_files)
        ):
            pass


    # Merge samples in one dataframe
    list_raw = []
    list_emb = []
    train_files = []
    test_files = []
    for sample_id, file in enumerate(tqdm.tqdm(os.listdir(save_path_emb))):
        file_adm = int(file.split('_')[0])
        if file_adm in adm_train:
            train_files.append(file)
        elif file_adm in adm_test:
            test_files.append(file)
        else:
            raise ValueError(f'{file} not exists in training / test admissions.')

        raw_data = pickle.load(open(os.path.join(save_path_raw_norm, file), 'rb'))
        emb_data = pickle.load(open(os.path.join(save_path_emb, file), 'rb'))
        raw_data = raw_data.reindex(columns=['sample_id']+raw_data.columns.to_list())
        emb_data = emb_data.reindex(columns=['sample_id']+emb_data.columns.to_list())
        raw_data['sample_id'] = sample_id
        emb_data['sample_id'] = sample_id

        list_raw.append(raw_data)
        list_emb.append(emb_data)

    df_raw = pd.concat(list_raw, axis=0)
    df_emb = pd.concat(list_emb, axis=0)
    pickle.dump(df_raw, open(os.path.join(data_processed_path, 'data_raw.pkl'), 'wb'))
    pickle.dump(df_emb, open(os.path.join(data_processed_path, 'data_emb.pkl'), 'wb'))
    pickle.dump(train_files, open(os.path.join(data_processed_path, 'files_train.pkl'), 'wb'))
    pickle.dump(test_files, open(os.path.join(data_processed_path, 'files_test.pkl'), 'wb'))

    # df_raw = pickle.load(open(os.path.join(data_processed_path, 'data_raw.pkl'), 'rb'))
    # df_emb = pickle.load(open(os.path.join(data_processed_path, 'data_emb.pkl'), 'rb'))



    print(111)

