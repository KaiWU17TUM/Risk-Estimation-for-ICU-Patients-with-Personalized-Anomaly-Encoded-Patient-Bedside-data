import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from scipy.interpolate import interp1d
import statsmodels.api as sm
from loess.loess_1d import loess_1d

from scipy.signal import find_peaks

from utils.common import *

def load_data(file_folder=os.path.join(os.path.abspath(''), '..', 'data')):
    vital_ref_dict = pd.read_csv(os.path.join(file_folder, 'vital_ref_dict'), header=0, index_col=[0])
    df_info = pd.read_csv(os.path.join(file_folder, 'patient_information'), header=0, index_col=[0])
    df_chart = pd.read_csv(os.path.join(file_folder, 'patient_chartdata'), header=0, index_col=[0])
    df_chart['charttime'] = pd.to_datetime(df_chart['charttime'])

    selected_items = np.load(os.path.join(file_folder, 'selected_items.npy'), allow_pickle=True).item()
    # item_vital = selected_items['vital']
    # item_resp = selected_items['resp']
    # item_lab = selected_items['lab']
    # item_treat = selected_items['treatment']
    # item_pain = selected_items['pain']
    return df_info, df_chart, vital_ref_dict, selected_items


def get_unixtime(dt64):
    # convert numpy.datetime64 to timestamp
    return dt64.astype('datetime64[s]').astype('int')


def locate_data_nearest_charttime(data, iid, charttime):
    try:
        val = data[(data['itemid']==iid)&(data['charttime']==charttime)]['valuenum'].item()
    except:
        df = data[data['itemid']==iid]

        _charttime = df.loc[[(abs(pd.to_datetime(df['charttime'])-charttime.to_datetime64())).idxmin()]]['charttime'].item()
        val = data[(data['itemid']==iid)&(data['charttime']==_charttime)]['valuenum'].item()
    return val, charttime


def align_datetime(data):
    # align charttime from different admissions to the same start date for comparison
    # data: pd.Dataframe
    ref_ct = np.datetime64('2020-01-01')
    ct = pd.to_datetime(data['charttime']).copy().values
    ct = ct - ct[0].astype('datetime64[D]') + ref_ct
    return ct


def interpolate_charttime_data(data, freq='1H', method='fillin', begin=None, end=None):
    # interpolate chart event to be temporal evenly distributed
    # data: pd.Dataframe
    # iid: itemid of selected source
    # freq: interpolation interval
    x_dt = data['charttime'].values
    x = get_unixtime(x_dt)
    y = data['valuenum'].values

    if method == 'interp':
        # f_interp = interp1d(x, y, kind='previous', fill_value=np.nan, bounds_error=False)
        f_interp = interp1d(x, y, kind='nearest', fill_value='extrapolate', bounds_error=False)
        start = pd.to_datetime(x_dt[0])
        delta_h = pd.Timedelta(hours=start.minute//30)
        start = start.replace(hour=start.hour, minute=0, second=0, microsecond=0) + delta_h
        x_dt_ = np.array(pd.date_range(start, x_dt[-1], freq=freq).to_pydatetime(), dtype=np.datetime64)
        x_ = get_unixtime(x_dt_)
        y_ = f_interp(x_)

    elif method == 'fillin':
        if begin is None:
            begin = pd.to_datetime(x_dt[0])
            begin = begin.replace(hour=begin.hour, minute=0, second=0, microsecond=0)
        if end is None:
            end = pd.to_datetime(x_dt[-1])

        df_ = pd.DataFrame(columns=['charttime', 'valuenum'])
        df_['charttime'] = pd.date_range(begin, end, freq=freq)
        df_['valuenum'] = np.nan

        for ct in df_['charttime']:
            dt = pd.Timedelta(hours=1)
            val = data.loc[(data['charttime'] >= ct) & (data['charttime'] < ct+dt), 'valuenum'].mean()
            df_.loc[df_['charttime']==ct, 'valuenum'] = val
        x_dt_ = df_['charttime'].to_numpy()
        y_ = df_['valuenum'].to_numpy()

    return x_dt_, y_


def fillna(df):
    for iid in df['itemid'].unique():
        df.loc[df['itemid']==iid, 'valuenum'] = df.loc[df['itemid']==iid, 'valuenum'].fillna(method='ffill')
        df.loc[df['itemid']==iid, 'valuenum'] = df.loc[df['itemid']==iid, 'valuenum'].fillna(method='bfill')
    return df


def interpolate_charttime_df(df, freq='1H', method='fillin', begin=None, end=None, iids=None):
    if iids == None:
        iids = df['itemid'].unique()

    df_interp = pd.DataFrame(columns=['hadm_id', 'charttime', 'itemid', 'valuenum'])
    for adm in df['hadm_id'].unique():
        for iid in iids:
            df_ = pd.DataFrame(columns=df_interp.columns)
            data = df[(df['hadm_id']==adm) & (df['itemid']==iid)]

            x_dt_, y_ = interpolate_charttime_data(data, freq, method=method, begin=begin, end=end)

            df_['charttime'] = x_dt_
            df_['valuenum'] = y_
            df_['hadm_id'] = adm
            df_['itemid'] = iid

            df_interp = pd.concat((df_interp, df_), ignore_index=True)
            df_interp['charttime'] = pd.to_datetime(df_interp['charttime'])

    return df_interp


def calculate_smoothed_seasonal(stl, method='lowess', frac=.25, loess_deg=2):
    # smooth seasonality component with Lowess algorithm
    # stl: statsmodel.tsa.seasonal.STL object
    x = np.array([xi for xi in range(len(stl.observed))])
    # frac = 5 / len(x)
    if method == 'lowess':
        # if frac > 0.5:
        #     frac = 0.5
        seasonal_smoothed = sm.nonparametric.lowess(stl.seasonal.values, x, frac=frac)
        return seasonal_smoothed[:,1]

    elif method == 'loess':
        seasonal_loess_x, seasonal_loess_y, seasonal_loess_w = loess_1d(x, stl.seasonal.values,
                                                                        xnew=None, degree=loess_deg, frac=frac,
                                                                        npoints=None, rotate=False, sigy=None)
        return seasonal_loess_y



def get_data_near_event(data, event_time, toi_prev=3, toi_post=3, fillna=None):
    # select subsequence of time series data around the onset of a discrete event
    # toi_prev: time of interest before event in hour
    # toi_post: time of interest after event in hour
    # delta_prev = np.timedelta64(toi_prev, 'h')
    # delta_post = np.timedelta64(toi_post, 'h')
    # begin = event_time - delta_prev
    # end = event_time + delta_post
    event_time = pd.to_datetime(event_time) - pd.Timedelta(minutes=1)
    toi_prev = pd.Timedelta(hours = toi_prev-1)
    toi_post = pd.Timedelta(hours = toi_post)
    # begin = event_time.replace(hour=event_time.hour, minute=0, second=0, microsecond=0) - toi_prev
    # end = event_time.replace(hour=event_time.hour, minute=0, second=0, microsecond=0) + toi_post
    begin = event_time.replace(hour=event_time.hour, minute=0, second=0, microsecond=0) - toi_prev
    end = event_time + toi_post

    df_ = pd.DataFrame(columns=['charttime', 'valuenum'])
    df_['charttime'] = pd.date_range(begin, end, freq='1H')
    df_['valuenum'] = np.nan
    if type(fillna) == int or type(fillna) == float:
        df_['valuenum'] = fillna

    data = data[(data['charttime']>=begin) & (data['charttime']<=end)]
    if data.shape[0] == 0:
        return df_
    # elif data.shape[0] == 1:
    #     data_ct = pd.to_datetime(data['charttime'].item())
    #     data_ct = data_ct.replace(hour=data_ct.hour, minute=0, second=0, microsecond=0) \
    #               + pd.Timedelta(hours=data_ct.minute//30)
    #     df_.loc[df_['charttime']==data_ct, 'valuenum'] = data['valuenum'].item()
    #     return df_
    #
    # ct_, data_ = interpolate_charttime_data(data)
    # if pd.isnull(data_).any():
    #     print(11111)
    # if pd.isnull(data_).all():
    #     print(22222)
    # for ct, d in zip(ct_, data_) :
    #     df_.loc[df_['charttime']==ct, 'valuenum'] = d

    else:
        for ct in df_['charttime']:
            dt = pd.Timedelta(hours=1)
            val = data.loc[(data['charttime'] >= ct) & (data['charttime'] < ct+dt), 'valuenum'].mean()
            df_.loc[df_['charttime']==ct, 'valuenum'] = val

    # if fillna == 'neighbor':
    #     df_ = fillna(df_)
    #     # df_ = df_.fillna(method='ffill')
    #     # df_ = df_.fillna(method='bfill')
    return df_


def normalize_df(df, type=None, return_param=True, selected_itemids=None, item_dict=item_dict):
    if type == 'info':
        df_normalized = df.copy()

        age_max = df_normalized['age'].max()
        age_min = df_normalized['age'].min()
        df_normalized['age'] = (df_normalized['age'] - age_min) / (age_max - age_min)

        weight_max = df_normalized['admission_weight'].max()
        weight_min = df_normalized['admission_weight'].min()
        df_normalized['admission_weight'] = (df_normalized['admission_weight'] - weight_min) / (weight_max - weight_min)
        normalize_param = {
            'age': [age_min, age_max],
            'weight': [weight_min, weight_max],
        }

        df_normalized.gender = pd.Categorical(df_normalized.gender)
        df_normalized.admission_type = pd.Categorical(df_normalized.admission_type)
        df_normalized.admission_location = pd.Categorical(df_normalized.admission_location)
        df_normalized.discharge_location = pd.Categorical(df_normalized.discharge_location)

    elif type == 'chart':
        if selected_itemids is None or item_dict is None:
            raise AssertionError('Need "selected_itemids" and "item_dict" for chartdata normalization.')

        df_normalized = df.copy()
        normalize_param = {}
        for iid in selected_itemids:
            if item_dict[item_dict['itemid']==iid]['param_type'].item() == 'Text':
                cat_item = pd.Categorical(df_normalized[df_normalized['itemid'] == iid]['value'])
                df_normalized.loc[df_normalized['itemid'] == iid, 'valuenum'] = cat_item.codes
            else:
                normalize_param[iid] = [df_normalized[df_normalized['itemid'] == iid]['valuenum'].mean(),
                                              df_normalized[df_normalized['itemid'] == iid]['valuenum'].std()]

                d = df_normalized[df_normalized['itemid'] == iid]['valuenum']
                d = (d - normalize_param[iid][0]) / normalize_param[iid][1]
                df_normalized.loc[df_normalized['itemid'] == iid, 'valuenum'] = d

        for iid in ref_itemids:
            if iid in HR_ref_itemids:
                d = df_normalized[df_normalized['itemid'] == iid]['valuenum']
                d = (d - normalize_param[220045][0]) / normalize_param[220045][1]
            elif iid in BPs_ref_itemids:
                d = df_normalized[df_normalized['itemid'] == iid]['valuenum']
                d = (d - normalize_param[220179][0]) / normalize_param[220179][1]
            elif iid in BPd_ref_itemids:
                d = df_normalized[df_normalized['itemid'] == iid]['valuenum']
                d = (d - normalize_param[220180][0]) / normalize_param[220180][1]
            else:
                raise AssertionError(f'itemid{iid} not found in selected orthostatic vital signs.')
            df_normalized.loc[df_normalized['itemid'] == iid, 'valuenum'] = d
    else:
        raise AssertionError('Only support data type "info" or "chart".')

    if return_param:
        return df_normalized, normalize_param
    else:
        return  df_normalized


def denorm_data(data, data_source, normalize_param, iid=None):
    if data_source in ['age', 'weight']:
        return data * (normalize_param[data_source][1] - normalize_param[data_source][0]) + normalize_param[data_source][0]
    elif data_source in ['chart']:
        if iid == None:
            raise AssertionError('Need "itemid" to de-normalize chartdata.')
        return data * normalize_param[iid][1] + normalize_param[iid][0]
    else:
        raise AssertionError(f'Input data source: {data_source}. Supported data source: "age", "weight", "chart"')


def detect_sudden_change_anomaly(x, y, margin=0.3, plot=False):
    # x: 1d array
    diff = y[1:] - y[:-1]
    diff2 = diff[1:] * diff[:-1]
    thres = margin * (np.nanmax(y) - np.nanmin(y))
    height_thres = np.nanmin(y) + 0.8 * (np.nanmax(y) - np.nanmin(y))
    sudden_changes = np.where(abs(diff) > thres)[0] + 1
    # peak_points = np.where(diff2 < 0)[0] + 1
    # peak_points = np.intersect1d(peak_points, sudden_changes)
    # peak_points = np.array([idx for idx in peak_points if (abs(diff[idx-1]) > .5 * thres) and (abs(diff[idx]) > .5 * thres)])
    peak_points = find_peaks(y, height=height_thres, threshold=0.5*thres, distance=2)[0]
    # sudden_changes = np.setdiff1d(sudden_changes, peak_points, assume_unique=True)
    sudden_changes = np.setdiff1d(sudden_changes, np.union1d(np.union1d(peak_points, peak_points-1), peak_points+1))

    if plot:
        fig = plt.figure()
        plt.scatter(x, y, c='gray')
        if sudden_changes.size:
            plt.scatter(x[sudden_changes], y[sudden_changes], c='g', label='sudden changes')
        if peak_points.size:
            plt.scatter(x[peak_points], y[peak_points], c='r', marker='^', label='peaks')
        plt.legend()
        plt.show()

    return {
        'sudden_changes': sudden_changes,
        'peaks': peak_points,
    }

