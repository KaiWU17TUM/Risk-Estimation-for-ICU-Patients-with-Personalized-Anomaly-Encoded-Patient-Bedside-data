from utils.common import *

# Train-test split based on subject_id
PID_TRAIN = pickle.load(open(os.path.join(data_processed_path, 'pid_train.pkl'), 'rb'))
PID_TEST = pickle.load(open(os.path.join(data_processed_path, 'pid_test.pkl'), 'rb'))

# Text item one-hot encoding
info_text_vals = pickle.load(
    open(os.path.join(data_path, 'text_info_onehotkey.pkl'), 'rb')
)
item_text_vals = pickle.load(
    open(os.path.join(data_path, 'text_chart_onehotkey.pkl'), 'rb')
)
item_text_vals = {
    item_dict[item_dict['itemid']==iid]['abbreviation'].item(): item_text_vals[iid]
    for iid in item_text_vals
}

# Normalization parameters
norm_param_info = pickle.load(open(os.path.join(data_processed_path, 'norm_param_info.pkl'), 'rb'))
norm_param_numeric = pickle.load(open(os.path.join(data_processed_path, 'norm_param_chart.pkl'), 'rb'))

zero_imp_info = {
    col: - norm_param_info[col]['mean'] / norm_param_info[col]['std']
    for col in norm_param_info
}
zero_imp_numeric = {
    col: - norm_param_numeric[col]['mean'] / norm_param_numeric[col]['std']
    for col in norm_param_numeric
}
zero_imp = {}
for col in zero_imp_numeric:
    zero_imp[col] = zero_imp_numeric[col]
    zero_imp[col+'_mean'] = zero_imp[col]
    zero_imp[col+'_var'] = 1.0
zero_imp = dict(zero_imp, **zero_imp_info)