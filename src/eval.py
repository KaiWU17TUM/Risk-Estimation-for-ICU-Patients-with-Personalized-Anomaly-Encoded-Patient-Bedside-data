from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score, precision_score, roc_auc_score, auc

from utils.common_processed import *
from src.los_pred_cnn import eval_cnn_los
from src.mortality_pred_cnn import eval_cnn_mortality
from src.los_pred_xgboost_randomforest import eval_xgboost

NP_RANDOM_SEED = 0


def load_test_data():
    datafile_raw = 'data/preprocessed/data_raw.pkl'
    datafile_encoded = 'data/preprocessed/data_emb.pkl'
    df_data_raw = pickle.load(open(datafile_raw, 'rb'))
    df_data_encoded = pickle.load(open(datafile_encoded, 'rb'))

    data_test = {}

    for pred_task in ['mortality', 'los']:
        if pred_task == 'los':
            df_data_raw_ = df_data_raw[df_data_raw['mortality']==0]
            df_data_enc_ = df_data_encoded[df_data_encoded['mortality']==0]
        else:
            df_data_raw_ = df_data_raw
            df_data_enc_ = df_data_encoded

        data_test[pred_task] = {
            'raw':{
                'subject': df_data_raw_[df_data_raw_['subject_id'].isin(PID_TEST)]
            },
            'encoded':{
                'subject': df_data_enc_[df_data_enc_['subject_id'].isin(PID_TEST)]
            }
        }

    return data_test




if __name__=='__main__':
    data_test = load_test_data()

    MODELTYPE = ['CNN', 'randomforest', 'XGBoost']
    PRED_TASK = ['mortality', 'los']
    LOS = [7, 14, 21]
    SPLIT = ['subject']
    DATATYPE = ['raw', 'encoded']

    model_folders = {}
    for model_type in MODELTYPE:
        model_folders[model_type] = {}
        if model_type == 'CNN':
            for split in SPLIT:
                model_folders[model_type][split] = {}
                for datatype in DATATYPE:
                    model_folders[model_type][split][datatype] = {}
                    for pred_task in PRED_TASK:
                        if pred_task == 'mortality':
                            model_folders[model_type][split][datatype][pred_task] =\
                            f'models/CNN_LOS_PRED/{pred_task}/{split}_{datatype}_cnn_clf_380-256cnn_3-3kernel_0.0dropout_512fc_0.5dropout/version_0'
                        else:
                            model_folders[model_type][split][datatype][pred_task] = {}
                            for los in LOS:
                                model_folders[model_type][split][datatype][pred_task][los] =\
                                f'models/CNN_LOS_PRED/los{los}/{split}_{datatype}_cnn_clf_380-256cnn_3-3kernel_0.0dropout_512fc_0.5dropout_{los}day/version_0'
        else:
            model_folders[model_type] = {}
            for split in SPLIT:
                model_folders[model_type][split] = {}
                for datatype in ['avg', 'max']:
                    model_folders[model_type][split][datatype] = {}
                    for pred_task in PRED_TASK:
                        if pred_task == 'mortality':
                            model_folders[model_type][split][datatype][pred_task] =\
                            f'models/XGBoost/mortality/raw/{split}/{model_type}/{datatype}/raw_5000estimators_{datatype}ts_mortality/version_0/'
                        else:
                            model_folders[model_type][split][datatype][pred_task] = {}
                            for los in LOS:
                                model_folders[model_type][split][datatype][pred_task][los] =\
                                f'models/XGBoost/LOS/raw/{split}/{model_type}/{datatype}/raw_5000estimators_{datatype}ts_{los}day/version_0/'


    model_preds = {}
    for model_type in MODELTYPE:
        model_preds[model_type] = {}
        if model_type == 'CNN':
            for split in SPLIT:
                model_preds[model_type][split] = {}
                for datatype in DATATYPE:
                    model_preds[model_type][split][datatype] = {}
                    for pred_task in PRED_TASK:
                        if pred_task == 'mortality':
                            model_folder = model_folders[model_type][split][datatype][pred_task]
                            y_true, y_prob = eval_cnn_mortality(model_folder, data_test, pred_task)
                            y_pred = [0 if y < 0.5 else 1 for y in y_prob]
                            model_preds[model_type][split][datatype][pred_task] = {
                                'y_true': y_true,
                                'y_prob': y_prob,
                                'y_pred': y_pred,
                            }

                            precision, recall, thres = precision_recall_curve(y_true, y_prob)
                            auprc = auc(recall, precision)
                            auroc = roc_auc_score(y_true, y_prob)
                            accuracy = accuracy_score(y_true, y_pred)
                            recall = recall_score(y_true, y_pred, pos_label=1)
                            precision = precision_score(y_true, y_pred)
                            specificity = recall_score(y_true, y_pred, pos_label=0)
                            print(f'Finish: {model_type}-{split}-{datatype}-{pred_task}')
                            print(f'auroc: {auroc:.3f}\tauprc: {auprc:.3f}\tacc: {accuracy:.3f}\trecall: {recall:.3f}\tprecision: {precision:.3f}\tspecificiy: {specificity:.3f}')

                            pickle.dump(model_preds, open('models/eval_prediction.pkl', 'wb'))
                        else:
                            model_preds[model_type][split][datatype][pred_task] = {}
                            for los in LOS:
                                model_folder = model_folders[model_type][split][datatype][pred_task][los]
                                y_true, y_prob = eval_cnn_los(model_folder, data_test, pred_task)
                                y_pred = [0 if y < 0.5 else 1 for y in y_prob]
                                model_preds[model_type][split][datatype][pred_task][los] = {
                                    'y_true': y_true,
                                    'y_prob': y_prob,
                                    'y_pred': y_pred,
                                }

                                precision, recall, thres = precision_recall_curve(y_true, y_prob)
                                auprc = auc(recall, precision)
                                auroc = roc_auc_score(y_true, y_prob)
                                accuracy = accuracy_score(y_true, y_pred)
                                recall = recall_score(y_true, y_pred, pos_label=1)
                                precision = precision_score(y_true, y_pred)
                                specificity = recall_score(y_true, y_pred, pos_label=0)
                                print(f'Finish: {model_type}-{split}-{datatype}-{pred_task}-{los}')
                                print(f'auroc: {auroc:.3f}\tauprc: {auprc:.3f}\tacc: {accuracy:.3f}\trecall: {recall:.3f}\tprecision: {precision:.3f}\tspecificiy: {specificity:.3f}')

                                pickle.dump(model_preds, open('models/eval_prediction.pkl', 'wb'))
        else:
            model_preds[model_type] = {}
            for split in SPLIT:
                model_preds[model_type][split] = {}
                for datatype in ['avg', 'max']:
                    model_preds[model_type][split][datatype] = {}
                    for pred_task in PRED_TASK:
                        if pred_task == 'mortality':
                            model_folder = model_folders[model_type][split][datatype][pred_task]
                            y_true, y_prob = eval_xgboost(model_folder, data_test, pred_task)
                            y_pred = [0 if y < 0.5 else 1 for y in y_prob]
                            model_preds[model_type][split][datatype][pred_task] = {
                                'y_true': y_true,
                                'y_prob': y_prob,
                                'y_pred': y_pred,
                            }
                            print(f'Finish: {model_type}-{split}-{datatype}-{pred_task}')
                            precision, recall, thres = precision_recall_curve(y_true, y_prob)
                            auprc = auc(recall, precision)
                            auroc = roc_auc_score(y_true, y_prob)
                            accuracy = accuracy_score(y_true, y_pred)
                            recall = recall_score(y_true, y_pred, pos_label=1)
                            precision = precision_score(y_true, y_pred)
                            specificity = recall_score(y_true, y_pred, pos_label=0)
                            print(f'auroc: {auroc:.3f}\tauprc: {auprc:.3f}\tacc: {accuracy:.3f}\trecall: {recall:.3f}\tprecision: {precision:.3f}\tspecificiy: {specificity:.3f}')

                            pickle.dump(model_preds, open('models/eval_prediction.pkl', 'wb'))
                        else:
                            model_preds[model_type][split][datatype][pred_task] = {}
                            for los in LOS:
                                model_folder = model_folders[model_type][split][datatype][pred_task][los]
                                y_true, y_prob = eval_xgboost(model_folder, data_test, pred_task)
                                y_pred = [0 if y < 0.5 else 1 for y in y_prob]
                                model_preds[model_type][split][datatype][pred_task][los] = {
                                    'y_true': y_true,
                                    'y_prob': y_prob,
                                    'y_pred': y_pred,
                                }
                                print(f'Finish: {model_type}-{split}-{datatype}-{pred_task}-{los}')
                                precision, recall, thres = precision_recall_curve(y_true, y_prob)
                                auprc = auc(recall, precision)
                                auroc = roc_auc_score(y_true, y_prob)
                                accuracy = accuracy_score(y_true, y_pred)
                                recall = recall_score(y_true, y_pred, pos_label=1)
                                precision = precision_score(y_true, y_pred)
                                specificity = recall_score(y_true, y_pred, pos_label=0)
                                print(f'auroc: {auroc:.3f}\tauprc: {auprc:.3f}\tacc: {accuracy:.3f}\trecall: {recall:.3f}\tprecision: {precision:.3f}\tspecificiy: {specificity:.3f}')

                                pickle.dump(model_preds, open('models/eval_prediction.pkl', 'wb'))


    print(111)