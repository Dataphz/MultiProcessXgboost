import sys
import time
#sys.path.append("/disk2/Shengli/project/")
from data_prepare.point_to_label.data_util_shallow import *
import numpy as np
import csv
import xgboost as xgb
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
plt.switch_backend('agg')

""" generate main filepath """
filepath = os.getcwd()
mainpath = filepath

def check_folder(filepath):
    """
    check whether filepath exists.
    """
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath

def xgboost_search_params():
    xtype="binary_xgboost"
    
    train_data, validation_data, test_data = get_input(paras = {'norm':'GN','ts':False})
    optimal_ap = 0.0
    optimal_params = {}
    eta = [0.01, 0.05, 0.2]
    _lambda = [0.01, 0.03, 0.05, 0.1]
    depth = [3, 6, 8, 10, 12]
    min_child_weight = [1,3,5]
    sub_sample = [0.7, 0.8, 0.9, 1.0]
    colsample_bytree = [0.7, 0.8, 0.9, 1.0]
    
    start_time = time.ctime()
    for e in eta:
        for l in _lambda:
            for m in min_child_weight:
                for s in sub_sample:
                    for c in colsample_bytree:
                        for d in depth:
                            init_params = {
                                    "objective":"binary:logistic",
                                    "eta":e,
                                    "max_depth":d,
                                    "silent":1,
                                    "nthread":30,
                                    "eval_metric":"map",
                                    "lambda": l,
                                    "min_child_weigh":m,
                                    "sub_sample":s,
                                    "colsample_bytree":c
                                }
                            # param = [e,d,l,m,s,c]
                            paramString = "e_{}_d_{}_l_{}_m_{}_s_{}_c_{}".format(
                                e,d,l,m,s,c)
                            model = XGBoost(init_params, paramString,start_time, xtype="binary_xgboost")
                            model.dataset_prepare(train_data, validation_data, test_data)
                            ap = model.train()
                            print("ap: {}".format(ap))
                            if optimal_ap < ap:
                                optimal_ap = ap
                                optimal_params = init_params
                                model.test_predict()

                                optimal_params["ap"] = optimal_ap
                                file = open(os.path.join(mainpath,"Results/point_to_label/{}/optimal_params.pkl".format(xtype)),'wb')  
                                pickle.dump(optimal_params,file)  
                                file.close()  

class XGBoost(object):

    def __init__(self, init_params, paramsString, start_time, xtype="binary_xgboost"):
        """
        XGBoost initialize.
        """
        self.type = xtype
        self.time = start_time
        self.init_params = init_params

        """ 测试结果保存路径 """
        self.test_result_filepath = check_folder(os.path.join(mainpath,"Results/point_to_label/{}/test_result/{}".format(xtype,paramsString)))
        """ 测试结果图保存路径 """
        self.test_presentation_filepath = check_folder(os.path.join(mainpath,"Results/point_to_label/{}/test_presentation".format(xtype)))
        """ 模型保存路径 """
        self.model_path = os.path.join(mainpath,"models/models_weight/{}/{}.model".format(xtype,xtype))

        
    
    def dataset_prepare(self,train_data, validation_data, test_data):
        """
        构造数据集
        """
        dataset_path = os.path.join(mainpath,"models/Shallowmodel/xgboost_data")
        check_folder(dataset_path)

        train_datafile = os.path.join(dataset_path,  "traindata_{}.buffer".format(self.time))
        eval_datafile = os.path.join(dataset_path, "evaldata_{}.buffer".format(self.time))
        test_datafile = os.path.join(dataset_path, "testdata_{}.buffer".format(self.time))

        train_x = train_data[0]
        train_y = train_data[1]
        self.val_x = validation_data[0]
        self.val_y = validation_data[1]
        self.test_x = test_data[0]
        self.test_y = test_data[1]

        if os.path.exists(train_datafile) and os.path.exists(eval_datafile):
            print('reading from existing datafile...')
            self.xg_train = xgb.DMatrix(train_datafile)
            self.xg_val = xgb.DMatrix(eval_datafile)

        else:
            
            self.xg_train = xgb.DMatrix(train_x, label=train_y)
            self.xg_val = xgb.DMatrix(self.val_x, label=self.val_y)

            self.xg_train.save_binary(train_datafile)
            self.xg_val.save_binary(eval_datafile)


    def train(self):
        """
        train binary xgboost model.
        """
        
        num_round = 480 
        if False and os.path.exists(self.model_path):
            self.model  = xgb.Booster(model_file = self.model_path)
        else:
            print("training ...")
            evallist = [(self.xg_train,'train'), (self.xg_val,'eval')]
            self.model = xgb.train(self.init_params, self.xg_train, num_round, evals=evallist)#,early_stopping_rounds=10
        val_pred = self.model.predict(self.xg_val)    
        ap = average_precision_score(self.val_y, val_pred)
        return ap

    def test_predict(self):
        print("saving and predict...")
        self.model.save_model(self.model_path)
        
        xg_test = xgb.DMatrix(self.test_x)
        pred = self.model.predict(xg_test)

        """ save predict result to csv """
        
        os.chdir(self.test_result_filepath)
        
        with open('pre_label.csv', 'w') as file:
            write = csv.writer(file)
            for i in range(len(self.test_x)):
                row = []
                row.append(pred[i])
                row.append(self.test_y[i])
                write.writerow(row)
        
        
        """ metric evaluate """
        with open('evaluate_metrics.csv', 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow(['Threshold', 'TP', 'TN', 'FP', 'FN', 'precision', 'recall', 'FDR', 'TDR'])
            for i in range(200):
                threshold = i / 199
                evaulate(threshold)
                (TP, TN, FP, FN), (precision, recall), (FPR, TPR) = calc_metrics()
                writer.writerow([threshold, TP, TN, FP, FN, precision, recall, FPR, TPR])

        """ plot PR and ROC Curve """
        evaluate_plot(self.test_result_filepath, self.test_presentation_filepath)


def convert(data, threshold):
    conv_data = []
    if float(data[0]) >= threshold:
        conv_data.append(1)
        conv_data.append(int(float(data[1])))
    else:
        conv_data.append(0)
        conv_data.append(int(float(data[1])))
    return conv_data


def evaulate(threshold, filename='pre_label.csv'):
    with open(filename) as readfile, open('process_' + filename, 'w') as writefile:
        writer = csv.writer(writefile, lineterminator='\n')
        content = csv.reader(readfile)
        for i, line in enumerate(content):
            data = convert(line, threshold)
            writer.writerow(data)


def calc_metrics(filename='process_pre_label.csv'):
    epsilon = 0.1
    TP, FP, TN, FN = (0, 0, 0, 0)
    Acc = 0
    nb = 0
    with open(filename) as file:
        content = csv.reader(file)
        for data in content:
            nb += 1
            if int(data[0]) == 1:
                if int(data[1]) == 1:
                    Acc += 1
                    TP += 1
                else:
                    FP += 1
            else:
                if int(data[1]) == 0:
                    Acc += 1
                    TN += 1
                else:
                    FN += 1
    precision = (epsilon + TP) / (epsilon + FP + TP)
    recall = (epsilon + TP) / (epsilon + TP + FN)
    FPR = (epsilon + FP) / (epsilon + FP + TN)
    TPR = (epsilon + TP) / (epsilon + TP + FN)
    return (TP, TN, FP, Acc / nb), (precision, recall), (FPR, TPR)


def evaluate_plot(result_filepath,presentation_filepath):
    num = 1
    Precision = np.empty((num, 200), dtype='float32')
    Recall = np.empty((num, 200), dtype='float32')
    FPR = np.empty((num, 200), dtype='float32')
    TPR = np.empty((num, 200), dtype='float32')
    fig = plt.figure(2,figsize=(20,10))
    fig.suptitle("XGBoost Optimal Result")

    os.chdir(result_filepath)
    j = 0
    i = 0
    with open('evaluate_metrics.csv', 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if line[0] != 'Threshold':
                Precision[i][j] = float(line[5])
                Recall[i][j] = float(line[6])
                FPR[i][j] = float(line[7])
                TPR[i][j] = float(line[8])
                j += 1

    os.chdir(presentation_filepath)

    """ plot ROC Curve """
    ax = plt.subplot(1,2,1)
    ax.plot(FPR[i, :], TPR[i, :])
    ax.set_title('ROC Curve', fontsize=15)
    ax.set_xlabel('FPR', fontsize=15)
    ax.set_ylabel('TPR', fontsize=15)
    #legend = plt.legend(loc='center left', bbox_to_anchor=(0.85, 0.3), borderpad=0.1, labelspacing=0.1)

    """ plot PR Curve """
    ax = plt.subplot(1,2,2)
    ax.plot(Recall[i, :], Precision[i, :])
    ax.set_title('PR Curve', fontsize=15)
    ax.set_xlabel('Recall', fontsize=15)
    ax.set_ylabel('Precision', fontsize=15)
    plt.savefig('XGBoost_Result.png', dpi=500)
    plt.close('all')



if __name__ == "__main__":
    params = {
            "objective":"binary:logistic",
            "eta":0.01,
            "max_depth":6,
            "silent":1,
            "nthread":30,
            "eval_metric":"auc",
            "lambda": 0.05,
            "min_child_weigh":5,
            "sub_sample":0.9,
            "colsample_bytree":0.8,
            "scale_pos_weight": 1.0
        }
    model = XGBoost(init_params = params)
    model.train()
    model.test_predict()

