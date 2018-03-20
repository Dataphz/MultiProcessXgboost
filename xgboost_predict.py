import os
import time 
import math
import pickle
import struct
import csv
import numpy as np
import matplotlib.pyplot as plt 
import xgboost as xgb
import threading
from multiprocessing import Pool, sharedctypes
import warnings


from Configure.global_config import file_loc_gl

filepath = os.getcwd()
mainpath = filepath

def check_folder(filepath):
    """
    check whether filepath exists.
    """
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath

class FileConfig:
    """
    文件路径管理
    """

    #地震数据文件路径
    filepath_seismic = file_loc_gl.seismic_sgy_file_path_base
    #高相关性文件及路径
    corr_filepath = check_folder(os.path.join(mainpath, "data/full_train_data"))
    corr_file = "high_correlation_attrs.pkl"
    #模型路径
    model_path = os.path.join(mainpath, "models/models_weight/binary_xgboost/binary_xgboost.model")
    #统计信息路径
    max_min_mean_std_file = os.path.join(mainpath, "Results/max_min_mean_std_new.pkl")
    #预测结果保存路径
    predict_result_filepath = check_folder(os.path.join(mainpath, "Results/point_to_label/binary_xgboost/predict_result"))
    #数据保存路径
    plane_savepath = os.path.join(mainpath, "data/plane_loc")


class DataConfig:
    """
    数据配置管理
    """
    feature_nb = 76
    line_nb = 1641
    time_nb = 1251
    trace_nb = 664

    line_start = 627
    trace_start = 1189

    data_header_byte = 240
    header_nb = 3600
    data_content_byte=4

    thread_nb = 32

#全局变量
multiprocess = True
#global_all_data = np.zeros((DataConfig.line_nb, DataConfig.trace_nb, DataConfig.time_nb, DataConfig.feature_nb), dtype=np.float32)

def lines_predict(line, line_range, Xgboost):
    """
    predict [line, line+line_range] seismic data
    line: start index, >=0
    line_range: read length of line, <=1641
    """
    lines_block = 15
    corr = read_corr()
    print(corr)

    start_time = time.ctime()
    print("Begin_{}".format(line))

    with open(FileConfig.max_min_mean_std_file, 'rb') as file:
        mean_std = pickle.load(file)
    
    means = np.empty((DataConfig.feature_nb), dtype=np.float32)
    stds = np.empty((DataConfig.feature_nb), dtype=np.float32)
    files = {}


    fileindex = {}
    for feature_index, feature in enumerate(mean_std.keys()):
        fileindex[feature] = feature_index


    os.chdir(FileConfig.filepath_seismic)

    child_dir = []
    for dir_name in os.listdir(FileConfig.filepath_seismic):
        if os.path.splitext(dir_name)[1] == '':
            child_dir.append(dir_name)
    index = 0
    for file_dir in child_dir:
        dir_path = os.path.join(FileConfig.filepath_seismic, file_dir)
        for file in os.listdir(dir_path):
            filename = file.split('.')[0]
            if filename in corr:
                i = fileindex[file]
                files[i] = os.path.join(dir_path, file)
                means[i] = mean_std[file][2]
                stds[i] = mean_std[file][3]
                
                index += 1
            if index >= DataConfig.feature_nb:
                break

    assert index == DataConfig.feature_nb


    #trace_data = np.empty((DataConfig.trace_nb, DataConfig.time_nb), dtype=np.int32)
    pred = np.empty((DataConfig.line_nb, DataConfig.trace_nb, DataConfig.time_nb), dtype=np.float32)

    if multiprocess:
        def _init(arr_to_populate):
            global global_plane_data_process
            global_plane_data_process = arr_to_populate
        tmp = np.ctypeslib.as_ctypes(np.zeros((lines_block, DataConfig.trace_nb, DataConfig.time_nb, DataConfig.feature_nb), dtype=np.float32))
        shared_arr = sharedctypes.Array(tmp._type_, tmp, lock=False)
        p= Pool(processes=32, initializer=_init, initargs=(shared_arr, ))
        for times in range(line_range // lines_block):
        #多进程读取多文件 
            index_start = line + times*lines_block
            print("reading line:{}".format(index_start))
            

            iters = DataConfig.feature_nb
            thread_iters = zip(range(iters), [files[i] for i in range(iters)], \
                                [means[i] for i in range(iters)], [stds[i] for i in range(iters)], \
                                [index_start for _ in range(iters)], [lines_block for _ in range(iters)])
            
            p.map(multiprocess_read_lines, iterable=thread_iters)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pred_data = np.ctypeslib.as_array(shared_arr)
            #np.save(os.path.join(FileConfig.predict_result_filepath,"pred_data.npy"), pred_data)
            index_start = times * lines_block
            pred[index_start: index_start+lines_block, :, :] = Xgboost.lines_predict(pred_data, lines_block)
        
        # Rest lines for predict
        res_lines = line_range % lines_block
        index_start = line + line_range - res_lines
        print("reading line:{}".format(index_start))

        #def _init(arr_to_populate):
        #    global global_plane_data_process
        #    global_plane_data_process = arr_to_populate

        iters = DataConfig.feature_nb
        thread_iters = zip(range(iters), [files[i] for i in range(iters)], \
                                [means[i] for i in range(iters)], [stds[i] for i in range(iters)],\
                                [index_start for _ in range(iters)], [res_lines for _ in range(iters)])
        tmp = np.ctypeslib.as_ctypes(np.zeros((res_lines, DataConfig.trace_nb, DataConfig.time_nb, DataConfig.feature_nb), dtype=np.float32))
        shared_arr = sharedctypes.Array(tmp._type_, tmp, lock=False)
        p= Pool(processes=32, initializer=_init, initargs=(shared_arr, ))
        p.map(multiprocess_read_lines, iterable=thread_iters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pred_data = np.ctypeslib.as_array(shared_arr)
        
        pred[index_start: index_start+res_lines, :, :] = Xgboost.lines_predict(pred_data, res_lines)
        end_time = time.ctime()
        print("From {} to {}...".format(start_time,end_time))
        print("Saving .npy file")
        np.save(os.path.join(FileConfig.predict_result_filepath,"pred_lines_{}_{}.npy".format(line,line_range)), pred)
        
        header_byte = struct.pack('i', 1)
        with open('svc_prediction.sgy', 'wb') as file:
            for x in range(DataConfig.line_nb):  # DataConfig.line_nb
                for y in range(DataConfig.trace_nb):
                    for i in range(DataConfig.data_header_byte // 4):
                        file.write(header_byte)
                    for z in range(DataConfig.time_nb):
                        byte = struct.pack('!f', pred[x, y, z])
                        file.write(byte)
        end_time = time.ctime()
        print('start_time:{} and end_time:{}'.format(start_time, end_time))



def multiprocess_read_lines(args):
    """
    multi threading to read plane data.
    """
    index, file, mean, std, line_start, lines_block = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = np.ctypeslib.as_array(global_plane_data_process)
    pid = os.getpid()
    print("pid:{} is reading plane data of feature_index:{}".format(pid, index))
    #print("mean:{}, std:{}...".format(mean,std))
    start = time.time()
    plane_feature_data = read_lines_seismic_data(file, line_start, lines_block)
    data[:,:,:,index] = (plane_feature_data - mean) / std#data[:,:,index]
    end = time.time()
    print("pid:%s runs %0.2f minutes"%(pid, (end - start)/60.0))

def read_lines_seismic_data(filename, line, lines_block = 1):
    """
    read range of lines seismic feature data.
    filename: feature
    line: start index, >=0
    lines_block: read length of line, <=1641
    """
    seismic_data = np.empty((lines_block, DataConfig.trace_nb, DataConfig.time_nb), dtype=np.float32)
    with open(filename, 'rb') as file:
        file.read(DataConfig.header_nb)
        for x in range(DataConfig.line_nb):
            if x >= line and x < lines_block+line:
                for y in range(DataConfig.trace_nb):
                    file.read(DataConfig.data_header_byte)
                    temp = file.read(DataConfig.time_nb * DataConfig.data_content_byte)
                    seismic_data[x-line,y,:] = struct.unpack("!"+str(DataConfig.time_nb)+"f", temp)
                if x == (lines_block+line-1):
                    return seismic_data
            elif x < line:
                for y in range(DataConfig.trace_nb):
                    file.read(DataConfig.data_header_byte)
                    file.read(DataConfig.time_nb * DataConfig.data_content_byte)
            else:
                print("read_range_seismic_data Error: file/{}".format(filename))
                break


def read_corr():
    """
    read correlation file.
    """
    os.chdir(FileConfig.corr_filepath)
    with open(FileConfig.corr_file, 'rb') as f:
        temp = pickle.load(f)

    temp = sorted(temp.items(), key=lambda x:x[1], reverse=True)
    corr=[]
    for i in range(DataConfig.feature_nb):
        corr.append(temp[i][0].split(".")[0])

    return corr

class Xgboost_Predictor:
    """
    Predictor of Xgboost
    """
    def __init__(self):
        """
        Predictor constructer.
        """
        self.model_path = os.path.join(filepath, "models/models_weight/binary_xgboost/binary_xgboost.model")
        if os.path.exists(self.model_path):
            self.model = xgb.Booster(model_file=self.model_path)
        else:
            print("model doesn't exist! please train xgboost model.")

    def plane_predict(self):
        """
        predict plane data.
        """
        data = read_plane_data()
        np.save(os.path.join(FileConfig.plane_savepath,"plane_data.npy"), data)
        print("predict...")
        xg_data = xgb.DMatrix(data)
        pred = self.model.predict(xg_data)
        pred = np.reshape(pred, newshape=(DataConfig.line_nb, DataConfig.trace_nb))

        os.chdir(FileConfig.predict_result_filepath)
        np.save("plane_result.npy", pred)

        filename = FileConfig.plane_filepath.split("/")[-1]
        plt.imshow(pred)
        plt.savefig("{}_predict_plane.png".format(filename))

    def lines_predict(self, line_data, lines_block):
        """
        predict lines data.
        line_data: input
        lines_block: number of lines
        """
        line_data = np.reshape(line_data, newshape=(lines_block * DataConfig.trace_nb * DataConfig.time_nb, DataConfig.feature_nb))
        xg_data = xgb.DMatrix(line_data)
        pred = self.model.predict(xg_data)
        pred = np.reshape(pred, newshape=(lines_block, DataConfig.trace_nb, DataConfig.time_nb))
        return pred

