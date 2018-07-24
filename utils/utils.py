import numpy as np
import pickle

import os
import tensorflow as tf

import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

def Chn2idx(chn_data):
    '''
    make chninese idx dictionary and convert chn to idx
    '''
    print('[chinese to num idx]')
    # chn dictionary
    all_chn = [] 
    for chn in chn_data:
        all_chn.extend(chn)
    
    chn_list = set(all_chn)
    print(chn_list) 

    chn_dic = {c: i+1 for i,c in enumerate(chn_list)} # for zero padding
    chn_dic['Pad'] = 0
 
    print('num of chn : {}'.format(len(chn_list)))

     
    # chn to idx 
    num_data = []
    for chn in chn_data:
        idx_data = [chn_dic[c] for c in chn]
        num_data.append(idx_data) 
    return num_data, chn_dic

def check_path_exists(path):
    '''
    check a path exists or not
    '''
    if isinstance(path, list):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
    else:
        if not os.path.exists(p):
            os.makedirs(path)

def padding(input_data, label_data, max_len):
    '''
    zero padding for same length
    '''
    print('[padding all of data]')
    
    
    # chn to idx
    input_data, chn_dic = Chn2idx(input_data)
    label_data, label_dic = Chn2idx(label_data)
    
    for i in range(len(input_data)):
        if len(input_data[i]) < max_len:
            for _ in range(max_len - len(input_data[i])):
                input_data[i].append(0)
                label_data[i].append(0) 
         
        if len(input_data[i]) != max_len or len(label_data[i]) != max_len:
            print('----------------error---------')
    
    with open('chn.dic', 'wb') as f:
        pickle.dump(chn_dic, f) 

    with open('label.dic', 'wb') as f:
        pickle.dump(label_dic, f) 

    return input_data, label_data

def load_split_data(input_path, label_path, length_path, chn_dic_path, label_dic_path):
    '''
    load and padding, split to train and test, split to batch data
    '''
    print('Load data......')   
    with open(input_path, 'rb') as f:
        input_data = pickle.load(f)
    with open(label_path, 'rb') as f:
        label_data = pickle.load(f)
    with open(length_path, 'rb') as f:
        length_data = pickle.load(f)
    with open(chn_dic_path, 'rb') as f:
        global chn_dic
        chn_dic = pickle.load(f)
 
    with open(label_dic_path, 'rb') as f:
        global label_dic
        label_dic = pickle.load(f)
    return input_data, label_data, length_data, chn_dic, label_dic

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def count_params(model, mode='trainable'):
    ''' count all parameters of a tensorflow graph
    '''
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
        raise TypeError('mode should be all or trainable.')
    print('number of '+mode+' parameters: '+str(num))
    return num

def real_len(inputs):
    '''
    real sequence length
    '''
    print(inputs.shape)
    used = tf.sign(tf.reduce_max(tf.abs(inputs), 2))
    used = tf.sign(tf.reduce_max(inputs, reduction_indices=2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def score_old(target, pred):
    '''
    calculate f1-score and acc from target and prediction
    ''' 

    batch_p = np.zeros(len(target))
    batch_r = np.zeros(len(target))
    batch_f1 = np.zeros(len(target))
 
    for i, (t,p) in enumerate(zip(target, pred)):
        p, r, f1, _ = precision_recall_fscore_support(t, p, average='macro')
        batch_p[i] = p
        batch_r[i] = r 
        batch_f1[i] = f1

    batch_p = sum(batch_p) / len(batch_p)
    batch_r = sum(batch_r) / len(batch_r)
    batch_f1 = sum(batch_f1) / len(batch_f1)
    return batch_p, batch_r, batch_f1

def score(batch_target, batch_predction):

    batch_p = np.zeros(len(batch_target))
    batch_r = np.zeros(len(batch_target))
    batch_f1 = np.zeros(len(batch_target))
   
    for i, (target, prediction) in enumerate(zip(batch_target, batch_predction)): 
        tp = 0
        for t in target:
            for p in prediction:
                if t == p:
                    tp += 1
                    break
        fp = len(prediction) - tp
        fn = len(target) - tp
       
        try: 
            precision = float(tp) / (tp + fp)
        except ZeroDivisionError:
            precision = 0.0

        try: 
            recall = float(tp) / (tp + fn)
        except ZeroDivisionError:
            recall = 0.0
        
        try: 
            f1 = 2*precision*recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        
        batch_p[i] = precision
        batch_r[i] = recall 
        batch_f1[i] = f1

    batch_p = sum(batch_p) / len(batch_p)
    batch_r = sum(batch_r) / len(batch_r)
    batch_f1 = sum(batch_f1) / len(batch_f1)
    
    return batch_p, batch_r, batch_f1

def logging(model, logfile, l, p, r, f1, epoch, b, b_epoch, delta_time=0,mode='train'):
    ''' log the cost and error rate and time while training or testing
    '''
    logfile = logfile
    if mode == 'config':
        with open(logfile, "a") as myfile:
            myfile.write('\n'+str(time.strftime('%X %x %Z'))+'\n')
            myfile.write('\n'+str(model.config)+'\n')

    elif mode == 'batch':
        with open(logfile, "a") as myfile:
            myfile.write('Epoch {}, batch: {}/{}, loss: {:.4f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(epoch+1, b+1, b_epoch, l, p, r, f1))

    elif mode =='train': 
        with open(logfile, "a") as myfile:
            myfile.write("Epoch:"+str(epoch+1)+' '+"precision:"+str(p) + ' recall: '+str(r)+" f1:"+str(f1)+'\n')
            myfile.write("Epoch:"+str(epoch+1)+' '+"train time:"+str(delta_time)+' s\n')

    elif mode == 'test':
        logfile = logfile+'_TEST'
        with open(logfile, "a") as myfile:
            myfile.write(str(model.config)+'\n')
            myfile.write(str(time.strftime('%X %x %Z'))+'\n')
            myfile.write("precision:"+str(p) + ' recall: '+str(r)+" f1:"+str(f1)+'\n')
   
 
def next_batch(batch_size, input_data, label_data, length_data):
    '''
    split all of data to batch set
    '''   
    idx = np.arange(0, len(input_data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    inputs_shuffle = [input_data[i] for i in idx]
    labels_shuffle = [label_data[i] for i in idx]
    length_shuffle = [length_data[i] for i in idx]

    return np.asarray(inputs_shuffle), np.asarray(labels_shuffle), np.asarray(length_shuffle)
     
def idx2chn(idx_data, chn_dic):
    chn_data = []
    v = list(chn_dic.values())
    for idx_sent in idx_data:
        idx2chn = []
        for idx in idx_sent:
            idx2chn.append(list(chn_dic.keys())[v.index(idx)])
        chn_data.append(idx2chn) 
    return chn_data

def ne_detector_old(chn_data, label_data):
    # each sentence 
    ne_list = []
    ne_one = []
    for i,(label,chn) in enumerate(zip(label_data,chn_data)):
        if label == 'B':
            ne_one.append(chn)
        elif label == 'M':
            ne_one.append(chn)
        elif label == 'E':
            ne_one.append(chn)
            ne_list.append(''.join(ne_one))
            ne_one = []
        elif label == 'S':
            ne_list.append(chn)
    
    return ne_list

def ne_detector(batch_sentence, batch_label):
  
    batch_ne_list = [] 
    for label_data, chn_data in zip(batch_label, batch_sentence):    
 
        ne_list = []
        ne_one = []
        for i,(label,chn) in enumerate(zip(label_data,chn_data)):
            if label == 'B':
                ne_one.append(chn)
            elif label == 'I':
                if len(ne_one) != 0:
                    ne_one.append(chn)
            elif label == 'E':
                if len(ne_one) != 0:
                    ne_one.append(chn)
                    ne_list.append(''.join(ne_one))
                    ne_one = []
            elif label == 'S':
                ne_list.append(chn)
        batch_ne_list.append(ne_list) 
    return batch_ne_list

def delete_pad(target, prediction,max_len):
    for i in range(len(target)):
        if len(target[i]) == max_len:
            break 
        pad = target[i][-1]
        for j in range(len(target[i])):
            if target[i][j] == pad:
                target[i] = target[i][:j]
                prediction[i] = prediction[i][:j]
                break
    return target, prediction 
             
def rm_pad(chn_data):
    ori_chn_list = []
    for chn_sent in chn_data:
        for i,chn in enumerate(chn_sent):
            if chn == 'Pad' or chn == 'P':
                ori_chn_sent = chn_sent[:i]
                break
            if i == len(chn_sent)-1:
                ori_chn_sent = chn_sent 
                break
        ori_chn_list.append(ori_chn_sent)
        
    return ori_chn_list

def weighted_loss(dic, target, batch_size, max_len):
    
    w = np.zeros([batch_size,max_len])
    
    for i,tl in enumerate(target):
        for j,t in enumerate(tl):
            if dic.keys()[dic.values().index(int(t))] == 'O':
                w[i][j] = 1.0

            elif dic.keys()[dic.values().index(int(t))] == 'B':
                w[i][j] = 5.0
            
            elif dic.keys()[dic.values().index(int(t))] == 'E':
                w[i][j] = 5.0

            elif dic.keys()[dic.values().index(int(t))] == 'I':
                w[i][j] = 5.0

            elif dic.keys()[dic.values().index(int(t))] == 'S':
                w[i][j] = 5.0
           
            else: 
                w[i][j] = 0.0
    return w
def get_tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth =True
    return config

