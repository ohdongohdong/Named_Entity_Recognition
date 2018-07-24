
import time
import os
import datetime

import numpy as np
import tensorflow as tf

#from Named_Entity.model.rnn import BLSTM
from Named_Entity.model.lstm_crf import BLSTM
from Named_Entity.utils.utils import check_path_exists, load_split_data, dotdict,\
                    count_params, score, logging, next_batch, real_len, idx2chn, \
                    ne_detector, rm_pad, delete_pad, get_tf_config

from tensorflow.python.platform import flags

#flags.DEFINE_string('model', 'BLSTM', 'set the model to use, BLSTM, seq2seq..')
flags.DEFINE_string('mode', 'test', 'test mode')
flags.DEFINE_string('rnn_cell', 'LSTM', 'set the rnn cell to use RNN, LSTM, GRU...')
flags.DEFINE_integer('num_layer', 3, 'set the layers of rnn')

flags.DEFINE_integer('batch_size', 256, 'set the batch size')
flags.DEFINE_integer('hidden_size', 512, 'set the hidden size of rnn cell')
flags.DEFINE_integer('embedding_size', 200, 'set the embedding size')
flags.DEFINE_integer('num_classes', 6, 'set the number of output classes')
flags.DEFINE_integer('epoch', 1, 'set the number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'set the learning rate')
flags.DEFINE_float('keep_prob', 1.0, 'set probability for dropout')
flags.DEFINE_string('data_dir', '../data', 'set the data root directory')
flags.DEFINE_string('log_dir', '../log', 'set the log directory')

FLAGS = flags.FLAGS

mode = FLAGS.mode

rnn_cell = FLAGS.rnn_cell
num_layer = FLAGS.num_layer

batch_size = FLAGS.batch_size
hidden_size = FLAGS.hidden_size
embedding_size = FLAGS.embedding_size
num_classes = FLAGS.num_classes
epoch = FLAGS.epoch
learning_rate = FLAGS.learning_rate

data_dir = FLAGS.data_dir

log_dir = FLAGS.log_dir
save_dir = os.path.join(log_dir, 'save')
result_dir = os.path.join(log_dir, 'result')
logging_dir = os.path.join(log_dir, 'logging')
check_path_exists([log_dir, save_dir, result_dir, logging_dir])

keep_prob = FLAGS.keep_prob

logfile = os.path.join(logging_dir, str(datetime.datetime.strftime(datetime.datetime.now(),
    '%Y-%m-%d %H:%M:%S') + '.txt').replace(' ', '').replace('/', ''))

input_path = os.path.join(data_dir, 'test','chosun.chn')
label_path = os.path.join(data_dir, 'test','chosun.label')
length_path = os.path.join(data_dir, 'test','chosun.len')
chn_dic_path = os.path.join(data_dir, 'chn.dic')
label_dic_path = os.path.join(data_dir, 'label.dic')


class Runner(object):
    def _default_configs(self):
        return {'mode': mode,
                'rnn_cell': rnn_cell,
                'num_layer' : num_layer,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'embedding_size': embedding_size,
                'num_classes': num_classes,
                'learning_rate': learning_rate,
                'keep_prob': keep_prob,
                }
    
    def load_data(self, args):
        return load_split_data(input_path, label_path, length_path, chn_dic_path, label_dic_path)
    
    def run(self):
        # load data
        args_dict = self._default_configs()
        args = dotdict(args_dict)
        test_input, test_label, test_len, chn_dic, label_dic = self.load_data(args)
        
        print('test input data : {}'.format(test_input.shape))
        print('test label data : {}'.format(test_label.shape)) 
         
        total_data = len(test_input) 
         
        # load model 
        max_len = test_input.shape[1]
        chn_size = len(chn_dic)

        model = BLSTM(args, max_len, chn_size, label_dic)
       
        # count the num of parameters
        num_params = count_params(model, mode='trainable')
        all_num_params = count_params(model, mode='all')
        model.config['trainable params'] = num_params
        model.config['all params'] = all_num_params
        print(model.config)  
        
        with tf.Session(graph=model.graph, config=get_tf_config()) as sess:
            ckpt = tf.train.get_checkpoint_state(save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model restored from" ' + save_dir)
 
            for each_epoch in range(epoch):
                # test
                start = time.time()
               
                batch_epoch = test_input.shape[0] / batch_size
                batch_epoch = int(batch_epoch) 
                batch_p = np.zeros(batch_epoch)
                batch_r = np.zeros(batch_epoch)
                batch_f1 = np.zeros(batch_epoch)
                for b in range(batch_epoch):
                    batch_inputs, batch_labels, batch_seq_len = next_batch(batch_size, test_input, test_label, test_len) 
                    feed = {model.inputs: batch_inputs, model.targets: batch_labels\
                                    ,model.seq_len:batch_seq_len}
                    l, pre, y = sess.run([model.loss,
                                            model.prediction, model.targets], feed_dict=feed) 
                    
                    batch_inputs = idx2chn(batch_inputs, chn_dic) 
                    batch_labels, pre = delete_pad(batch_labels, pre, max_len) 
                    batch_labels = idx2chn(batch_labels, label_dic)
                    pre = idx2chn(pre, label_dic)
                      
                    batch_ne_target = ne_detector(batch_inputs, batch_labels)
                    batch_ne_pred = ne_detector(batch_inputs, pre)
                    p, r, f1 = score(batch_ne_target, batch_ne_pred)
                    
                    if b % 100 == 0: 
    
                        batch_chn = rm_pad(batch_inputs)         
                        print('\nsentence')
                        print(batch_chn[0]) 
                        print('\ntarget')
                        print(batch_ne_target[0])
                        print('\nprediction') 
                        print(batch_ne_pred[0])
                        print('\n')
                        
                        print('\n batch: {}/{}, epoch: {}/{}, loss={},\
                                             precision={}, recall={}, f1={}'\
                            .format(b+1, batch_epoch, each_epoch+1, epoch, l, p, r, f1)) 
         
                    batch_p[b] = p * args.batch_size
                    batch_r[b] = r * args.batch_size
                    batch_f1[b] = f1 * args.batch_size
             
                end = time.time()
                delta_time = end - start
                print('Epoch ' + str(each_epoch +1) + ' needs time: ' + str(delta_time) + ' s')
           
                 
                batch_chn = rm_pad(batch_inputs)         
                with open(os.path.join(result_dir, 'result.txt'), 'w') as result:
                    for i in range(100): 
                        result.write('input\n')        
                        result.write(str(''.join(batch_chn[i])))
                        result.write('\ntarget\n')
                        result.write(str(batch_ne_target[i]))
                        result.write('\nprediction\n')
                        result.write(str(batch_ne_pred[i]))
                        result.write('\n')
                        result.write('-'*30)
                        result.write('\n')
                 
                epoch_p = batch_p.sum() / total_data
                epoch_r = batch_r.sum() / total_data
                epoch_f1 = batch_f1.sum() / total_data
 
                print('Epoch', each_epoch + 1, 'loss:', l,\
                    'precision:', epoch_p, 'recall:', epoch_r, ' f1:', epoch_f1)

                logging(model, logfile, l, epoch_p, epoch_r, epoch_f1,\
                                 each_epoch,b,batch_epoch,  delta_time, mode='test')

if __name__ == '__main__':
    runner = Runner()
    runner.run()
                    
