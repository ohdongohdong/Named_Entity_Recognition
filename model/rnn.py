import tensorflow as tf
import numpy as np

from Named_Entity.utils.utils import real_len, weighted_loss

def build_BRNN(args, max_len, seq_len, input_embed, cell_fn):
    
    inputs = input_embed
    for i in range(args.num_layer):
        scope = 'BLSTM_' + str(i+1)
        fw_cell = cell_fn(args.hidden_size)
        bw_cell = cell_fn(args.hidden_size)

        # tensor = [batch_size, max_len, embedding_size]
        outputs, output_states =\
                        tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                        inputs=inputs,
                                                        sequence_length=real_len(inputs),
                                                        dtype=tf.float32,
                                                        scope=scope)
        output_fw, output_bw = outputs
        output_states_fw, output_states_bw = output_states
        output_fb = tf.concat([output_fw, output_bw], 2)
        shape = output_fb.get_shape().as_list()
        output_fb = tf.reshape(output_fb, [shape[0], shape[1], 2, int(shape[2] / 2)])
        hidden = tf.reduce_sum(output_fb, 2)
        output_hid = tf.contrib.layers.dropout(hidden, keep_prob=args.keep_prob,
                                                is_training=(args.mode == 'train'))
    
        if i != args.num_layer-1:
            inputs = output_hid
        else:
            outputs = tf.reshape(output_hid, [-1, args.hidden_size]) # for FC layer
            #outputs = output_hid 
    return outputs        
                                                        
class BLSTM(object):
    def __init__(self, args, max_len, chn_size, label_dic):
        self.args = args
        self.max_len = max_len
        
        if args.rnn_cell == 'RNN':
            self.cell_fn = tf.contrib.rnn.BasicRNNCell
        elif args.rnn_cell == 'LSTM':
            self.cell_fn = tf.contrib.rnn.BasicLSTMCell
        elif args.rnn_cell == 'GRU':
            self.cell_fn = tf.contrib.rnn.GRUCell
        else:
            raise Exception('rnn_cell type not supported')
    
        self.build_graph(args, max_len, chn_size, label_dic)

    def build_graph(self, args, max_len, chn_size, label_dic):
        self.graph = tf.Graph()
        with self.graph.as_default():
             
            self.inputs = tf.placeholder(tf.int64,
                                shape=(args.batch_size, max_len))   # [64, max_len]
            
            # embedding
            embedding = tf.Variable(
                            tf.random_uniform([chn_size, args.embedding_size], -1.0, 1.0))
            self.input_embed = tf.nn.embedding_lookup(embedding, self.inputs)
            
            self.targets = tf.placeholder(tf.int32,
                                shape=(args.batch_size, max_len))
            self.seq_len = tf.placeholder(tf.int32, shape=(args.batch_size))
 
            self.config = {'rnn_cell': self.cell_fn,
                            'num_layer': args.num_layer,
                            'hidden_size': args.hidden_size,
                            'num_classes': args.num_classes,
                            'learning_rate': args.learning_rate,
                            'keep_prob': args.keep_prob}
            
            model_output = build_BRNN(self.args, max_len, self.seq_len,
                                            self.input_embed, self.cell_fn)
            
            # softmax layer
            softmax_w = tf.get_variable("softmax_w", [args.hidden_size, args.num_classes])
            softmax_b = tf.get_variable("softmax_b",[args.num_classes])
            outputs_sf = tf.matmul(model_output,softmax_w) + softmax_b
            
            outputs_fc = tf.contrib.layers.fully_connected(model_output,args.num_classes)
                

            # reshape the output_fc for sequence_loss
            outputs = tf.reshape(outputs_fc, [args.batch_size, max_len, args.num_classes])

            weights = tf.ones([args.batch_size, max_len], dtype=tf.float32)
            seq_loss = tf.contrib.seq2seq.sequence_loss(\
                                logits=outputs, targets=self.targets, weights=weights)
            
            self.loss = tf.reduce_mean(seq_loss)
            
           
            #tf.summary.scalar('loss', self.loss)
            #self.merged = tf.summary.merge_all()
 
            self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
           
            self.prediction = tf.argmax(outputs, axis=2)

             
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()
         
            
            self.initial_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5,
                                        keep_checkpoint_every_n_hours=1)

