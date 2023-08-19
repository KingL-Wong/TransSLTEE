import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, GRUCell
from transformers import TFBertModel, TFBertForMaskedLM
import tensorflow as tf
from utils import FullConnect, MultiHeadsAtten, risk

# from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, MultiHeadAttention

'''
num_layers is the parameter for specifying how many iterations the encoder block should 
have. d_model is the dimensionality of the input, num_heads is the number of attention heads, 
and dff is the dimensionality of the feed-forward network. The rate parameter is for the dropout rate.
'''

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadsAtten()
        self.fc = FullConnect()
    def forward(self, x):  # Added default value for mask
        atten_score = self.mha(x, x, x, Type='E')
        out = self.fc(atten_score) 
        return out

class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(TransformerDecoderBlock, self).__init__()
        self.mha = MultiHeadsAtten()
        self.fc = FullConnect()
    def call(self, y, enc_output):  
        attn1_score = self.mha(y, y, y, Type='D')  # Self attention
        # print(attn1_score.shape, enc_output.shape)
        attn2_score = self.mha(attn1_score, enc_output, enc_output, Type='D-E')  # Encoder-decoder attention
        out = self.fc(attn2_score) 
        return out

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = [TransformerEncoderBlock() 
                           for _ in range(num_layers)]
    def call(self, x):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x  # (batch_size, input_seq_len, d_model)

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.dec_layers = [TransformerDecoderBlock() 
                           for _ in range(num_layers)]
    def call(self, y, enc_output):
        for i in range(self.num_layers):
            y = self.dec_layers[i](y, enc_output)
        return y


class MyModel_TransLTEE(Model):
    def __init__(self, t0, input_dim=100, num_layers=3):
        super(MyModel_TransLTEE, self).__init__()
        self.t0 = t0
        self.regularizer = tf.keras.regularizers.l2(l2=1.0)
        self.input_phi = tf.keras.layers.Dense(input_dim, activation='relu', kernel_regularizer=self.regularizer)
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers)
        self.transformer_decoder = TransformerDecoder(num_layers=num_layers)
        self.dense = tf.keras.layers.Dense(100)
        self.linear = tf.keras.layers.Dense(1)
        # self.softmax = tf.keras.layers.Softmax()

    def call(self, x, t, tar_input, tar_real):
        # seq_len = tf.shape(phi_x)[1]
        i0 = tf.cast(tf.where(t < 1)[:,0], tf.int32)
        i1 = tf.cast(tf.where(t > 0)[:,0], tf.int32)
        # print(x.shape)
        x_0 = tf.gather(x[:,:], i0)
        x_1 = tf.gather(x[:,:], i1)
        tar_0 = self.dense(tf.expand_dims(tf.gather(tar_input[:,:], i0),-1))
        tar_1 = self.dense(tf.expand_dims(tf.gather(tar_input[:,:], i1),-1))

        phi_0 = self.input_phi(x_0)
        phi_1 = self.input_phi(x_1)

        encoded0 = self.transformer_encoder(phi_0)
        encoded0 = tf.repeat(encoded0[:, np.newaxis, :], self.t0, axis=1)    # [n, 100] -> [n, time, 100]
        decoded0 = self.transformer_decoder(tar_0, encoded0)
        output_0 = self.linear(decoded0)

        encoded1 = self.transformer_encoder(phi_1)
        encoded1 = tf.repeat(encoded1[:, np.newaxis, :], self.t0, axis=1)    # [n, 100] -> [n, time, 100]
        decoded1 = self.transformer_decoder(tar_1, encoded1)
        output_1 = self.linear(decoded1)
        encoded = tf.concat((encoded0, encoded1), axis=0)
        output = tf.concat((output_0, output_1), axis=0)

        predicted_error = risk().pred_error(output, tar_real)
        dis = risk().distance(encoded, self.t0, t)
        # # output = self.softmax(linear_output)
        # # return self.dense(encoded), self.linear(decoded), encoded, decoded, output
        return output, predicted_error, dis


# class Seq2SeqEncoder(tf.keras.layers.Layer):
#     """用于序列到序列学习的循环神经网络编码器"""
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
#         super().__init__(*kwargs)
#         # 嵌入层
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
#         self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
#             [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
#              for _ in range(num_layers)]), return_sequences=True,
#                                        return_state=True)

#     def call(self, X, *args, **kwargs):
#         # 输入'X'的形状：(batch_size,num_steps)
#         # 输出'X'的形状：(batch_size,num_steps,embed_size)
#         X = self.embedding(X)
#         output = self.rnn(X, **kwargs)
#         state = output[1:]
#         return output[0], state

# class MyModel_DHRNN(Model):
#     def __init__(self, input_dim, num_layers=7, num_heads=5, dff=50, dropout_rate=0.1):
#         super(MyModel_DHRNN, self).__init__()
#         self.input_dim = input_dim
#         self.regularizer = tf.keras.regularizers.l2(l2=1.0)
#         self.input_phi = tf.keras.layers.Dense(input_dim, activation='relu', kernel_regularizer=self.regularizer)
#         self.transformer_encoder = TransformerEncoder(num_layers=num_layers)
#         self.transformer_decoder = TransformerDecoder(num_layers=num_layers)
#         self.dense = tf.keras.layers.Dense(100)
#         self.linear = tf.keras.layers.Dense(1)
#         # self.softmax = tf.keras.layers.Softmax()

#     def call(self, x, t0, t, tar_input, tar_real, training=False, mask=None, num_layers=7, num_heads=5, dff=50, dropout_rate=0.1):
#         phi_x = self.input_phi(x)
