from config import Config
import numpy as np
import tensorflow as tf

config = Config()

class FullConnect(tf.keras.Model):
    def __init__(self, epsilon = config.epsilon, dropoutrate = config.dropoutrate):
        super().__init__()
        self.fc = tf.keras.Sequential(tf.keras.layers.Dense(100,activation='relu'))
        self.dropout = tf.keras.layers.Dropout(dropoutrate)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)
    def call(self, x):
        clone_x = x
        x = self.norm(x)
        out = self.fc(x)
        out = self.dropout(out)
        fc_out = clone_x + x
        return fc_out

class MultiHeadsAtten(tf.keras.Model):
    def __init__(self, epsilon = config.epsilon, dropoutrate = config.dropoutrate):
        super().__init__()
        self.fc_Q = tf.keras.layers.Dense(config.input_dim)
        self.fc_K = tf.keras.layers.Dense(config.input_dim)
        self.fc_V = tf.keras.layers.Dense(config.input_dim)
        self.out_fc = tf.keras.layers.Dense(config.input_dim)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.dropout = tf.keras.layers.Dropout(dropoutrate)

    def call(self, Q, K, V, Type):
        n = Q.shape[0]
        clone_Q = Q
        # normalization -> FeedForward
        Q = self.fc_Q(self.norm(Q))
        K = self.fc_K(self.norm(K))
        V = self.fc_V(self.norm(V))
        # print(K)
        if (Type == 'E'):
            score = self.Encoder_Atten(Q, K, V)
        elif (Type == 'D'):
            score = self.Decoder_Atten(Q, K, V, LAmask=True)
        else:
            score = self.Decoder_Atten(Q, K, V, LAmask=False)
        score = clone_Q + self.dropout(self.out_fc(score))
        return score
        
    def Encoder_Atten(self, Q, K, V, num_heads=config.num_heads, LAmask=False):
        # n个样本，每个样本100个attribute，5个头一个头20个attribute
        # 输入[n,100] -> Q,K,V = [5, n, 20] -> 找样本之间的联系
        Q = tf.transpose(tf.reshape(Q,[-1,num_heads,Q.shape[1]/num_heads]),[1,0,2])
        K = tf.transpose(tf.reshape(K,[-1,num_heads,K.shape[1]/num_heads]),[1,0,2])
        V = tf.transpose(tf.reshape(V,[-1,num_heads,V.shape[1]/num_heads]),[1,0,2])
        # Q,K相乘求每个词注意力分数
        score = tf.matmul(Q, tf.transpose(K,perm=[0,2,1]))
        # score -> [5, n, n]
        if (LAmask == True):
            # tf.linalg.band_part(input, num_lower下三角保留对角数据数, num_upper上三角保留对角数据数（负数即全保留）
            mask = 1 - tf.linalg.band_part(tf.ones((tf.shape(score)[1], tf.shape(score)[1])), -1, 0)    
            score = tf.where(mask, score, -float('inf'))
        score = tf.nn.softmax(score, dim=-1)
        # 注意力分数×V  [5, n, n]*[5, n, 20] -> [5, n, 20]
        score = tf.matmul(score, V)
        # 每个头结果合并 [5, n, 20] -> [n, 100]
        score = tf.reshape(tf.transpose(score,perm=[1,0,2]),[-1,Q.shape[1]])
        return score
    
    def Decoder_Atten(self, Q, K, V, LAmask, num_heads=config.num_heads):
        # n个样本，每个样本100个attribute，5个头一个头20个attribute
        # 输入[n, time, 1] -> [n, time, 100] -> Q,K,V = [n, 5, time, 20] -> 找样本之间的联系
        n, dim = Q.shape[0], Q.shape[2]
        Q = tf.transpose(tf.reshape(Q,[n,-1,num_heads,tf.cast(dim/num_heads,tf.int32)]),[0,2,1,3])
        K = tf.transpose(tf.reshape(K,[n,-1,num_heads,tf.cast(dim/num_heads,tf.int32)]),[0,2,1,3])
        V = tf.transpose(tf.reshape(V,[n,-1,num_heads,tf.cast(dim/num_heads,tf.int32)]),[0,2,1,3])
        # Q,K相乘求每个词注意力分数
        score = tf.matmul(Q, tf.transpose(K,perm=[0,1,3,2]))
        # score -> [n, 5, time, time]
        if (LAmask == True):
            # tf.linalg.band_part(input, num_lower下三角保留对角数据数, num_upper上三角保留对角数据数（负数即全保留）
            mask = tf.linalg.band_part(tf.ones((tf.shape(score)[2], tf.shape(score)[2])), -1, 0)
            mask = tf.cast(mask, tf.bool)
            # print(mask)
            score = tf.where(mask, score, float('-inf'))
            # print(score)
        score = tf.nn.softmax(score, axis=-1)
        # 注意力分数×V  [n, 5, time, time]*[n, 5, time, 20] -> [n, 5, time, 20]
        score = tf.matmul(score, V)
        # 每个头结果合并 [n, 5, time, 20] -> [n, time, 100]
        score = tf.reshape(tf.transpose(score,perm=[0,2,1,3]),[n,-1,dim])
        return score



class risk(tf.keras.Model):
    def __init__(self):
        pass

    def pdist2sq(self, X,Y):
        """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
        C = -2*tf.matmul(X,tf.transpose(Y))
        nx = tf.reduce_sum(tf.square(X),1,True)
        ny = tf.reduce_sum(tf.square(Y),1,True)
        D = (C + tf.transpose(ny)) + nx
        return D
    
    def safe_sqrt(self, x, lbound=1e-10):
        ''' Numerically safe version of TensorFlow sqrt '''
        return tf.sqrt(tf.clip_by_value(x, lbound, np.inf)) # x in [10^-10,+∞]
    
    def wasserstein(self, seq_len, X_ls,t,p,lam=10,its=10,sq=True,backpropT=False):
        """ Returns the Wasserstein distance between treatment groups """
        D=0
        for i in range(seq_len):
            X=X_ls[:,i,:]
            it = tf.where(t>0)[:,0]
            ic = tf.where(t<1)[:,0]
            Xc = tf.gather(X,ic)
            Xt = tf.gather(X,it)
            nc = tf.cast(tf.shape(Xc)[0],tf.float32)
            nt = tf.cast(tf.shape(Xt)[0],tf.float32)
    
            ''' Compute distance matrix'''
            if sq:
                M = self.pdist2sq(Xt,Xc)
            else:
                M = self.safe_sqrt(self.pdist2sq(Xt,Xc))
    
            ''' Estimate lambda and delta '''
            M_mean = tf.reduce_mean(M)
            M_drop = tf.nn.dropout(M,10/(nc*nt))
            delta = tf.stop_gradient(tf.reduce_max(M))
            eff_lam = tf.stop_gradient(lam/M_mean)
    
            ''' Compute new distance matrix '''
            Mt = M
            row = delta*tf.ones(tf.shape(M[0:1,:]))
            col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))],0)
            Mt = tf.concat([M,row],0)
            Mt = tf.concat([Mt,col],1)
    
            ''' Compute marginal vectors '''
            a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))],0)
            b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))],0)
    
            ''' Compute kernel matrix'''
            Mlam = eff_lam*Mt
            K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
            U = K*Mt
            ainvK = K/a
    
            u = a
            for i in range(0,its):
                u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
            v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))
    
            T = u*(tf.transpose(v)*K)
    
            if not backpropT:
                T = tf.stop_gradient(T)
    
            E = T*Mt
            D += 2*tf.reduce_sum(E)
    
        return D

    def safe_sqrt(x, lbound=1e-10):
        ''' Numerically safe version of TensorFlow sqrt '''
        return tf.sqrt(tf.clip_by_value(x, lbound, np.inf)) # x in [10^-10,+∞]

    def pred_error(self, tar_out, tar_real):
        pred_y = tf.squeeze(tar_out)
        real_y = tf.cast(tf.squeeze(tar_real), tf.float32)
        # print(pred_y)
        # print(pred_y.shape, real_y.shape,pred_y.dtype, real_y.dtype)
        # print(tf.subtract(pred_y,tar_real))
        pred_error = tf.keras.losses.mean_squared_error(real_y,pred_y)
        pred_errors = tf.reduce_mean(pred_error)
        # print(pred_errors,pred_errors.shape)
        return pred_errors
    
    def distance(self, encoded, seq_len, t):
        t = tf.cast(t, tf.float32)
        p_t = np.mean(t)
        dis = self.wasserstein(seq_len,encoded,t,p_t,lam=1,its=20)
        return dis
