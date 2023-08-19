import numpy as np
import tensorflow as tf
from models import MyModel_TransLTEE  # Please replace with your actual model
from config import Config
import time
# from sklearn.model_selection import train_test_split

def main():
    # Load and process the data
    t0=5
    config = Config()
    model = MyModel_TransLTEE(t0)
    
    # for j in range(1, 11):
    TY = np.loadtxt('./test.csv', delimiter=',')
    matrix = TY[:, 1:]
    N = TY.shape[0]
    
    ts = np.reshape(TY[:, 0], (N, 1))
    out_treat = np.loadtxt('./y.txt', delimiter=',')
    ys = out_treat[:, :t0+1]

    x, y, t = matrix, ys, ts
    n_train = x.shape[0]
    print(x.shape, y.shape, t.shape, n_train)
    # _ = model(x,t,y[:,:-1],y[:,1:])

    print('COMPILING MODEL WITH THE OPTIMIZER AND LOSS FUNCTION...')
    print('OPTIMIZER AND LOSS FUNCTION SUCCESSFULLY COMPILED')
    print('TRAINING MODEL...')

    ## Gradient Descent
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config.lr, decay=config.weight_decay)
    sched = tf.keras.optimizers.schedules.ExponentialDecay(config.lr, decay_steps=3, decay_rate=0.5)
    loss_func = tf.keras.metrics.CategoricalCrossentropy()
    accuracy_func = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # predictions, predict_error, distance = model.call(
    #             x, t, y[:,:-1], y[:,1:])
    # print(predictions.shape, predict_error.shape, distance.shape)
    
    def train_step(X_train, t_train, tar_train, tar_real, gama = 1e-4):

        with tf.GradientTape() as tape:
            predictions, predict_error, distance = model.call(
                X_train, t_train, tar_train, tar_real)
            
            CF_predictions, _, _ = model.call(
                X_train, (1-t_train), tar_train, tar_real)
            # model.build(input_shape=[N, TY.shape[1]])
            # loss_groundtruth = train_loss(abs(CF_predictions), groundtruth)
            # loss = predict_error + loss_groundtruth + gama*distance
            pred_effect = tf.reduce_mean(abs(predict_error-CF_predictions),axis=0)
            # print(pred_effect)
            loss = predict_error + gama*distance

        gradients = tape.gradient(loss, model.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_func(loss)
        # train_accuracy(pred_effect, groundtruth)
        accuracy_func(tar_real, predictions)
        # print([CF_predictions])

    ## Training
    for epoch in range(10):
        loss_func.reset_states()
        accuracy_func.reset_states()
        start = time.time()

        # tar_batch = tf.expand_dims(y_train[:,:-1],-1)
        # tar_real_batch = tf.expand_dims(y_train[:, 1:],-1)        

        # tf.keras.Model.compile(model,optimizer='Adam',
        #                        loss='sparse_categorical_crossentropy',
        #                        metrics=['accuracy'])
        # tf.keras.Model.fit(MyModel_TransLTEE,)
        train_step(x, ts, y[:,:-1], y[:,1:])

        # Timing and printing happens here after each epoch
        epoch_time = time.time() - start
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {epoch_time:.2f} secs\n')

    print('MODEL SUCCESSFULLY TRAINED!')



if __name__ == '__main__':
    main()