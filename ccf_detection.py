import numpy as np
import time
from datetime import timedelta

np.random.seed(2)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras import backend as K

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.cross_validation import StratifiedKFold

import tensorflow as tf

nb_classes = 2

def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

def load_data():
    rd = np.loadtxt('creditcard.csv', delimiter=',', converters={30: lambda s: float(s.replace('\"', ''))}, skiprows=1)
    print "> Preprocessing Data"
    rds = np.split(rd, [30, 31], axis=1)

    data = rds[0]
    data_s = data[:, 29:30]
    data_s = normalize(data_s, axis=0)
    data = np.concatenate((data[:, :29], data_s), axis=1)

    data_s = data[:, :1]
    data_s = normalize(data_s, axis=0)
    data = np.concatenate((data_s, data[:, 1:]), axis=1)

    return data[:, 1:], rds[1].reshape((len(rds[1])))

def run(X_train, y_train, X_val, y_val):
    batch_size = 4
    nb_epoch = 1
    init = 'glorot_normal'

    p = np.random.permutation(len(X_train))
    X_train = X_train[p]
    y_train = y_train[p]

    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_val.shape[0], 'validation samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)

    model = Sequential()

    model.add(Dense(16, input_shape=X_train[0].shape, init=init))
    model.add(Activation('tanh'))
    model.add(Dense(8, init=init))
    model.add(Activation('tanh'))
    model.add(Dense(nb_classes, init=init, activation="softmax", name="predictions"))

    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=["acc"])
    # model.summary()

    cw = { 0: 1, 1: 577}
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_data=(X_val, Y_val), class_weight=cw)

    print "> MODEL GENERATED"
    print "> VALIDATING MODEL"


    start = time.time()

    preds = model.predict(X_val)
    preds = np.argmax(preds, axis=1)

    end = time.time()

    # print f1_score(y_val, preds, average='macro')
    cf = confusion_matrix(y_val, preds)
    print "> Genuine Transaction detected as Genuine : ", cf[0][0]
    print "> Genuine Transcation detected as Fraud   : ", cf[0][1]
    print "> Fraud Transcation detected as Genuine   : ", cf[1][0]
    print "> Fraud Transcation detected as Fraud     : ", cf[1][1]
    print ">",len(preds), "Data validated in", (end - start), "seconds"
    print ">", (1 / (end - start)) * len(preds), "validation / seconds"

    # tf.train.Saver().save(K.get_session(), "cc_w.ckpt")
    # tf.train.write_graph(K.get_session().graph.as_graph_def(), ".", "cc_model.pb", False)


if __name__ == "__main__":
    print "> Loading CSV Data"
    n_folds = 4
    data, labels = load_data()
    # data, labels = balanced_subsample(data, labels)
    skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)

    for i, (train, test) in enumerate(skf):
        print "> GENERATE MODEL"
        run(data[train], labels[train], data[test], labels[test])
        break
