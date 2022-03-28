import pandas as pd
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Conv1D,GlobalAveragePooling1D,GlobalMaxPooling1D,MaxPooling1D
from keras.datasets import imdb
from sklearn.metrics import accuracy_score,classification_report
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.losses import BinaryCrossentropy
import random
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding,Input
from keras.models import Model
from sklearn import metrics

import tx_data


from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.utils import plot_model


x_train,y_train,x_test,y_test = tx_data.load_CNN_data()


#
#
max_length = 128
# 将句子填充到最大长度400 使数据长度保持一致
#将每条样本变为相同的长度
x_train = sequence.pad_sequences(x_train,maxlen=max_length,padding='post')
print(x_train[0])
x_test = sequence.pad_sequences(x_test,maxlen=max_length,padding='post')
print('x_train.shape:',x_train.shape)
print('x_test.shape:',x_test.shape)




def TextCNN_model_1(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):


    main_input = Input(shape=(max_length,), dtype='int32')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(40, 300, input_length=max_length, trainable=False)
    embed = embedder(main_input)

    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=48)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=48)(cnn3)
    # 合并三个模型的输出向量
    print(cnn1.shape)
    print(cnn2.shape)
    print(cnn3.shape)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)

    flat = Flatten()(cnn)
    drop = Dropout(0.4)(flat)

    main_output = Dense(2, activation='softmax')(drop) #修改输出空间维度
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # main_output = Dense(2, activation='sigmoid')(drop)
    # model = Model(inputs=main_input, outputs=main_output)
    # model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    one_hot_labels = keras.utils.np_utils.to_categorical(y_train, num_classes=2)  # 将标签转换为one-hot编码，改变标签的维度
    model.fit(x_train_padded_seqs, one_hot_labels, batch_size=200, epochs=20,validation_split = 0.1,shuffle=True)
    # model.fit(x_train_padded_seqs, one_hot_labels, batch_size=20, epochs=20,  shuffle=True)
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    y_test = list(map(str, y_test))
    y_train_predict = model.predict(x_train_padded_seqs)
    y_train_predict = np.argmax(y_train_predict, axis=1)

    y_train_predict = list(map(str, y_train_predict))
    y_train = list(map(str, y_train))

    print('准确率', metrics.accuracy_score(y_test, y_predict))#返回正确分类的比例
    #metrics.accuracy_score(y_test, y_predict，normalize=False)#返回正确分类的个数
    print('精度',metrics.precision_score(y_test, y_predict,average='weighted'))
    print('召回率',metrics.recall_score(y_test, y_predict,average='weighted'))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))

    print('\n\nCNN 1D - Train accuracy:',round(metrics.accuracy_score(y_train,y_train_predict),3)) #训练数据的准确度
    print('\nCNN 1D of Training data\n',metrics.classification_report(y_train,y_train_predict))
    print('\nCNN 1D - Train Confusion Matrix\n\n',pd.crosstab(np.array(y_train),np.array(y_train_predict),
                        rownames=['Actuall'],colnames=['Predicted']))
    print('\nCNN 1D - Test accuracy:',round(metrics.accuracy_score(y_test,y_predict),3))
    print('\nCNN 1D of Test data\n',classification_report(y_test,y_predict))
    print('\nCNN 1D - Test Confusion Matrix\n\n',pd.crosstab(np.array(y_test),np.array(y_predict),
                        rownames=['Actuall'],colnames=['Predicted']))
  
    

#
TextCNN_model_1(x_train, y_train, x_test, y_test)


