import pandas as pd
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Conv1D,GlobalAveragePooling1D,GlobalMaxPooling1D,MaxPooling1D
from sklearn.metrics import accuracy_score,classification_report
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding,Input
from keras.models import Model
from sklearn import metrics
import tx_data


CNN_train,BP_train,train_lable,CNN_test,BP_test,test_lable = tx_data.load_data()



max_length = 1000
# 将句子填充到最大长度 使数据长度保持一致

CNN_train = sequence.pad_sequences(CNN_train,maxlen=max_length,padding='post')

CNN_test = sequence.pad_sequences(CNN_test,maxlen=max_length,padding='post')




def TextCNN_model_1(CNN_train,BP_trian,train_lable, CNN_test,BP_test,test_lable):

    CNN_input = Input(shape=(1000,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(40, 300, input_length=1000, trainable=False)
    embed = embedder(CNN_input)

    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 6, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, 7, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=48)(cnn2)
    cnn3 = Conv1D(256, 8, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=48)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.4)(flat)

    CNN_output = drop

    model_CNN = Model(inputs=CNN_input, outputs=CNN_output)


    model_BP = Sequential()

    model_BP.add(Dense(20, input_dim=6, activation='sigmoid'))
    model_BP.add(Dense(30, activation='sigmoid'))
    model_BP.add(Dense(40, activation='sigmoid'))
    BP_input =  model_BP.input
    BP_output = model_BP.output

    model_concate = concatenate([CNN_output,BP_output])

    x = Dense(500, activation='sigmoid')(model_concate)
    x = Dense(1000, activation='sigmoid')(x)
    final_output = Dense(2, activation='softmax')(x)
    model = Model(inputs=[CNN_input, BP_input], outputs=final_output)

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


    train_labels = keras.utils.to_categorical(train_lable, num_classes=2)  # 将标签转换为one-hot编码，改变标签的维度


    model.fit([CNN_train,BP_trian], train_labels, batch_size=300, epochs=25,shuffle=True)

    test_result = model.predict([CNN_test,BP_test])  # 预测样本属于每个类别的概率
    test_labels = np.argmax(test_result, axis=1)  # 获得最大概率对应的标签
    test_predict = list(map(str, test_labels))

    train_result = model.predict([CNN_train,BP_train])
    train_labels = np.argmax(train_result, axis=1)

    train_predict = list(map(str, train_labels))


    train_lable = list(map(str, train_lable))

    test_lable - list(map(str,test_lable))

    print('\n\nCNN 1D - Train accuracy:',round(metrics.accuracy_score(train_lable,train_predict),3)) #训练数据的准确度
    print('\nCNN 1D of Training data\n',metrics.classification_report(train_lable,train_predict))
    print('\nCNN 1D - Train Confusion Matrix\n\n',pd.crosstab(np.array(train_lable),np.array(train_predict),
                        rownames=['Actuall'],colnames=['Predicted']))
    print('\nCNN 1D - Test accuracy:',round(metrics.accuracy_score(test_lable,test_predict),3))
    print('\nCNN 1D of Test data\n',classification_report(test_lable,test_predict))
    print('\nCNN 1D - Test Confusion Matrix\n\n',pd.crosstab(np.array(test_lable),np.array(test_predict),
                        rownames=['Actuall'],colnames=['Predicted']))
    json_string = model.to_json()
    open('vdata/model_architecture_concatenate.json', 'w').write(json_string)
    # save weights
    model.save_weights('vdata/model_weights_concatenate.h5')



TextCNN_model_1(CNN_train,BP_train, train_lable, CNN_test,BP_test, test_lable)