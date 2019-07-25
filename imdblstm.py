#!/usr/bin/env python
# coding: utf-8

##################数据预处理

import pandas as pd
#“header = 0”表示文件的第一行包含列名，“delimiter = \ t”表示字段由制表符分隔，quoting = 3告诉Python忽略双引号，否则可能会遇到错误试图读取文件。
train = pd.read_csv('labeledTrainData.tsv',header = 0,delimiter = '\t',quoting = 3)
firstreview = train['review'][0]

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords # import the stop word list
import nltk
def review_to_words(raw_review):
# 1. 移除 HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text() 
# 2. 通过正则表达式，用空格替代非a-zA-Z的单词       
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
# 3. 根据空格，将句子分离成单个单词
    words = letters_only.lower().split()                             
#4. 去除停用词
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
#5. 将单词拼接成语句并返回
    return( " ".join( meaningful_words ))
num_reviews = train['review'].size
clean_train_reviews = []
for i in range(0, num_reviews):
    clean_train_reviews.append( review_to_words( train['review'][i]))

clean_review = review_to_words(train['review'][0])

########################制作词向量

###size表示每个单词的特征数，我们选取50，sg = 1表示采取skip-gram算法，sg = 0表示CBOW算法，alpha学习率为0.025
words= []  
for i in range(0,num_reviews):
    words.append(clean_train_reviews[i].split())  
from gensim.models import word2vec
model = word2vec.Word2Vec(words,size = 50,alpha=0.025,window = 3,sg=1)
model.save("testskip-gram.model")
  
model.wv.most_similar('good')
'''Out[]: 
[('decent', 0.8644789457321167),
 ('bad', 0.8406391143798828),
 ('alright', 0.835037350654602),
 ('great', 0.8297604322433472),
 ('okay', 0.8241003751754761),
 ('passable', 0.8033417463302612),
 ('ok', 0.7996076941490173),
 ('darn', 0.7944133281707764),
 ('excellent', 0.7909718751907349),
 ('cool', 0.7908358573913574)]'''
model.wv.get_vector("good")
'''Out[]: 
array([-0.07004781,  0.21106675, -0.09964856, -0.27289432, -0.5918534 ,
    0.13783436, -1.2372327 , -0.13977386, -0.4440627 ,  0.43113962,
   -0.1683755 , -0.13584745, -0.4280578 ,  0.0668585 , -0.13680552,
    0.34245998,  0.6079611 ,  0.48834306, -0.07885542, -0.3921202 ,
   -0.6851052 , -0.8186165 , -0.06932819,  0.35389662, -0.05227594,
    0.19891539,  0.12392826, -0.22159393, -0.42917448, -0.17915241,
    0.1279066 , -0.05532034, -0.25165513,  0.08164779, -0.14381503,
    0.30917087,  0.09443554, -0.28672957,  0.07516106,  0.1655995 ,
    0.11752302,  0.58272016,  0.03357857, -0.39408216,  0.02313354,
   -0.18522774, -0.74761283,  0.39284348, -0.07226875, -0.1633621 ],
  dtype=float32)'''
  
text_lengths = [len(x.split()) for x in clean_train_reviews]
plt.pyplot.hist(text_lengths, bins=25)
plt.pyplot.title('Histogram of # of Words in Texts')

###选取250为最大句子长度，若超过250则会被截断，若不足将以0填充
###构建出每个句子的索引矩阵，其大小为25000*250 
wordindex = model.wv.index2word
maxSeqLength = 250
def getindex(reviews):
    ids =np.zeros((numFiles,maxSeqLength),dtype = 'int32')
    fileCounter = 0
    for i in reviews:
        indexCounter = 0
        for word in i:
            try:
                ids[fileCounter][indexCounter] = wordindex.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = 0
            indexCounter += 1
            if indexCounter >= maxSeqLength:
                break
        fileCounter += 1
        print(fileCounter)
    return ids
sentences  = getindex(words)

######################构建lstm模型进行训练

###导入参数，并将制作的词向量与句子的索引矩阵构建输入数据
import tensorflow as tf
from tensorflow.contrib import learn
numFiles = 25000
maxSeqLength = 250
batchSize = 36
lstmUnits = 64
numClasses = 2
iterations = 50000
numDimensions = 100 
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize,maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(model.wv.syn0,input_data)

###20000个作为训练集，5000个作为测试集，并构造辅助函数获取每个batch的数据
from random import randint
label = train['sentiment']
def getTrainBatch():
    labels = []
    arr = []
    for i in range(batchSize):
        num = randint(1,19999)
        if label[num] == 1:
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr.append(sentences[num])
    return np.array(arr), labels

def getTestBatch():
    labels = []
    arr = []
    for i in range(batchSize):
        num = randint(20000,24999)
        if label[num] == 1:
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr.append(sentences[num])
    return np.array(arr), labels

###构建LSTM模型，采用Adam优化
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
#设置参数
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
#取最终结果值
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
#定义损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

###进行训练
for i in range(iterations):
    nextBatch, nextBatchLabels = getTrainBatch();
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels}) 
 #记录损失
if (i % 100 == 0):
    nexttestBatch, nexttestBatchLabels = getTestBatch()
    trainloss_ = sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels})
    trianaccuracy_ = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
    testloss_ = sess.run(loss, {input_data: nexttestBatch, labels: nexttestBatchLabels})
    testaccuracy_ = sess.run(accuracy, {input_data: nexttestBatch, labels: nexttestBatchLabels})
    print("iteration {}/{}...".format(i+1, iterations),
          "loss {}...".format(trainloss_),
          "accuracy {}...".format(trianaccuracy_)) 
    print("iteration {}/{}...".format(i+1, iterations),
          "loss {}...".format(testloss_*100),
          "accuracy {}...".format(testaccuracy_*100)) 
    trianlosses.append(trainloss_)
    testlosses.append(testloss_)
    trianaccuracy.append(trianaccuracy_)
    testaccuracy.append(testaccuracy_)
#保存模型
if (i % 5000 == 0 and i != 0):
    save_path = saver.save(sess, "models/100pretrained_lstm.ckpt", global_step=i)
    print("saved to %s" % save_path)