#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#第一步 导入库
import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)


# In[33]:


#DEMO内置函数下载数据
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# In[34]:


print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))


# In[35]:


#展示数据
print(train_data[0])


# In[36]:


#每个数据长度不同
len(train_data[0]), len(train_data[1])


# In[37]:


# 内置库的由单次映射到数字的DIC对象
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# In[38]:


decode_review(train_data[0])


# In[39]:


#注意中文可能需要padding是pre

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# In[40]:


len(train_data[0]), len(train_data[1])


# In[41]:


print(train_data[0])


# In[42]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
#第一层是将10000个单词划分成16个维度
model.add(keras.layers.Embedding(vocab_size, 16))
#使得向量长度一致，便于模型理解
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
#sigmoid输出是01区间内 
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()


# In[43]:


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[44]:


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# In[45]:


#有记录方式来训练模型verbose0没有进度条 1 一个进度条 2 每次一个进度条
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[46]:


history_dict = history.history
history_dict.keys()


# In[ ]:


# import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[1]:


plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[ ]:




