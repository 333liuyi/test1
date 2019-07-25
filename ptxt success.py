# -*- coding: utf-8 -*-

#读取pcap文件，解析相应的信息，为了在记事本中显示的方便，把二进制的信息

#import Image
from PIL import Image
#import re
import operator as op
#数学工具
import numpy as np

#np.set_printoptions(threshold = 1e6)     #  threshold表示输出数组的元素数目

file = open("hundgood.txt","r")
list = file.readlines()
lists = []

for fields in list: 

	fields=fields.strip();

	fields=fields.strip("[]");

	fields=fields.split("|");
	#提取字节
	
	# #fields=str(fields)
	# #matchObj =re.match(r'\w', fields)
	# #print(fields)
	for field in fields:

		if (op.gt(field,'0/')and op.lt(field,'fg')):
			#print ("%s"%(field));
			lists.append(field)
			#保留字节（筛选00-ff之间的16进制字节），但意外保留的时间戳作为报文分割
			
# #for i in lists:
	# #if len(i)==24: lists.append('/')
	# #print ("--------------");
	# #if matchObj:
		# #lists.append(fields)
		
m=np.zeros((5,784)) #生成一个空白（补零）二维数组保存报文字节，每行表示一条报文

j,k=-1,0 #j从-1开始用来解决一上来就存在的时间戳计数，k为字节计数
flag=1 #用于分割报文标志

#以下内容将报文不满784字节补零（纯黑），超出的舍弃
for i in range(len(lists)):
	if len(lists[i])!=24 and flag == 1:
		m[j][k]=int(lists[i],16) #16进制转换成10进制
		k = k + 1
		if k == 784:
			#j = j + 1
			#k = 0
			flag = 0
			#continue
	elif len(lists[i])==24:
		j = j + 1
		k = 0
		flag = 1
		
# #m = m[1:]
# #print(m)
# #print(lists)

ftxt = open('gceshif.txt','a') #用于记录图片名称及标签

for i in range(len(m)):
	m1=m[i].reshape([28,28]) #784字节为MNIST手写数字识别（0-9）28*28像素输入格式，不同于此、流量分类只有两种输出选项（0、1）
	img=Image.fromarray(m1)
	img=img.convert('L') #将一行数组存成灰度图，用0-255表示，255表示纯白（归一化后为1）
	#img.show()
	lt='\\'+str(2*i)+'_0.jpg' #偶数表示正常流量、奇数代表异常流量，最后随机打乱图片顺序
	ftxt.write(str(2*i)+'_0.jpg'+' 0'+'\n') #图片名+标签，用空格分隔
	img.save(r'C:\Users\Liu Yi\work\gceshif'+lt) #照片保存路径分两部分表示，因其包含变量

ftxt.close()

#import numpy as np
#生成一个数组，维度为100*100，灰度值一定比255大
#narray=np.array([range(10000)],dtype='int')
#narray=narray.reshape([100,100])

# #调用Image库，数组归一化
# #img=Image.fromarray(narray*255.0/9999)
# #m = m/255
# m1=m[42].reshape([28,28])
# #print (m1)
# img=Image.fromarray(m1)
# #img=Image.fromarray(m1*1)
# #转换成灰度图
# img=img.convert('L')
# #print(narray)
# #可以调用Image库下的函数了，比如show()
# #img.show()

# lt=42
# lt='\\'+str(lt)+'_1.jpg'
# print(lt)
# img.save(r'C:\Users\Liu Yi\work\mydata_jpg'+lt)

# #Image类返回矩阵的操作
# imgdata=np.matrix(img.getdata(),dtype='float')
# imgdata=imgdata.reshape(narray.shape[0],narray.shape[1])
# #图像归一化，生成矩阵
# nmatrix=imgdata*9999/255.0


#im=Image.open('0.png')
#a=np.asarray(im)
#print(a)