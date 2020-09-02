from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time



# Example Sigmoid
# 这个类中包含了 forward 和backward函数
class Sigmoid():
    def __init__(self):
        pass

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, z):
        return self.forward(z) * (1 - self.forward(z))


## 在原 LeNet-5上进行少许修改后的 网路结构
"""
conv1: in_channels: 1, out_channel:6, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool1: in_channels: 6, out_channels:6, kernel_size = (2x2), stride=2
conv2: in_channels: 6, out_channel:16, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool2: in_channels: 16, out_channels:16, kernel_size = (2x2), stride=2
flatten
fc1: in_channel: 256, out_channels: 128, activation: relu
fc2: in_channel: 128, out_channels: 64, activation: relu
fc3: in_channel: 64, out_channels: 10, activation: relu
softmax:

tensor: (1x28x28)   --conv1    -->  (6x24x24)
tensor: (6x24x24)   --avgpool1 -->  (6x12x12)
tensor: (6x12x12)   --conv2    -->  (16x8x8)
tensor: (16x8x8)    --avgpool2 -->  (16x4x4)
tensor: (16x4x4)    --flatten  -->  (256)
tensor: (256)       --fc1      -->  (128)
tensor: (128)       --fc2      -->  (64)
tensor: (64)        --fc3      -->  (10)
tensor: (10)        --softmax  -->  (10)
"""
'''
  Simple LeNet5 by YBR
Version 3.1.4 'Oh, it worked, but WHY?'

>本质上是一个缝合怪

>整体网络架构和一些实现细节参考了Github上的这份代码
<https://github.com/toxtli/lenet-5-mnist-from-scratch-numpy/blob/master/app.py>

>卷积运算中的Im2Col操作参考了这篇知乎专栏文章
<https://zhuanlan.zhihu.com/p/63974249>

>卷积运算的Back Propagation参考了以下网页
<https://www.cnblogs.com/pinard/p/6494810.html>
<https://blog.csdn.net/ck1798333105/article/details/52369122>

>感谢我无敌强大的室友教会了我Xavier初始化和其他很多东西
'''

class ReLU():
    def __init__(self):
        self.data = None
    
    def forward(self, inputArr):
        self.data = inputArr
        self.outputArr = np.maximum(inputArr,0.0)
        return self.outputArr
    #ReLU前向传播
    
    def backward(self, inputError):
        result = np.copy(inputError)
        result[self.data <= 0] = 0
        return result
    #ReLU反向传播：麻了全麻 这咋写。
    

class convolution():
    #卷积核移动次数 = (n + 2p - f)/s + 1
    def __init__(self, inputChannel, outputChannel, kernelSize = 5, stride = 1, lr = 1.0e-3):
        self.inCh = inputChannel #输入通道数
        self.outCh = outputChannel #输出通道数
        self.kSize = kernelSize #卷积核大小
        self.stride = stride #步长
        xavierPara = np.sqrt(6/(self.inCh + self.outCh))
        self.kWeight = np.random.uniform(-xavierPara, xavierPara, (self.outCh, self.inCh, self.kSize, self.kSize))
        self.kBias = np.random.normal(0,1e-1,self.outCh)
        self.data = None
        self.rate = lr
        self.totalIterations = 0
        
    def im2Col(self, arr, kernel):
        '''
        输入矩阵；
        根据卷积核大小分成kSize*kSize列以及若干行
        '''
        (batch, channel, height, width) = arr.shape
        (kNum, kChannel, kHeight, kWidth) = kernel.shape
        outHeight = int(height - kHeight) + 1
        outWidth = int(width - kWidth) + 1
        outRow = outHeight * outWidth
        inputVec = np.zeros((batch*outHeight*outWidth, kHeight*kWidth*kChannel))
        
        kVec = np.reshape(kernel, (kNum, kHeight*kWidth*kChannel))
        kVec = kVec.T
        #重组卷积核
        
        for y in range(outHeight):
            rowStart, rowEnd = y, y + kHeight
            rowOutput = y * outHeight
            for x in range(outWidth):
                colStart, colEnd = x, x + kWidth
                inputVec[rowOutput+x::outRow, :] = arr[:,:,rowStart:rowEnd, colStart:colEnd].reshape(batch, -1)
        #重组输入矩阵
                
        return inputVec, kVec
    
    def conv(self, inputArr, kernel, bias = 0):
        batch, channel, height, width = inputArr.shape
        outCh, inCh, kHeight, kWidth = kernel.shape
        outHeight = int(height - kHeight) + 1
        inputVec, kVec = self.im2Col(inputArr, kernel)
        if bias == 1:
            result = np.dot(inputVec, kVec) + self.kBias
        else:
            result = np.dot(inputVec, kVec)
        result = np.reshape(result, (batch, result.shape[0]//batch, -1))
        result = np.reshape(result, (batch, outHeight, -1, outCh)) #BHWC
        result = np.transpose(result, (0,3,1,2)) #BCHW
        return result
    
    def forward(self, inputArr):
        self.data = inputArr
        output = self.conv(inputArr, self.kWeight, 1)
        return output
    
    def backward(self, inputError):
        self.totalIterations += 1
        if self.totalIterations % 3125 == 0:
            self.rate = self.rate*0.98
            #迫真学习率decay
        
        dBias = np.sum(inputError, (0,2,3))
        #偏置项
        

        tmpErr = inputError
        tmpIn = self.data
        tmpErr = np.transpose(tmpErr,(1,0,2,3))
        tmpIn = np.transpose(tmpIn,(1,0,2,3))
        dWeight = self.conv(tmpIn, tmpErr)
        dWeight = np.transpose(dWeight,(1,0,2,3))
        #卷积核->传入误差对输入矩阵做卷积
        
        Wrotate = np.rot90(np.rot90(self.kWeight, axes=(2,3)), axes=(2,3))
        paddedError = np.pad(inputError, ((0,0),(0,0),(self.kSize-1,self.kSize-1),(self.kSize-1,self.kSize-1)),'constant')
        Wrotate = np.transpose(Wrotate,(1,0,2,3))
        dInput = self.conv(paddedError, Wrotate)
        #输入项->卷积核旋转180°对padding后的传入误差做卷积

        self.kBias -= self.rate * dBias   
        self.kWeight -= self.rate * dWeight
        #更新权重和偏置
        
        return dInput
      
class fullyConnect():
    def __init__(self, inputDimension, outputDimension, lr = 1e-3):
        self.rate = lr
        xavierPara = np.sqrt(6/(inputDimension+outputDimension))
        self.weightArr = np.random.uniform(-xavierPara, xavierPara, (inputDimension, outputDimension))
        self.biasArr = np.random.normal(0,1e-1,(1, outputDimension)) #b(1*o)
        self.totalIterations = 0
        
    def forward(self, inputArr):
        self.data = inputArr #inputArr(1*i)
        return np.dot(self.data, self.weightArr) + self.biasArr #Output(1*o)
    
    def backward(self, inputError):
        self.totalIterations += 1
        if self.totalIterations % 3125 == 0:
            self.rate = self.rate*0.98
        #迫真学习率Decay
        
        dWeight = np.dot(self.data.T, inputError) 
        #dW(i*o):对权重矩阵的偏导
        dBias = np.sum(inputError.T, axis=1) 
        #dB(1*o):对偏置矩阵的偏导
        dInput = np.dot(inputError, self.weightArr.T) 
        #dX(1*i):对本层输入矩阵的偏导
        
        self.weightArr -= self.rate * dWeight
        self.biasArr -= self.rate * dBias
        #更新权重和偏置
        
        return dInput

class AvgPool:
    def __init__(self, size):
        self.data = None
        self.size = size
        self.avg = size**2
        
    def im2Col(self, arr, kernel, stride=2):
        (batch, channel, height, width) = arr.shape
        (kNum, kChannel, kHeight, kWidth) = kernel.shape
        outHeight = int((height - kHeight)/stride) + 1
        outWidth = int((width - kWidth)/stride) + 1
        outRow = outHeight * outWidth
        
        inputVec = np.zeros((batch*outHeight*outWidth, kHeight*kWidth*kChannel))
        
        kVec = np.reshape(kernel, (kNum, kHeight*kWidth*kChannel))
        kVec = kVec.T
        #重组卷积核
        
        for y in range(outHeight):
            rowStart, rowEnd = y*stride, y*stride + kHeight
            for x in range(outWidth):
                colStart, colEnd = x*stride, x*stride + kWidth
                rowOutput = y*outWidth + x
                inputVec[rowOutput::outRow, :] = arr[:,:,rowStart:rowEnd, colStart:colEnd].reshape(batch, -1)
                
        #重组输入矩阵      
        return inputVec, kVec
    
    def conv(self, inputArr, kernel, bias = 0, stride = 2):
        batch, channel, height, width = inputArr.shape
        outCh, inCh, kHeight, kWidth = kernel.shape
        outHeight = int((height - kHeight)/stride) + 1
        inputVec, kVec = self.im2Col(inputArr, kernel)
        if bias == 1:
            result = np.dot(inputVec, kVec) + self.kBias
        else:
            result = np.dot(inputVec, kVec)
        result = np.reshape(result, (batch, result.shape[0]//batch, -1))
        result = np.reshape(result, (batch, outHeight, -1, outCh)) #BHWC
        result = np.transpose(result, (0,3,1,2)) #BCHW
        
        return result
    
    def quickPass(self, inputArr):
        #基于卷积运算的优化pooling
        #输入矩阵变形为(B*C,1,H,W)后对每一个(H,W)做步长为2固定权重的卷积
        batch0, channel0, height0, width0 = inputArr.shape
        height0 = height0 // self.size
        width0 = width0 // self.size
        #保存初始形状，计算pooling后的H,W
        
        inputArr = np.reshape(inputArr,(inputArr.shape[0]*inputArr.shape[1],inputArr.shape[2],-1))
        inputArr = np.expand_dims(inputArr, 1)
        batch, channel, height, width = inputArr.shape
        #输入矩阵reshape
        
        avgKernel = np.full((channel, channel, self.size, self.size), 1/self.avg)
        #生成'卷积核'
        
        result = self.conv(inputArr, avgKernel)
        result = np.reshape(result, (batch0,channel0,height0,width0))
        
        return result
        
    def forward(self, inputArr):
        (batch, channel, height ,width) = inputArr.shape
        outputHeight = int(height/self.size)
        outputWidth = int(width/self.size)
        
        result = np.zeros((batch, channel, outputHeight, outputWidth))
        
        for b in range(batch):        
            for c in range(channel):
                for h in range(outputHeight):
                    for w in range(outputWidth):
                        result[b,c,h,w] = np.mean(inputArr[b, c, h*self.size:(h+1)*self.size, w*self.size:(w+1)*self.size])
        return result
    
    def backward(self, inputError):
        return (inputError/self.avg).repeat(2,-2).repeat(2,-1)
    
    
class softmax():
    def __init__(self):
        pass
    
    def forward(self, inputArr):
        maxVal = np.max(inputArr, axis=1)
        maxVal = maxVal.reshape(maxVal.shape[0], 1)
        Y = np.exp(inputArr - maxVal)
        #减去最大值防止指数函数上溢出
        
        result = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        return result

class LeNet(object):
    def __init__(self):
        '''
        初始化网路，在这里你需要，声明各Conv类， AvgPool类，Relu类， FC类对象，SoftMax类对象
        并给 Conv 类 与 FC 类对象赋予随机初始值
        注意： 不要求做 BatchNormlize 和 DropOut, 但是有兴趣的可以尝试
        '''
        print("initialize")
        self.conv1 = convolution(1,6)    #LAYER: Conv1(1*28*28 ---> 6*24*24)
        self.actConv1 = ReLU()           #LAYER: Activation Function for Conv1
        self.avgPool1 = AvgPool(2)       #LAYER: Avg Pooling 1(6*24*24 ---> 6*12*12)
        self.conv2 = convolution(6,16)   #LAYER: Conv2(6*12*12 ---> 16*8*8)
        self.actConv2 = ReLU()           #LAYER: Activation Function for Conv2
        self.avgPool2 = AvgPool(2)       #LAYER: Avg Pooling 2(16*8*8 ---> 16*4*4)
        #FLATTEN!
        self.FC1 = fullyConnect(256,128) #LAYER: FC1(1*256 ---> 1*128)
        self.actFC1 = ReLU()             #LAYER: Activation Function for FC1
        self.FC2 = fullyConnect(128,64)  #LAYER: FC2(1*128 ---> 1*64)
        self.actFC2 = ReLU()             #LAYER: Activation Function for FC2
        self.FC3 = fullyConnect(64,10)   #LAYER: FC3(1*64 ---> 1*10)
        self.actFC3 = ReLU()             #LAYER: Activation Function for FC3
        self.softMAX = softmax()         #LAYER: Softmax
        print("initialized!")
        
    def init_weight(self):
        pass
    
    def forward(self, x):
        """前向传播
        x是训练样本， shape是 B,C,H,W
        这里的C是单通道 c=1 因为 MNIST中都是灰度图像
        返回的是最后一层 softmax后的结果
        也就是 以 One-Hot 表示的类别概率
        
        Arguments:
            x {np.array} --shape为 B，C，H，W
            z --线性层输出
            h --激活层输出
            o --池化层(如果有)输出
        """
        z1 = self.conv1.forward(x)
        h1 = self.actConv1.forward(z1)
        o1 = self.avgPool1.quickPass(h1)
        z2 = self.conv2.forward(o1)
        h2 = self.actConv2.forward(z2)
        o2 = self.avgPool2.quickPass(h2)
        #o2 = o1
        self.shapeCache = o2.shape
        flt = o2.reshape((x.shape[0],-1))
        z3 = self.FC1.forward(flt)
        h3 = self.actFC1.forward(z3)
        z4 = self.FC2.forward(h3)
        h4 = self.actFC2.forward(z4)
        z5 = self.FC3.forward(h4)
        result = self.softMAX.forward(z5)
        return result
    
    def backward(self, error, lr=1.0e-4):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        error = self.FC3.backward(error)
        error = self.actFC2.backward(error)
        error = self.FC2.backward(error)
        error = self.actFC1.backward(error)
        error = self.FC1.backward(error)
        error = error.reshape(self.shapeCache)
        error = self.avgPool2.backward(error)
        error = self.actConv2.backward(error)
        error = self.conv2.backward(error)
        error = self.avgPool1.backward(error)
        error = self.actConv1.backward(error)
        error = self.conv1.backward(error)
        
        return error
    
    def evaluate(self, x, labels):
        """
        x是测试样本， shape 是BCHW
        labels是测试集中的标注， 为one-hot的向量
        返回的是分类正确的百分比
        
        在这个函数中，建议直接调用一次forward得到pred_labels,
        再与 labels 做判断
        
        Arguments:
            x {np array} -- BCWH
            labels {np array} -- B x 10
        """
        pred = self.forward(x)
        evaluation = np.argmax(pred, axis = 1) - np.argmax(labels,axis = 1)
        #如果结果是0说明pred和labels最大值对应下标一样，判断正确
        
        evaluation = list(evaluation)      
        return evaluation.count(0)/x.shape[0]
    
    def data_augmentation(self, images):
        '''
        数据增强，可选操作，非强制，但是需要合理
        一些常用的数据增强选项： ramdom scale， translate， color(grayscale) jittering， rotation, gaussian noise,
        这一块儿允许使用 opencv 库或者 PIL image库
        比如把6旋转90度变成了9，但是仍然标签为6 就不合理了
        '''
        #鸽了
        return images
    
    def compute_loss(self, pred, labels, epoch, batch):
        error = np.copy(pred)
        yIndex = np.argmax(labels, axis = 1)
        error[np.arange(16), yIndex] -= 1 
        #交叉熵和softmax复合函数的bp相当于pred的结果在真值的元素位置减1，其余不变
        
        loss = np.sum(-np.log(pred[range(16), yIndex]))/16
        #交叉熵损失函数(希望函数没写错)
        
        print("Epoch: " + str(epoch) + "; Batch: " + str(batch) + "; Loss: " + str(loss))
        return error
    
    def fit(
        self,
        train_image,
        train_label,
        test_image = None,
        test_label = None,
        epoches = 10,
        batch_size = 16,
        lr = 1.0e-3
    ):
        sum_time = 0
        accuracies = []
        
        for epoch in range(epoches):
            
            ## 可选操作，数据增强
            train_image = self.data_augmentation(train_image)
            ## 随机打乱 train_image 的顺序， 但是注意train_image 和 test_label 仍需对应
            ## 鸽了
            '''
            # 1. 一次forward，bachward肯定不能是所有的图像一起,
            因此需要根据 batch_size 将 train_image, 和 train_label 分成: [ batch0 | batch1 | ... | batch_last]
            '''
            batch_images = [] # 请实现 step #1
            batch_labels = [] # 请实现 step #1
            for i in  range(0, len(train_image), 16):
                batchX = train_image[i : i+16]
                batchY = train_label[i : i+16]
                batch_images.append(batchX)
                batch_labels.append(batchY)
            
            last = time.time() #计时开始
            batch = 0
            for imgs, labels in zip(batch_images, batch_labels):
                '''
                这里我只是给了一个范例， 大家在实现上可以不一定要按照这个严格的 2,3,4步骤
                我在验证大家的模型时， 只会在main中调用 fit函数 和 evaluate 函数。
                2. 做一次forward，得到pred结果  eg. pred = self.forward(imgs)
                3. pred 和 labels做一次 loss eg. error = self.compute_loss(pred, labels)
                4. 做一次backward， 更新网络权值  eg. self.backward(error, lr=1e-3)
                '''
                pred = self.forward(imgs)
                error = self.compute_loss(pred, labels, epoch, batch)
                batch += 1
                self.backward(error)
                
            duration = time.time() - last
            sum_time += duration
            
            if epoch % 5 == 0:
                accuracy = self.evaluate(test_image, test_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)
                
        avg_time = sum_time / epoches
        print("Average Time:{}".format(avg_time))
        return avg_time, accuracies


