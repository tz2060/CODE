import numpy as np
import pandas as pd
from sklearn.svm import SVC
import pywt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv('418_715_5.csv', header=None) # 读取数据文件
data = data.values # 转换为数组
print(data.shape) # 查看数据维度

features = [] # 存储特征向量
for i in range(data.shape[0]): # 遍历每个片段
    cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(data[i], 'db4', level=4) # 进行四层小波分解
    feature = np.concatenate((cA4, cD4, cD3, cD2, cD1)) # 将各层系数拼接成一个向量
    features.append(feature) # 添加到特征列表中
features = np.array(features) # 转换为数组
print(features.shape) # 查看特征维度

labels = {0: '正常心搏', 1: '房性心动过速', 2: '室性心动过速'}

y = [0, 0, 1, 2, 0, 1, ...] # 存储标签列表
y = np.array(y) # 转换为数组
print(y.shape) # 查看标签维度

clf = SVC(kernel='linear') # 创建分类器
clf.fit(X_train, y_train) # 训练分类器
y_pred = clf.predict(X_test) # 预测测试集

acc = accuracy_score(y_test, y_pred) # 计算准确率
rec = recall_score(y_test, y_pred, average='macro') # 计算召回率
f1 = f1_score(y_test, y_pred, average='macro') # 计算F1分数
cm = confusion_matrix(y_test, y_pred) # 计算混淆矩阵
fpr, tpr, thresholds = roc_curve(y_test, y_pred) # 计算ROC曲线
roc_auc = auc(fpr, tpr) # 计算AUC值
print('Accuracy:', acc) # 打印准确率
print('Recall:', rec) # 打印召回率
print('F1-score:', f1) # 打印F1分数
print('Confusion matrix:\n', cm) # 打印混淆矩阵
plt.figure() # 创建图形
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) # 绘制ROC曲线
plt.plot([0, 1], [0, 1], 'k--') # 绘制对角线
plt.xlim([0.0, 1.0]) # 设置x轴范围
plt.ylim([0.0, 1.05]) # 设置y轴范围
plt.xlabel('False Positive Rate') # 设置x轴标签
plt.ylabel('True Positive Rate') # 设置y轴标签
plt.title('Receiver operating characteristic') # 设置标题
plt.legend(loc="lower right") # 设置图例位置
plt.show() # 显示图形

"""import sys
import time
import numpy as np
import pywt
from sklearn.svm import SVC
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLCDNumber
from PyQt5.QtCore import QTimer

# 创建分类器并加载训练好的模型（假设已经保存在model.pkl文件中）
clf = SVC(kernel='linear')
clf.load('model.pkl')

# 创建一个字典，存储不同类型的心律失常的名称和颜色
labels = {0: ('正常心搏', 'green'), 1: ('房性心动过速', 'yellow'), 2: ('室性心动过速', 'red')}

# 创建一个函数，用于读取心电图数据文件，并进行小波变换和预测
def read_and_predict():
    global data # 声明全局变量data，用于存储心电图数据
    global index # 声明全局变量index，用于记录当前读取的行数
    global label # 声明全局变量label，用于显示预测结果
    global timer # 声明全局变量timer，用于控制定时器
    global alarm # 声明全局变量alarm，用于显示报警信息
    if index < data.shape[0]: # 如果还有数据未读取
        segment = data[index] # 读取当前行的数据，作为一个片段
        cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(segment, 'db4', level=4) # 对片段进行四层小波分解
        feature = np.concatenate((cA4, cD4, cD3, cD2, cD1)) # 将各层系数拼接成一个向量
        feature = feature.reshape(1, -1) # 将向量转换为二维数组，以便输入分类器
        y_pred = clf.predict(feature) # 使用分类器进行预测
        y_pred = int(y_pred[0]) # 将预测结果转换为整数
        name, color = labels[y_pred] # 根据预测结果获取对应的名称和颜色
        label.setText(name) # 设置标签的文本为名称
        label.setStyleSheet('color: {}'.format(color)) # 设置标签的颜色为颜色
        if y_pred != 0: # 如果预测结果不是正常心搏
            alarm.setText('警报：检测到心律失常！') # 设置报警信息为警报
            alarm.setStyleSheet('color: red') # 设置报警信息的颜色为红色
            timer.stop() # 停止定时器
            # 这里可以添加其他的报警操作，如发出声音、记录日志等
        else: # 如果预测结果是正常心搏
            alarm.setText('') # 清空报警信息
        index += 1 # 更新行数
    else: # 如果没有数据未读取
        timer.stop() # 停止定时器

# 创建一个函数，用于重置定时器和报警信息，并从头开始读取数据文件并预测
def reset():
    global data # 声明全局变量data，用于存储心电图数据
    global index # 声明全局变量index，用于记录当前读取的行数
    global timer # 声明全局变量timer，用于控制定时器
    global alarm # 声明全局变量alarm，用于显示报警信息
    data = pd.read_csv('data.csv', header=None) # 重新读取数据文件
    data = data.values # 转换为数组
    index = 0 # 重置行数为0
    timer.start(1000) # 启动定时器，每隔1秒执行一次read_and_predict函数
    alarm.setText('') # 清空报警信息

# 创建一个应用程序对象
app = QApplication(sys.argv)

# 创建一个窗口对象
window = QWidget()

# 设置窗口的标题、大小和位置
window.setWindowTitle('心律失常检测系统')
window.resize(400, 200)
window.move(300, 300)

# 创建一个标签对象，用于显示预测结果
label = QLabel('正常心搏', window)
label.move(150, 50)
label.setStyleSheet('font-size: 36px; color: green')

# 创建一个按钮对象，用于重置系统
button = QPushButton('重置', window)
button.move(150, 100)
button.clicked.connect(reset) # 将按钮的点击事件与reset函数绑定

# 创建一个液晶显示对象，用于显示当前时间
lcd = QLCDNumber(window)
lcd.move(150, 150)
lcd.display(time.strftime('%H:%M:%S')) # 显示当前时间

# 创建一个标签对象，用于显示报警信息
alarm = QLabel('', window)
alarm.move(50, 180)
alarm.setStyleSheet('font-size: 18px')

# 创建一个定时器对象，用于定时执行函数
timer = QTimer()
timer.timeout.connect(read_and_predict) # 将定时器的超时事件与read_and_predict函数绑定
timer.timeout.connect(lambda: lcd.display(time.strftime('%H:%M:%S'))) # 将定时器的超时事件与显示当前时间的函数绑定

# 显示窗口
window.show()

# 进入应用程序的主循环，并通过exit函数确保主循环安全结束
sys.exit(app.exec_())"""
