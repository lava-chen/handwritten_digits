import tkinter as tk
import numpy as np
import cv2
import pickle
import random


class network(object):
    def __init__(self,sizes) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for y,x in zip(sizes[1:],sizes[:-1])]
        self.crroection_rate = 0
    
    def feedforward(self,a):
        '''
            一层一层的计算出神经网络的结果
        '''
        for b , w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    def SGD(self,training_data,epoches,mini_batch_size,eta,test_data=None):
        '''
            随机梯度下降函数
            >超参数
                epoches:整个训练数据集被遍历一次的过程
                mini_batch_size:批大小
                eta:learning_rate
        '''
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epoches):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} %".format(j+1, 100*self.evaluate(test_data)/n_test))
                self.crroection_rate = self.evaluate(test_data)/n_test
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        '''
            更新批次数据内的bias和wieght
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        '''
            对比正确率
        '''
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # 一个列表，用于存储每一层的激活值
        zs = [] # 一个列表，用于存储每一层的输入值

        # 前向传播
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z) # 存储每一层的输入值
            activation = sigmoid(z)
            activations.append(activation) # 存储每一层的激活值

        # 反向传播
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) # 计算最后一层的delta
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# 从文件加载 net 对象


class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("手写数字识别")

        # 创建画布
        self.canvas_size = 280  # 画布大小
        self.pixel_size = self.canvas_size // 28  # 每个像素的大小
        self.canvas = tk.Canvas(master, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)

        # 创建识别按钮
        self.button = tk.Button(master, text="识别数字", command=self.recognize)
        self.button.pack()

        # 创建清除按钮
        self.clear_button = tk.Button(master, text="清除", command=self.clear_canvas)
        self.clear_button.pack()

        # 创建标签来显示识别结果
        self.result_label = tk.Label(master, text="", font=('Helvetica', 24))
        self.result_label.pack()

        # 初始化画布图像
        self.image = np.zeros((28, 28), np.uint8)

        # 绘制网格
        self.draw_grid()

    def draw_grid(self):
        for i in range(28):
            for j in range(28):
                x0 = j * self.pixel_size
                y0 = i * self.pixel_size
                x1 = (j + 1) * self.pixel_size
                y1 = (i + 1) * self.pixel_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="gray")

    def paint(self, event):
        # 确定鼠标位置
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size
        
        # 在画布上绘制
        if 0 <= x < 28 and 0 <= y < 28:
            x0 = x * self.pixel_size
            y0 = y * self.pixel_size
            x1 = (x + 1) * self.pixel_size
            y1 = (y + 1) * self.pixel_size
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="black", outline="gray")
            # 更新图像数组以反映绘制
            self.image[y, x] = 255  # 在对应的像素位置设置为白色

    def recognize(self):
        # 预处理图像
        img_reshaped = np.reshape(self.image, (784, 1))
        img_normalized = img_reshaped / 255.0  # 归一化处理

        # 识别数字
        digit = self.recognize_digit(net, img_normalized)
        print(f"识别的数字是: {digit}")

        # 更新标签内容
        self.result_label.config(text=f"识别的数字是: {digit}")

    def clear_canvas(self):
        # 清空画布和图像
        self.canvas.delete("all")  # 删除画布上的所有内容
        self.image = np.zeros((28, 28), np.uint8)  # 重置图像
        self.result_label.config(text="")  # 清空结果标签文本
        self.draw_grid()  # 重新绘制网格

    def recognize_digit(self, net, img):
        img = np.reshape(img, (784, 1))
        output = net.feedforward(img)
        return np.argmax(output)

if __name__ == "__main__":
    with open('network_model.pkl', 'rb') as f:
        net = pickle.load(f)
    print("加载网络模型成功！")
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
