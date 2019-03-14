import os
import random
import torch
import vision
import numpy as np
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.config import JsonConfig
from glow.thops import onehot
from tqdm import tqdm
import matplotlib.pyplot as plt

import pdb
import scipy.io as sio

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

def run_z_single(graph, z, y_onehot, batch_size, num_classes):
    # z里一个一个进入的
    graph.eval()
    z = np.squeeze(z)
    z = np.expand_dims(z, 0)
    z = np.repeat(z, batch_size, 0)

    if y_onehot is None:
        y_onehot = torch.ones([batch_size], dtype=torch.long)
        y_onehot = y_onehot.cuda()
        y_onehot = onehot(y_onehot, num_classes)
        y_onehot = torch.zeros_like(y_onehot, dtype=y_onehot.dtype, device=y_onehot.device)

    else:
        y_onehot = np.repeat(y_onehot, batch_size)
        y_onehot = torch.tensor(y_onehot, dtype=torch.long)
        y_onehot = y_onehot.cuda()
        y_onehot = onehot(y_onehot, num_classes)

    x = graph(z=torch.tensor(z, dtype=torch.float).cuda(), y_onehot=y_onehot, eps_std=0.3, reverse=True)

    x = x[0].data.cpu().numpy()
    x = np.squeeze(x)
    x = np.expand_dims(x, 0)
    return x

# 以图片的形式保存结果x
def show_as_images(x):
    assert len(x.shape) == 3, 'Batch x H x W !'
    batch_size =  x.shape[0]
    fig = plt.figure()
    for i in range(batch_size):
        plt.subplot(batch_size, 1, i+1)
        plt.imshow(x[i])
    plt.savefig(os.path.join(z_dir, 'x.png'))
    return


# 主函数
hparams = JsonConfig("hparams/myo.json")
# build
graph = build(hparams, False)["graph"]

# 为这个程序设计一个gui，可以人工选择z的值，然后自动计算其对应的sEMG linear envelop是什么样子的
# 制作gui相关的东西
import tkinter as tk

window = tk.Tk()
window.title('z => x')
window.geometry(('1000x2000'))

z_shape = (16, 8, 2)
z_dims = 256 # z的特征值大小
z = np.zeros(z_dims)

# 设置对应z的值
nb_rows = 16
nb_cols = 16
row_width = 120
col_width = 50

def scan(v):
    # 输入的v根本没有用上
    # 对z进行重新幅值
    global z
    global scale_list
    for i in range(z_dims):
        z[i] = scale_list[i].get()
    
scale_list = []
for i in range(z_dims):
    row = i // nb_cols
    col = i % nb_cols

    s = tk.Scale(window, label='f' + str(i), from_=-75, to=75, orient=tk.HORIZONTAL, length=100, width=10, tickinterval=1, command=scan)
    s.place(x=row * row_width, y = col * col_width)
    scale_list.append(s)

# reset button
def reset_z():
    for s in scale_list:
        s.set(0)
    return
reset_button = tk.Button(window, text='reset value', width=20, height=2, command=reset_z)
reset_button.place(x=200, y=900) 

# 建立画布，用来显示输出的x
# 建立matplotlib Figure
fig = Figure(figsize=(5, 2), dpi=100)
# 建立canvas
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().place(x=800, y=820)

# transform button
def transform():
    global z
    global graph
    global hparams

    global fig
    global canvas

    batch_size = hparams.Train.batch_size//graph.num_device

    z_in = np.reshape(z, z_shape)
    x = run_z_single(graph, z_in, None, batch_size, hparams.Glow.y_classes)
    
    # 在fig上面绘制图像
    fig.clf()
    subplot = fig.add_subplot(111)
    subplot.imshow(np.transpose(np.squeeze(x)), vmin=0, vmax=0.5, cmap='bwr')

    subplot.axis('off')
    subplot.set_xticks([])
    subplot.set_yticks([])
    canvas.draw()

transform_button = tk.Button(window, text='transform z=> x', width=20, height=2, command=transform)
transform_button.place(x=400, y=900) 

# save button 用来存储fig的图片的
def save_image():
    global fig
    current_time = int(time.time())
    fig.savefig('./pictures/linear_envelop/p' + str(current_time) + '.svg')

save_button = tk.Button(window, text='save image', width=20, height=2, command=save_image)
save_button.place(x=0, y=900)


window.mainloop()


