import numpy as np
import neurolab as nl
import matplotlib.pyplot as mp
data = np.loadtxt('lvq.txt')
x, y = data[:, :2], data[:, 2:]
print(y)
labels = []
for row in y:
    row = row.astype(int).astype(str)
    labels.append('.'.join(row))
label_set = np.unique(labels)
codes = []
for label in labels:
    code = np.where(label_set == label)[0][0]
    codes.append(code)
codes = np.array(codes)
l, r, h = x[:, 0].min() - 1, x[:, 0].max() + 1, 0.01
b, t, v = x[:, 1].min() - 1, x[:, 1].max() + 1, 0.01
model = nl.net.newlvq(nl.tool.minmax(x), 10,
                      [0.25, 0.25, 0.25, 0.25])
error = model.train(x, y, epochs=100, goal=-1)
grid_x = np.meshgrid(np.arange(l, r, h), np.arange(b, t, v))
grid_y = model.sim(np.c_[grid_x[0].ravel(), grid_x[1].ravel()])
grid_labels = []
for grid_row in grid_y:
    grid_row = grid_row.astype(int).astype(str)
    grid_labels.append('.'.join(grid_row))
grid_codes = []
for grid_label in grid_labels:
    grid_code = np.where(label_set == grid_label)[0][0]
    grid_codes.append(grid_code)
grid_codes = np.array(grid_codes).reshape(grid_x[0].shape)
pred_y = model.sim(x)
print(pred_y)
mp.figure().set_facecolor(np.ones(3) * 240 / 255)
mp.title('Learning Vector Quantization', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x[0], grid_x[1], grid_codes, cmap='brg')
mp.xlim(grid_x[0].min(), grid_x[0].max())
mp.ylim(grid_x[1].min(), grid_x[1].max())
mp.scatter(x[:, 0], x[:, 1], c=codes, cmap='RdYlBu', s=80)
mp.figure().set_facecolor(np.ones(3) * 240 / 255)
mp.title('Training Progress', fontsize=20)
mp.xlabel('Number Of Epochs', fontsize=14)
mp.ylabel('Training Error', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(error, 'o-', c='orangered', label='Error')
mp.legend()
mp.show()
