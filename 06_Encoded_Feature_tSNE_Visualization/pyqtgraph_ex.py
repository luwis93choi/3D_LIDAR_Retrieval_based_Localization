from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.examples as pe
import numpy as np
import matplotlib
from matplotlib import cm

import time

app = pg.mkQApp()

X = np.random.normal(size=1000)
Y = np.random.normal(size=1000)

win = pg.PlotWidget()
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')
win.show()

scatter = pg.ScatterPlotItem(symbol='o', size=1)
win.addItem(scatter)

ptr = 0

color_map = []
cmap = matplotlib.cm.get_cmap('rainbow')
for i in range(1000):
    color_map.append(cmap(i/1000))

n = 1000
data = np.random.normal(size=(2, n))

def update():

    global win, scatter, ptr, color_map, data, n

    pos = []

    plot_color = tuple(255 * np.array(color_map[ptr]))[:3]

    point = {'pos': data[:, ptr], 
             'pen' : {'color' : plot_color, 'width' : 5}}
    
    pos.append(point)

    scatter.addPoints(pos)

    if ptr < n-1:
        ptr += 1

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

app.exec_()
