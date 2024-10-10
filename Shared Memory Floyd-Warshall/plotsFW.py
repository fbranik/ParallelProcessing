#!/usr/bin/env python

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd

colorsList = ["dodgerblue", "coral", "darkseagreen", "orchid", "navy"]
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=colorsList)


def autolabel(rects, ax, labelFontSize=12):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}\nsec'.format(round(height, 2)),
                    xy=(rect.get_x() + 0.4 * rect.get_width(), height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=labelFontSize)


def speedupPlot(inDict, title, fileName, xLabels):
    speedup = {}
    for key, value in inDict.items():
        if key == '1threads':
            serialTime = value
            continue
        else:
            speedup[key.replace('threads', '')] = round(serialTime / value, 2)
    print(speedup)

    x = np.arange(0, 2 * (len(inDict) - 1), step=2)
    fig, ax = plt.subplots()
    fig.set_size_inches(14.5, 12.5)

    ax.plot(x, speedup.values())

    ax.set_ylabel('Speedup')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(xLabels)
    plt.savefig(fileName, bbox_inches='tight')
    plt.close()


def barPlot(xLabels, title, barList, legendLabels, fileName, width=1, ylim=[0.001, 500], labelFontSize=12):
    x = np.arange(0, 24, step=24 / len(xLabels))
    fig, ax = plt.subplots()
    fig.set_size_inches(14.5, 12.5)

    rects = []
    xAxisOffset = np.linspace(start=-width, stop=width, num=len(barList))

    for i in range(len(barList)):
        tempRects = ax.bar(x + xAxisOffset[i], barList[i], width,
                           label=legendLabels[i])
        rects.append(tempRects)

    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_yscale('log')
    ax.set_xticklabels(xLabels)
    ax.legend()

    for i in rects:
        autolabel(i, ax, labelFontSize=labelFontSize)

    plt.ylim(ylim)
    plt.savefig(fileName, bbox_inches='tight')
    plt.close()


valsTableSize1024 = {}
valsTableSize1024BlockSize64 = {}
valsTableSize1024BlockSize128 = {}
valsTableSize1024BlockSize512 = {}

valsTableSize2048 = {}
valsTableSize2048BlockSize64 = {}
valsTableSize2048BlockSize128 = {}
valsTableSize2048BlockSize512 = {}

valsTableSize4096 = {}
valsTableSize4096BlockSize64 = {}
valsTableSize4096BlockSize128 = {}
valsTableSize4096BlockSize512 = {}

dictList = [
    valsTableSize1024BlockSize64,
    valsTableSize1024BlockSize128,
    valsTableSize1024BlockSize512,
    valsTableSize2048BlockSize64,
    valsTableSize2048BlockSize128,
    valsTableSize2048BlockSize512,
    valsTableSize4096BlockSize64,
    valsTableSize4096BlockSize128,
    valsTableSize4096BlockSize512]

fp = open('{}/a2/FW1/fwSrPar.out'.format(str(getcwd())))
line = fp.readline().replace('\n', '')

while line:
    tokens = line.split(',')
    if tokens[2] == '1024':
        valsTableSize1024[tokens[1] + 'threads' + tokens[3]] = float(tokens[4])
        if tokens[3] == '64':
            valsTableSize1024BlockSize64[tokens[1] + 'threads'] = float(tokens[4])
        elif tokens[3] == '128':
            valsTableSize1024BlockSize128[tokens[1] + 'threads'] = float(tokens[4])
        elif tokens[3] == '512':
            valsTableSize1024BlockSize512[tokens[1] + 'threads'] = float(tokens[4])
    elif tokens[2] == '2048':
        valsTableSize2048[tokens[1] + 'threads' + tokens[3]] = float(tokens[4])
        if tokens[3] == '64':
            valsTableSize2048BlockSize64[tokens[1] + 'threads'] = float(tokens[4])
        elif tokens[3] == '128':
            valsTableSize2048BlockSize128[tokens[1] + 'threads'] = float(tokens[4])
        elif tokens[3] == '512':
            valsTableSize2048BlockSize512[tokens[1] + 'threads'] = float(tokens[4])
    elif tokens[2] == '4096':
        valsTableSize4096[tokens[1] + 'threads' + tokens[3]] = float(tokens[4])
        if tokens[3] == '64':
            valsTableSize4096BlockSize64[tokens[1] + 'threads'] = float(tokens[4])
        elif tokens[3] == '128':
            valsTableSize4096BlockSize128[tokens[1] + 'threads'] = float(tokens[4])
        elif tokens[3] == '512':
            valsTableSize4096BlockSize512[tokens[1] + 'threads'] = float(tokens[4])
    line = fp.readline().replace('\n', '')

print(valsTableSize1024)
print(valsTableSize2048)
print(valsTableSize4096)

minIdx1024 = min(valsTableSize1024, key=valsTableSize1024.get)
minIdx2048 = min(valsTableSize2048, key=valsTableSize2048.get)
minIdx4096 = min(valsTableSize4096, key=valsTableSize4096.get)

print(minIdx1024, valsTableSize1024[minIdx1024])
print(minIdx2048, valsTableSize2048[minIdx2048])
print(minIdx4096, valsTableSize4096[minIdx4096])

labels = ['1 thread', '2 threads', '4 threads', '8 threads', '16 threads', '32 threads', '64 threads']
barList1024 = [list(valsTableSize1024BlockSize64.values()), list(valsTableSize1024BlockSize128.values()),
               list(valsTableSize1024BlockSize512.values())]
legendLabels = ['Blocksize: 64', 'Blocksize: 128', 'Blocksize: 512']
barPlot(labels, 'FW Recursive Parallel: Table Size = 1024', barList1024, legendLabels, 'fwRecPar1024.png',
        ylim=[0.001, 10])

labels = ['1 thread', '2 threads', '4 threads', '8 threads', '16 threads', '32 threads', '64 threads']
barList2048 = [list(valsTableSize2048BlockSize64.values()), list(valsTableSize2048BlockSize128.values()),
               list(valsTableSize2048BlockSize512.values())]
legendLabels = ['Blocksize: 64', 'Blocksize: 128', 'Blocksize: 512']
barPlot(labels, 'FW Recursive Parallel: Table Size = 2048', barList2048, legendLabels, 'fwRecPar2048.png')

labels = ['1 thread', '2 threads', '4 threads', '8 threads', '16 threads', '32 threads', '64 threads']
barList4096 = [list(valsTableSize4096BlockSize64.values()), list(valsTableSize4096BlockSize128.values()),
               list(valsTableSize4096BlockSize512.values())]
legendLabels = ['Blocksize: 64', 'Blocksize: 128', 'Blocksize: 512']
barPlot(labels, 'FW Recursive Parallel: Table Size = 4096', barList4096, legendLabels, 'fwRecPar4096.png',
        ylim=[0.001, 800], labelFontSize=9)

labels = ['1 thread', '2 threads', '4 threads', '8 threads', '16 threads', '32 threads', '64 threads']
barList = [list(valsTableSize1024BlockSize128.values()), list(valsTableSize2048BlockSize128.values()),
           list(valsTableSize4096BlockSize512.values())]
legendLabels = ['Table Size: 1024X1024 (blocksize: 128)', 'Table Size: 2048X2048 (blocksize: 128)',
                'Table Size: 4096X4096 (blocksize: 512)']
barPlot(labels, 'FW Recursive Parallel: All Table Sizes', barList, legendLabels, 'fwRecParBestSizes.png')

filenameList = ['TableSize1024BlockSize64.png',
                'TableSize1024BlockSize128.png',
                'TableSize1024BlockSize512.png',
                'TableSize2048BlockSize64.png',
                'TableSize2048BlockSize128.png',
                'TableSize2048BlockSize512.png',
                'TableSize4096BlockSize64.png',
                'TableSize4096BlockSize128.png',
                'TableSize4096BlockSize512.png']

for i, valDict in enumerate(dictList):
    speedupPlot(valDict, filenameList[i].replace('TableSize', 'Table Size: ').replace('BlockSize', ', Block Size: ').replace('.png', ''),
                filenameList[i], labels[1:])
