#!/usr/bin/env python

import matplotlib
import re

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd

colorsList = ["dodgerblue", "coral", "darkseagreen", "orchid", "navy"]
matplotlib.rcParams.update({'font.size': 15})
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
        if key == '1':
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


def readKmeansOutFile(fileName):
    fp = open('{}/a2/kmeans/{}.out'.format(str(getcwd()), fileName))
    line = fp.readline()
    newThreadFlag = False
    outDict = {}
    while line:
        if "threads" in line:
            newThreadFlag = True
            thisThread = line.split(": ")[1].replace(")\n", '')
        if newThreadFlag and "total = " in line:
            thisTime = re.sub(r's\).*', '', line.split("total = ")[1])
            outDict[thisThread] = float(thisTime.replace("\n", ''))
            newThreadFlag = False

        line = fp.readline()

    fp.close()
    return outDict


kmeansNaive = readKmeansOutFile('kmeansNaive')
print(kmeansNaive)

kmeansNaiveCPUAff = readKmeansOutFile('kmeansNaiveCPUAff')
print(kmeansNaiveCPUAff)

kmeansReductionF1 = readKmeansOutFile('kmeansReductionF1')
print(kmeansReductionF1)

kmeansReductionF2 = readKmeansOutFile('kmeansReductionF2')
print(kmeansReductionF2)

kmeansReductionSmallF1 = readKmeansOutFile('kmeansReductionSmallF1')
print(kmeansReductionSmallF1)

kmeansReductionSmallF2 = readKmeansOutFile('kmeansReductionSmallF2')
print(kmeansReductionSmallF2)

labels = ['1 thread', '2 threads', '4 threads', '8 threads', '16 threads', '32 threads', '64 threads']
barList = [list(kmeansNaive.values()), list(kmeansNaiveCPUAff.values()),
           list(kmeansReductionF1.values())]
legendLabels = ['Naive kmeans', 'Naive kmeans w/ CPU AFF',
                'Reduction kmeans']
barPlot(labels, 'Kmeans Parallel Runtimes', barList, legendLabels, 'kmeansParallelFirst.png', ylim=[0.001, 1200],
        labelFontSize=10)

labels = ['1 thread', '2 threads', '4 threads', '8 threads', '16 threads', '32 threads', '64 threads']
barList = [list(kmeansReductionF1.values()), list(kmeansReductionSmallF1.values()),
           list(kmeansReductionF2.values()), list(kmeansReductionSmallF2.values())]
legendLabels = ['Simple Reduction kmeans', 'Simple Reduction kmeans (Smaller sizes)',
                '"First Touch Aware" Reduction kmeans', '"First Touch Aware" Reduction kmeans (Smaller sizes)']
barPlot(labels, 'Kmeans w/ Reduction Runtimes', barList, legendLabels, 'kmeansReductionBars.png', ylim=[0.001, 1500],
        labelFontSize=8)

dictList = [kmeansNaive, kmeansNaiveCPUAff,
            kmeansReductionF1, kmeansReductionSmallF1,
            kmeansReductionF2, kmeansReductionSmallF2]

filenameList = ['kmeansNaive.png',
                'kmeansNaiveCPUAff.png',
                'kmeansReductionF1.png',
                'kmeansReductionSmallF1.png',
                'kmeansReductionF2.png',
                'kmeansReductionSmallF2.png']

labelList = ['Naive kmeans',
             'Naive kmeans w/ CPU Aff.',
             'Simple Reduction kmeans',
             'Simple Reduction kmeans (smaller sizes)',
             '"First Touch Aware" Reduction kmeans',
             '"First Touch Aware" Reduction kmeans (Smaller sizes)']

for i, valDict in enumerate(dictList):
    speedupPlot(valDict, labelList[i],
                filenameList[i], labels[1:])
