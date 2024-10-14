#!/usr/bin/env python

import matplotlib
import re

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd

colorsList = ["navy", "dodgerblue", "coral", "forestgreen", "orchid", "slategrey",  "crimson", "yellow", "chocolate"]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=colorsList)


def autolabel(rects, ax, labelFontSize=12):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{} sec'.format(round(height, 2)),
                    xy=(rect.get_x() + 0.4 * rect.get_width(), height),
                    xytext=(1, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=labelFontSize, rotation=90)


def speedupPlot(inDictList, title, fileName, xLabels, legendLabels):
    fig, ax = plt.subplots()
    fig.set_size_inches(14.5, 14.5)
    for idx, iDict in enumerate(inDictList):
        speedup = {}
        for key, value in iDict.items():
            if key == '1':
                serialTime = value
                continue
            else:
                speedup[key.replace('threads', '')] = round(serialTime / value, 2)
        print(speedup)

        x = np.arange(0, 2 * (len(iDict) - 1), step=2)

        ax.plot(x, speedup.values(), marker='x', label=legendLabels[idx])

    ax.set_ylabel('Speedup')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(1, 42, 3))
    ax.grid('on')
    ax.set_xticklabels(xLabels)
    plt.savefig(fileName, bbox_inches='tight')
    plt.close()


def barPlot(xLabels, title, barList, legendLabels, fileName, width=1, ylim=[0.001, 500], labelFontSize=12):
    x = np.arange(0, len(legendLabels) * len(xLabels), step=len(legendLabels))
    fig, ax = plt.subplots()
    fig.set_size_inches(16.5, 12.5)

    rects = []
    xAxisOffset = np.linspace(start=-len(legendLabels) * width / 2, stop=len(legendLabels) * width / 2,
                              num=len(legendLabels))
    for i in range(len(barList)):
        tempRects = ax.bar(x + xAxisOffset[i], barList[i], width,
                           label=legendLabels[i])

        rects.append(tempRects)

    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(xLabels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)

    for i in rects:
        autolabel(i, ax, labelFontSize=labelFontSize)

    plt.ylim(ylim)
    plt.savefig(fileName, bbox_inches='tight')
    plt.close()


def readKmeansOutFile(fileName):
    fp = open('{}/{}.out'.format(str(getcwd()), fileName))
    line = fp.readline()
    newThreadFlag = False
    outDict = {}
    tempDict = {}
    thisThread = ''
    prevLock = ''
    while line:
        if "threads" in line:
            newThreadFlag = True
            thisThread = line.split(": ")[1].replace(")\n", '')
        if newThreadFlag and "total = " in line:
            thisTime = re.sub(r's\).*', '', line.split("total = ")[1])
            tempDict[thisThread] = float(thisTime.replace("\n", ''))
            newThreadFlag = False
        if thisThread == '32':
            outDict = tempDict

        line = fp.readline()

    fp.close()
    return outDict

kmeansRevisedNaive                = readKmeansOutFile('kmeansNaive')
kmeansRevisedNaiveCPUAff          = readKmeansOutFile('kmeansRevisedNaive')
kmeansRevisedSimpleReduction      = readKmeansOutFile('kmeansRevisedSimpleReduction')
kmeansRevisedSimpleReductionSmall = readKmeansOutFile('kmeansRevisedSimpleReductionSmall')
kmeansRevisedAdvReductionSmall    = readKmeansOutFile('kmeansRevisedAdvReductionSmall')
kmeansRevisedReductionSmallNUMA   = readKmeansOutFile('kmeansRevisedBonusSmall')
kmeansRevisedReductionNUMA        = readKmeansOutFile('kmeansRevisedBonus')


dictList = [kmeansRevisedNaive,kmeansRevisedNaiveCPUAff, kmeansRevisedSimpleReduction,
            kmeansRevisedSimpleReductionSmall, kmeansRevisedAdvReductionSmall, kmeansRevisedReductionSmallNUMA, kmeansRevisedReductionNUMA     ]

labels = ['1 thread', '2 threads', '4 threads', '8 threads', '16 threads', '32 threads', '64 threads']
barList = [
    list(kmeansRevisedNaive.values()),
    list(kmeansRevisedNaiveCPUAff.values()),
    list(kmeansRevisedSimpleReduction.values()),
    list(kmeansRevisedSimpleReductionSmall.values()),
    list(kmeansRevisedAdvReductionSmall.values()),
    list(kmeansRevisedReductionSmallNUMA.values()),
    list(kmeansRevisedReductionNUMA.values())
]
print(barList)
legendLabels = [
    'Naive',
    'Naive with CPU Affinity',
    'Simple Reduction',
    'Simple Reduction (Small Parameters)',
    'Reduction with Local Copies Thread-Allocated (Small Parameters)',
    'NUMA Aware object Generation (Small Parameters)',
    'NUMA Aware object Generation'
]

print(dictList)

barPlot(labels, 'Kmeans Parallel Runtimes', barList, legendLabels, 'allVersionsBars.png',
        ylim=[0.1, 350], labelFontSize=10, width=(len(legendLabels) - 2) / len(labels))

speedupPlot(dictList, 'Kmeans Parallel Speedup', 'allVersionsSpeedup.png', labels[1:],
            legendLabels=legendLabels)
