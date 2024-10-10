#!/usr/bin/env python

import matplotlib
import re

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd

colorsList = ["crimson",
              "dodgerblue", "slategrey",
              "forestgreen", "orchid", "crimson", "yellow", "chocolate", "slategrey"
              ]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=colorsList)


def autolabel(rects, ax, labelFontSize=12):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{} sec'.format(round(height, 4)),
                    xy=(rect.get_x() + 0.4 * rect.get_width(), height),
                    xytext=(1, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=labelFontSize, rotation=90)


def speedupPlot(inDictList, title, fileName, xLabels, legendLabels):
    fig, ax = plt.subplots()
    fig.set_size_inches(16.5, 16.5)
    for idx, iDict in enumerate(inDictList):
        speedup = {}
        for key, value in iDict.items():
            if key == 1:
                serialTime = value
                continue
            else:
                speedup[key] = round(serialTime / value, 2)
        print(iDict)

        x = np.arange(0, 2 * (len(iDict) - 1), step=2)

        if idx>2:
            ax.plot(x, speedup.values(), '--o', label=legendLabels[idx], color=colorsList[idx-3])
        # elif idx >4:
        #     ax.plot(x, speedup.values(), ':>', label=legendLabels[idx], color=colorsList[idx-7])
        else:

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
    ax.set_yticks(np.arange(1, 11, 1))
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
        if i>2:
            tempRects = ax.bar(x + xAxisOffset[i], barList[i], width,
                               label=legendLabels[i], hatch="xx", color=colorsList[i-3])
        # elif i>6:
        #     tempRects = ax.bar(x + xAxisOffset[i], barList[i], width,
        #                        label=legendLabels[i], hatch="--",color=colorsList[i-7])
        else:
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


def readFWOutFile(fileName):
    fp = open('{}/{}.out'.format(str(getcwd()), fileName))
    line = fp.readline()
    outDict = {}
    tempDict = {}
    prevSizeN = '1024'
    prevSizeB = '32'
    #    threads = [1, 2, 4, 8, 16, 32, 64]
    threads = 1
    while line:
        tempMatchObj = re.match('FW_SR,(\d+)\,(\d+)\,(\d+\.\d+)', line)
        thisSizeN = tempMatchObj[1]
        thisSizeB = tempMatchObj[2]
        thisTime = float(tempMatchObj[3])

        if prevSizeN != thisSizeN:
            outDict[(prevSizeN, prevSizeB)] = tempDict
            threads = 1
            tempDict = {}
            prevSizeN = thisSizeN
            prevSizeB = thisSizeB

        tempDict[threads] = thisTime
        threads *= 2
        line = fp.readline()
    outDict[(prevSizeN, thisSizeB)] = tempDict
    fp.close()
    return outDict


fwRevised = readFWOutFile('fwRevisedNested0')
fwRevisedAdv = readFWOutFile('fwRevisedNested1')
print(fwRevised)
bSizes = ['64', '128']
nSizes = ['1024', '2048', '4096']

bestSimple = {}
bestAdv = {}

for thisN in nSizes:
    prevB = bSizes[0]
    bestSimple[thisN] = (prevB, fwRevised[(thisN, prevB)])
    bestAdv[thisN] = (prevB, fwRevisedAdv[(thisN, prevB)])
    print('................................')
    for thisB in bSizes:
        print(bestSimple[thisN][1], min(bestSimple[thisN][1].values()), 'eok')
        for i in [1, 2, 4, 8, 16, 32, 64]:
            if fwRevised[(thisN, thisB)][i] < min(bestSimple[thisN][1].values()):
                bestSimple[thisN] = (thisB, fwRevised[(thisN, thisB)])
        for i in [1, 2, 4, 8, 16, 32, 64]:
            if fwRevisedAdv[(thisN, thisB)][i] < min(bestAdv[thisN][1].values()):
                bestAdv[thisN] = (thisB, fwRevisedAdv[(thisN, thisB)])

for thisN in nSizes:
    print('................................')
    for thisB in bSizes:
        print(fwRevisedAdv[(thisN, thisB)])

print(bestAdv)
print(bestSimple)

dictList = [bestSimple['1024'][1], bestSimple['2048'][1], bestSimple['4096'][1],
            bestAdv['1024'][1], bestAdv['2048'][1], bestAdv['4096'][1]]

labels = ['1 thread', '2 threads', '4 threads', '8 threads', '16 threads', '32 threads', '64 threads']
barList = [
    list(bestSimple['1024'][1].values()),
    list(bestSimple['2048'][1].values()),
    list(bestSimple['4096'][1].values()),
    list(bestAdv['1024'][1].values()),
    list(bestAdv['2048'][1].values()),
    list(bestAdv['4096'][1].values())
]
# print(barList)
legendLabels = ['Simple task-based Parallel Recursive FW (N=1024, B={})'.format(bestSimple['1024'][0]),
                'Simple task-based Parallel Recursive FW (N=2048, B={})'.format(bestSimple['2048'][0]),
                'Simple task-based Parallel Recursive FW (N=4096, B={})'.format(bestSimple['4096'][0]),
                'Optimized task-based Parallel Recursive FW (N=1024, B={})'.format(bestAdv['1024'][0]),
                'Optimized task-based Parallel Recursive FW (N=2048, B={})'.format(bestAdv['2048'][0]),
                'Optimized task-based Parallel Recursive FW (N=4096, B={})'.format(bestAdv['4096'][0])
                ]
#
# print(dictList)
#
barPlot(labels, 'FW Parallel Runtimes', barList, legendLabels, 'fwAllVersionsBars.png',
        ylim=[0.001, 150], labelFontSize=10, width=(len(legendLabels) - 1) / len(labels))

speedupPlot(dictList, 'FW Parallel Speedup', 'fwAllVersionsSpeedup.png', labels[1:],
            legendLabels=legendLabels)

#
# fwRevisedTiled = readFWOutFile('fwSrRevisedTiled')
# print(fwRevisedTiled)
# dictList = [
#             fwRevisedTiled[('1024', '64')],
#             fwRevisedTiled[('2048', '64')],
#             fwRevisedTiled[('4096', '64')],
#             fwRevisedTiled[('1024', '32')],
#
#             fwRevisedTiled[('1024', '128')],
#             fwRevisedTiled[('2048', '128')],
#             fwRevisedTiled[('4096', '128')],
#
#             fwRevisedTiled[('1024', '256')],
#             fwRevisedTiled[('2048', '256')],
#             fwRevisedTiled[('4096', '256')]
#             ]
#
# labels = ['1 thread', '2 threads', '4 threads', '8 threads', '16 threads', '32 threads', '64 threads']
# barList = [list(fwRevisedTiled[('1024', '64')].values()),
#            list(fwRevisedTiled[('2048', '64')].values()),
#            list(fwRevisedTiled[('4096', '64')].values()),
#
#            list(fwRevisedTiled[('1024', '128')].values()),
#            list(fwRevisedTiled[('2048', '128')].values()),
#            list(fwRevisedTiled[('4096', '128')].values()),
#
#            list(fwRevisedTiled[('1024', '256')].values()),
#            list(fwRevisedTiled[('2048', '256')].values()),
#            list(fwRevisedTiled[('4096', '256')].values())
#
#            ]
# # print(barList)
# # legendLabels = [
# #                 'Parallel Tiled FW (N=1024, B=64)',
# #                 'Parallel Tiled FW (N=2048, B=64)',
# #                 'Parallel Tiled FW (N=4096, B=64)',
# #
# #                 'Parallel Tiled FW (N=1024, B=128)',
# #                 'Parallel Tiled FW (N=2048, B=128)',
# #                 'Parallel Tiled FW (N=4096, B=128)',
# #
# #                 'Parallel Tiled FW (N=1024, B=256)',
# #                 'Parallel Tiled FW (N=2048, B=256)',
# #                 'Parallel Tiled FW (N=4096, B=256)',
# #                 ]
# #
# # print(dictList)
# #
# # barPlot(labels, 'FW Tiled Parallel Runtimes', barList, legendLabels, 'fwTiledVersionsBars.png',
# #         ylim=[0.01, 150], labelFontSize=10, width=(len(legendLabels) - 4) / len(labels))
#
# # legendLabels = [
# #                 'Parallel Tiled FW (N=1024, B=64)',
# #                 'Parallel Tiled FW (N=2048, B=64)',
# #                 'Parallel Tiled FW (N=4096, B=64)',
# #                 'Parallel Tiled FW (N=1024, B=32)',
# #
# #                 'Parallel Tiled FW (N=1024, B=128)',
# #                 'Parallel Tiled FW (N=2048, B=128)',
# #                 'Parallel Tiled FW (N=4096, B=128)',
# #
# #                 'Parallel Tiled FW (N=1024, B=256)',
# #                 'Parallel Tiled FW (N=2048, B=256)',
# #                 'Parallel Tiled FW (N=4096, B=256)',
# #                 ]
# # speedupPlot(dictList, 'FW Tiled Parallel Speedup', 'fwTiledVersionsSpeedup.png', labels[1:],
# #             legendLabels=legendLabels)
#
# # dictList = [
# #             fwRevisedTiled[('1024', '32')],
# #             fwRevisedTiled[('4096', '64')]
# #             ]
# # legendLabels = ['Parallel Tiled FW (N=1024, B=32)',
# #                 'Parallel Tiled FW (N=4096, B=64)'
# #                 ]
# # speedupPlot(dictList, 'FW Tiled Parallel Speedup', 'fwTiledVersionsExtraSpeedup.png', labels[1:],
# #             legendLabels=legendLabels)
