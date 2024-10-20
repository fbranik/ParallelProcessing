#!/usr/bin/env python

import matplotlib
import re

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd

colorsList = ["dodgerblue", "orchid", "coral", "forestgreen", "coral", "forestgreen", "orchid", "yellow", "navy",
              "slategrey"]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=colorsList)


def autolabel(rects, ax, labelFontSize=12):
    for bar in rects:
        if bar.get_height() > 0.3:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 3 + 0.98 * bar.get_y(),
                    round(bar.get_height(), 2), ha='center',
                    color='black', rotation=90, size=9)
        # height = rect.get_height()
        # ax.annotate('{} sec'.format(round(height, 2)),
        #             xy=(rect.get_x() + 0.4 * rect.get_width(), height+rect.get_y()),
        #             xytext=(1, 3),
        #             textcoords="offset points",
        #             ha='center', va='bottom', fontsize=labelFontSize, rotation=90)


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
    ax.set_yticks(np.arange(1, 30, 3))
    ax.grid('on')
    ax.set_xticklabels(xLabels)
    plt.savefig(fileName, bbox_inches='tight')
    plt.close()


def barPlot(xLabels, title, barList, legendLabels, fileName, width=1, ylim=[0.001, 500], labelFontSize=12, xxLabels=[]):
    x = np.arange(0, len(xLabels) * len(xLabels), step=len(xLabels))
    fig = plt.figure()
    fig.set_size_inches(16.5, 12.5)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2)
    rects = []
    xAxisOffset = np.linspace(start=-len(barList) * width / 2, stop=len(barList) * width / 2,
                              num=len(barList))

    xx = []
    for iX in x:
        for iOffset in xAxisOffset:
            xx.append(iX + iOffset)

    for i in range(len(barList)):
        tTransfers = np.array([item[0] for item in barList[i]])
        tGPU = np.array([item[1] for item in barList[i]])
        tCPU = np.array([item[2] for item in barList[i]])
        tempList = np.array([tGPU, tCPU, tTransfers])
        for j, toPlot in enumerate(tempList):
            if j == 0:
                tempRects = ax.bar(x + xAxisOffset[i], toPlot, width,
                                   color=colorsList[j])
            else:
                tempRects = ax.bar(x + xAxisOffset[i], toPlot, width, bottom=np.sum(tempList[0:j], axis=0),
                                   label=legendLabels[j], color=colorsList[j])
            rects.append(tempRects)

    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xticks(xx)
    xxLabels = xxLabels * 6
    ax.xaxis.set_tick_params(labelsize=10)
    ax.set_xticklabels(xxLabels, rotation=90)
    # box = ax.get_position()
    # # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    # #                  box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(legendLabels, loc='upper center', bbox_to_anchor=(0.5, 1),
              fancybox=True, shadow=True, ncol=3)

    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.18))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    ax2.set_xlim(ax.get_xlim())

    ax2.xaxis.set_tick_params(labelsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xLabels)
    ax2.set_xlabel('Thread Block Size', fontsize=12)

    for i in rects:
        autolabel(i, ax, labelFontSize=labelFontSize)

    plt.ylim(ylim)
    plt.savefig(fileName, bbox_inches='tight')
    plt.close()


def readKmeansOutFile(fileName):
    fp = open('{}/{}.out'.format(str(getcwd()), fileName))
    line = fp.readline()
    newBlockSizeFlag = False
    newExecFlag = False
    outDict = {}
    tempDict = {}
    thisNumCoords = ''
    thisBlockSize = ''
    thisExec = ''
    while line:
        if line.startswith("dataset_size"):
            tempMatchObj = re.search(r'numCoords\s= (\d+)\s+numClusters = \d+\s+block_size = (\d+)', line).groups()
            thisNumCoords = int(tempMatchObj[0])
            thisBlockSize = int(tempMatchObj[1])
            newBlockSizeFlag = True
        if newBlockSizeFlag and line.startswith('|-'):
            tempMatchObj = re.search(r'\-([\w\s()]+)\-', line).groups()
            thisExec = tempMatchObj[0].replace(' GPU Kmeans', '')
            if thisExec not in outDict:
                outDict[thisExec] = {}
            newExecFlag = True
            newBlockSizeFlag = False
        if line.startswith('tTransfers') and newExecFlag:
            thisKey = (thisNumCoords, thisBlockSize)
            tempMatchObj = re.search(r'tTransfers=(\d+\.\d+), tGPU=(\d+\.\d+), tCPU=(\d+\.\d+)', line).groups()
            tTransfers = float(tempMatchObj[0])
            tGPU = float(tempMatchObj[1])
            tCPU = float(tempMatchObj[2])
            tTotal = tTransfers + tGPU + tCPU
            newExecFlag = False
            tempDict[thisKey] = (tTransfers, tGPU, tCPU, tTotal)
            if thisBlockSize == 1024 or thisExec == 'Sequential Kmeans':
                outDict[thisExec].update(tempDict)
                tempDict = {}

        line = fp.readline()

    fp.close()
    return outDict


kmeansCudaVersions = ['Sequential Kmeans', 'Naive', 'Transpose', 'Shared', 'All', 'All (ParallelBlockReduction)']

kmeansCudaDicts = readKmeansOutFile('runCudaKmeansAllGPU')

cudaSeq = kmeansCudaDicts['Sequential Kmeans']
cudaNaive = kmeansCudaDicts['Naive']
cudaTranspose = kmeansCudaDicts['Transpose']
cudaShared = kmeansCudaDicts['Shared']
cudaAllGPU = kmeansCudaDicts['All']
cudaAllGPUAdv = kmeansCudaDicts['All (ParallelBlockReduction)']

cudaSeq2 = {k: cudaSeq[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if k in cudaSeq}

cudaNaive2 = {k: cudaNaive[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if k in cudaNaive}

cudaTranspose2 = {k: cudaTranspose[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if
                  k in cudaTranspose}

cudaShared2 = {k: cudaShared[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if k in cudaShared}

cudaAllGPU2 = {k: cudaAllGPU[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if k in cudaAllGPU}

cudaAllGPUAdv2 = {k: cudaAllGPUAdv[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if
                  k in cudaAllGPUAdv}

cudaNaive16 = {k: cudaNaive[k] for k in [(16, 32), (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024)] if
               k in cudaNaive}

cudaTranspose16 = {k: cudaTranspose[k] for k in [(16, 32), (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024)] if
                   k in cudaTranspose}

cudaShared16 = {k: cudaShared[k] for k in [(16, 32), (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024)] if
                k in cudaShared}

cudaAllGPU16 = {k: cudaAllGPU[k] for k in [(16, 32), (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024)] if
                k in cudaAllGPU}

cudaAllGPUAdv16 = {k: cudaAllGPUAdv[k] for k in [(16, 32), (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024)] if
                   k in cudaAllGPUAdv}

dictList = [cudaSeq2, cudaNaive16, cudaTranspose16, cudaShared16, cudaAllGPU16, cudaAllGPUAdv16]

labels = ['32', '64', '128', '256', '512', '1024']

legendLabels = ['GPU Time', 'CPU Time', 'Memory Transfers Time']

barList = [
    list(cudaNaive16.values()),
    list(cudaTranspose16.values()),
    list(cudaShared16.values()),
    list(cudaAllGPU16.values()),
    list(cudaAllGPUAdv16.values())
]

versionLabels = ['Naive', 'Transpose', 'Shared', 'All GPU', 'All GPU (Adv.)']

barPlot(labels, 'GPU Kmeans Parallel Detailed Runtimes (numCoords = 16)', barList, legendLabels,
        'cudaAllVersionsBars16CoordsAdvTimers.png',
        ylim=[0.1, 3], labelFontSize=10, width=(len(versionLabels) - 2) / len(labels), xxLabels=versionLabels)
#
# speedupPlot(dictList, 'GPU Kmeans Parallel Speedup (numCoords = 16)', 'cudaAllVersionsBars16Coords.png', labels[1:],
#             legendLabels=legendLabels)
barList = [
    list(cudaNaive2.values()),
    list(cudaTranspose2.values()),
    list(cudaShared2.values()),
    list(cudaAllGPU2.values()),
    list(cudaAllGPUAdv2.values())
]

dictList = [cudaSeq2, cudaNaive2, cudaTranspose2, cudaShared2, cudaAllGPU2, cudaAllGPUAdv2]

barPlot(labels, 'GPU Kmeans Parallel Detailed Runtimes (numCoords = 2)', barList, legendLabels,
        'cudaAllVersionsBars2CoordsAdvTimers.png',
        ylim=[0.1, 20], labelFontSize=10, width=(len(versionLabels) - 2) / len(labels), xxLabels=versionLabels)
barList = [
    list(cudaNaive2.values()),
    list(cudaTranspose2.values()),
    list(cudaShared2.values()),
    list(cudaAllGPU2.values()),
    list(cudaAllGPUAdv2.values())
]
print(barList)
# speedupPlot(dictList, 'Kmeans Parallel Speedup', 'allVersionsSpeedup.png', labels[1:],
#             legendLabels=legendLabels)
