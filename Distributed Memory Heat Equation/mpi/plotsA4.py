#!/usr/bin/env python

import matplotlib
import re

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd

colorsList = ["dodgerblue", "orchid", "coral", "forestgreen", "crimson", "lime", "chocolate", "yellow", "navy",
              "slategrey"]
speedupColors = [
    "navy", "green", "darkred",
    "royalblue", "limegreen", "crimson",
    "slateblue", "palegreen", "red"
]
speedupMarkers = [
    'x', 'x', 'x',
    'o', 'o', 'o',
    '>', '>', '>'
]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=colorsList)


def autolabel(rects, ax, labelFontSize=12):
    for bar in rects:

        if 100 > bar.get_height() > 3.2:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 3 + bar.get_y(),
                    str(round(bar.get_height(), 2)), ha='center',
                    color='black', rotation=0, size=7)
        elif bar.get_height() >= 100:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 3 + bar.get_y(),
                    str(round(bar.get_height(), 2)), ha='center',
                    color='black', rotation=0, size=6)
        # else:
        #     ax.text(bar.get_x() + bar.get_width() / 2,
        #             bar.get_height() / 2 + bar.get_y(),
        #             str(round(bar.get_height(), 2))+' sec', ha='center',
        #             color='black', rotation=90, size=9)

        # height = rect.get_height()
        # ax.annotate('{} sec'.format(round(height, 2)),
        #             xy=(rect.get_x() + 0.4 * rect.get_width(), height+rect.get_y()),
        #             xytext=(1, 3),
        #             textcoords="offset points",
        #             ha='center', va='bottom', fontsize=labelFontSize, rotation=90)


def speedupPlot(inList, title, fileName, xLabels, legendLabels):
    fig, ax = plt.subplots()
    fig.set_size_inches(14.5, 14.5)
    for idx, iList in enumerate(inList):
        print(iList)
        speedup = {}
        for key, value in enumerate(iList):
            if key == 0:
                serialTime = value[-1]
                continue
            else:
                speedup[2**key] = round(serialTime / value[-1], 2)

        x = np.arange(0, 2 * (len(iList) - 1), step=2)
        ax.plot(x, speedup.values(), marker=speedupMarkers[idx], label=legendLabels[idx], color=speedupColors[idx])
    ax.set_ylabel('Speedup')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 170, 10))
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
    print(barList)
    for iX in x:
        for iOffset in xAxisOffset:
            xx.append(iX + iOffset)
    for i in range(len(barList)):
        if len(legendLabels) == 1:
            tTotal = np.array([item[-1] for item in barList[i]])
            tempList = np.array([tTotal])
        else:
            tComp = np.array([item[0] for item in barList[i]])
            tComm = np.array([item[1] for item in barList[i]])
            tempList = np.array([tComp, tComm])
        for j, toPlot in enumerate(tempList):
            if j == 0:
                if len(legendLabels)==1:
                    tempColor = "slategrey"
                else:
                    tempColor = colorsList[j]
                tempRects = ax.bar(x+xAxisOffset[i], toPlot, width,
                                   color=tempColor)
            else:
                tempRects = ax.bar(x + xAxisOffset[i], toPlot, width, bottom=np.sum(tempList[0:j], axis=0),
                                   label=legendLabels[j], color=colorsList[j])
            rects.append(tempRects)

    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    # ax.set_yscale('log')
    ax.set_xticks(xx)
    xxLabels = xxLabels * 7
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
    ax2.set_xlabel('Number of MPI Processes', fontsize=12)

    for i in rects:
        autolabel(i, ax, labelFontSize=labelFontSize)

    plt.ylim(ylim)
    plt.savefig(fileName, bbox_inches='tight')
    plt.close()


def readOutFile(fileName, convFile=False):
    fp = open('{}/{}.out'.format(str(getcwd()), fileName))
    line = fp.readline()
    newBlockSizeFlag = False
    newExecFlag = False
    outDict = {}

    while line:
        if line.startswith("Jacobi"):
            version = "Jacobi"
        elif line.startswith("GaussSeidel"):
            version = "Gauss Seidel SOR"
        elif line.startswith("RedBlack"):
            version = "Red Black SOR"
        else:
            pass
        tempMatchObj = re.search(
            r'X\s(\d+) Y\s\d+ Px\s(\d+) Py\s(\d+) Iter \d+ ComputationTime (\d+\.\d+) CommsTime (\d+\.\d+) ConvTime (\d+\.\d+) TotalTime (\d+\.\d+)',
            line).groups()
        arraySize = int(tempMatchObj[0])
        numProcs  = int(tempMatchObj[1]) * int(tempMatchObj[2])
        compTime  = float(tempMatchObj[3])
        commsTime = float(tempMatchObj[4])
        totalTime = float(tempMatchObj[6])
        convTime  = float(tempMatchObj[5])
        outDict[(version, arraySize, numProcs)] = [compTime, commsTime, convTime, totalTime]
        line = fp.readline()

    fp.close()
    return outDict

# heatDictsConv = readOutFile('results/conv1024')
#
# labels = ['']
#
# legendLabels = ['Computation Time', 'Communications Time', 'Convergence Time']
#
# barList = [
#     heatDictsConv[('Gauss Seidel SOR', 1024, 64)],
#     heatDictsConv[('Red Black SOR', 1024, 64)]
# ]
#
# versionLabels = ['Gauss Seidel SOR', 'Red Black SOR']
# #
# barPlot(labels, 'Parallel Heat Equation solution (1024X1024, with Convergence Test)', barList, legendLabels,
#         'conv1024_GS_RB.png',
#         ylim=[0.1, 3], labelFontSize=10, width=0.05, xxLabels=versionLabels)
#
# barPlot(labels, 'Parallel Heat Equation solution (1024X1024, with Convergence Test)', [heatDictsConv[('Jacobi', 1024, 64)]], legendLabels,
#         'conv1024_J.png',
#         ylim=[1, 400], labelFontSize=10, width=0.5, xxLabels=['Jacobi'])

heatDicts = {}

heatDicts.update(readOutFile('results/noConvN1C1'))
heatDicts.update(readOutFile('results/noConvN1C2'))
heatDicts.update(readOutFile('results/noConvN1C4'))
heatDicts.update(readOutFile('results/noConvN1C8'))
heatDicts.update(readOutFile('results/noConvN2C8'))
heatDicts.update(readOutFile('results/noConvN4C8'))
heatDicts.update(readOutFile('results/noConvN8C8'))

speedupList = []
labels = ['1', '2', '4', '8', '16', '32', '64']

legendLabels = ['Computation Time', 'Communications Time']

barList = [
    list({k: heatDicts[k] for k in [('Jacobi', 2048, 1), ('Jacobi', 2048, 2),
                                    ('Jacobi', 2048, 4), ('Jacobi', 2048, 8),
                                    ('Jacobi', 2048, 16), ('Jacobi', 2048, 32),
                                    ('Jacobi', 2048, 64)] if k in heatDicts}.values()),
    list({k: heatDicts[k] for k in [('Gauss Seidel SOR', 2048, 1), ('Gauss Seidel SOR', 2048, 2),
                                    ('Gauss Seidel SOR', 2048, 4), ('Gauss Seidel SOR', 2048, 8),
                                    ('Gauss Seidel SOR', 2048, 16), ('Gauss Seidel SOR', 2048, 32),
                                    ('Gauss Seidel SOR', 2048, 64)] if k in heatDicts}.values()),
    list({k: heatDicts[k] for k in [('Red Black SOR', 2048, 1), ('Red Black SOR', 2048, 2),
                                    ('Red Black SOR', 2048, 4), ('Red Black SOR', 2048, 8),
                                    ('Red Black SOR', 2048, 16),('Red Black SOR', 2048, 32),
                                    ('Red Black SOR', 2048, 64)] if k in heatDicts}.values())
]
speedupList.extend(barList)
versionLabels = ['Jacobi', 'Gauss Seidel SOR', 'Red Black SOR']
#
barPlot(labels, 'Parallel Heat Equation solution (2048X2048, no Convergence Test, T=256)', barList, legendLabels,
        'conv2048.png',
        ylim=[0.1, 80], labelFontSize=10, width=1.2, xxLabels=versionLabels)

legendLabels = ['Total Time']

barPlot(labels, 'Parallel Heat Equation solution Total Time (2048X2048, no Convergence Test, T=256)', barList, legendLabels,
        'conv2048Total.png',
        ylim=[0.1, 80], labelFontSize=10, width=1.2, xxLabels=versionLabels)


barList = [
    list({k: heatDicts[k] for k in [('Jacobi', 4096, 1), ('Jacobi', 4096, 2),
                                    ('Jacobi', 4096, 4), ('Jacobi', 4096, 8),
                                    ('Jacobi', 4096, 16), ('Jacobi', 4096, 32),
                                    ('Jacobi', 4096, 64)] if k in heatDicts}.values()),
    list({k: heatDicts[k] for k in [('Gauss Seidel SOR', 4096, 1), ('Gauss Seidel SOR', 4096, 2),
                                    ('Gauss Seidel SOR', 4096, 4), ('Gauss Seidel SOR', 4096, 8),
                                    ('Gauss Seidel SOR', 4096, 16), ('Gauss Seidel SOR', 4096, 32),
                                    ('Gauss Seidel SOR', 4096, 64)] if k in heatDicts}.values()),
    list({k: heatDicts[k] for k in [('Red Black SOR', 4096, 1), ('Red Black SOR', 4096, 2),
                                    ('Red Black SOR', 4096, 4), ('Red Black SOR', 4096, 8),
                                    ('Red Black SOR', 4096, 16),('Red Black SOR', 4096, 32),
                                    ('Red Black SOR', 4096, 64)] if k in heatDicts}.values())
]
speedupList.extend(barList)
versionLabels = ['Jacobi', 'Gauss Seidel SOR', 'Red Black SOR']
legendLabels = ['Computation Time', 'Communications Time']

barPlot(labels, 'Parallel Heat Equation solution (4096X4096, no Convergence Test, T=256)', barList, legendLabels,
        'conv4096.png',
        ylim=[0.1, 300], labelFontSize=10, width=1.2, xxLabels=versionLabels)

legendLabels = ['Total Time']
barPlot(labels, 'Parallel Heat Equation solution Total Time (4096X4096, no Convergence Test, T=256)', barList, legendLabels,
        'conv4096Total.png',
        ylim=[0.1, 300], labelFontSize=10, width=1.2, xxLabels=versionLabels)

barList = [
    list({k: heatDicts[k] for k in [('Jacobi', 6144, 1), ('Jacobi', 6144, 2),
                                    ('Jacobi', 6144, 4), ('Jacobi', 6144, 8),
                                    ('Jacobi', 6144, 16), ('Jacobi', 6144, 32),
                                    ('Jacobi', 6144, 64)] if k in heatDicts}.values()),
    list({k: heatDicts[k] for k in [('Gauss Seidel SOR', 6144, 1), ('Gauss Seidel SOR', 6144, 2),
                                    ('Gauss Seidel SOR', 6144, 4), ('Gauss Seidel SOR', 6144, 8),
                                    ('Gauss Seidel SOR', 6144, 16), ('Gauss Seidel SOR', 6144, 32),
                                    ('Gauss Seidel SOR', 6144, 64)] if k in heatDicts}.values()),
    list({k: heatDicts[k] for k in [('Red Black SOR', 6144, 1), ('Red Black SOR', 6144, 2),
                                    ('Red Black SOR', 6144, 4), ('Red Black SOR', 6144, 8),
                                    ('Red Black SOR', 6144, 16),('Red Black SOR', 6144, 32),
                                    ('Red Black SOR', 6144, 64)] if k in heatDicts}.values())
]
speedupList.extend(barList)
versionLabels = ['Jacobi', 'Gauss Seidel SOR', 'Red Black SOR']

legendLabels = ['Computation Time', 'Communications Time']

barPlot(labels, 'Parallel Heat Equation solution (6144X6144, no Convergence Test, T=256)', barList, legendLabels,
        'conv6144.png',
        ylim=[0.1, 400], labelFontSize=10, width=1.2, xxLabels=versionLabels)

legendLabels = ['Total Time']

barPlot(labels, 'Parallel Heat Equation solution Total Time (6144X6144, no Convergence Test, T=256)', barList, legendLabels,
        'conv6144Total.png',
        ylim=[0.1, 250], labelFontSize=10, width=1.2, xxLabels=versionLabels)

legendLabels = [
    'Jacobi 2048', 'Gauss Seidel SOR 2048', 'Red Black SOR 2048',
    'Jacobi 4096', 'Gauss Seidel SOR 4096', 'Red Black SOR 4096',
    'Jacobi 6144', 'Gauss Seidel SOR 6144', 'Red Black SOR 6144'
]
speedupPlot(speedupList, 'Parallel Heat Equation solution Speedup', 'convSpeedup.png',
            labels[1:], legendLabels)

# cudaSeq = kmeansCudaDicts['Sequential Kmeans']
# cudaNaive = kmeansCudaDicts['Naive']
# cudaTranspose = kmeansCudaDicts['Transpose']
# cudaShared = kmeansCudaDicts['Shared']
# cudaAllGPU = kmeansCudaDicts['All']
# cudaAllGPUAdv = kmeansCudaDicts['All (ParallelBlockReduction)']
#
# cudaSeq2 = {k: cudaSeq[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if k in cudaSeq}
#
# cudaNaive2 = {k: cudaNaive[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if k in cudaNaive}
#
# cudaTranspose2 = {k: cudaTranspose[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if
#                   k in cudaTranspose}
#
# cudaShared2 = {k: cudaShared[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if k in cudaShared}
#
# cudaAllGPU2 = {k: cudaAllGPU[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if k in cudaAllGPU}
#
# cudaAllGPUAdv2 = {k: cudaAllGPUAdv[k] for k in [(2, 32), (2, 64), (2, 128), (2, 256), (2, 512), (2, 1024)] if
#                   k in cudaAllGPUAdv}
#
# cudaNaive16 = {k: cudaNaive[k] for k in [(16, 32), (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024)] if
#                k in cudaNaive}
#
# cudaTranspose16 = {k: cudaTranspose[k] for k in [(16, 32), (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024)] if
#                    k in cudaTranspose}
#
# cudaShared16 = {k: cudaShared[k] for k in [(16, 32), (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024)] if
#                 k in cudaShared}
#
# cudaAllGPU16 = {k: cudaAllGPU[k] for k in [(16, 32), (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024)] if
#                 k in cudaAllGPU}
#
# cudaAllGPUAdv16 = {k: cudaAllGPUAdv[k] for k in [(16, 32), (16, 64), (16, 128), (16, 256), (16, 512), (16, 1024)] if
#                    k in cudaAllGPUAdv}
#
# dictList = [cudaSeq2, cudaNaive16, cudaTranspose16, cudaShared16, cudaAllGPU16, cudaAllGPUAdv16]
#
# labels = ['32', '64', '128', '256', '512', '1024']
#
# legendLabels = ['GPU Time', 'CPU Time', 'Memory Transfers Time']
#
# barList = [
#     list(cudaNaive16.values()),
#     list(cudaTranspose16.values()),
#     list(cudaShared16.values()),
#     list(cudaAllGPU16.values()),
#     list(cudaAllGPUAdv16.values())
# ]
#
# versionLabels = ['Naive', 'Transpose', 'Shared', 'All GPU', 'All GPU (Adv.)']
#
# barPlot(labels, 'GPU Kmeans Parallel Detailed Runtimes (numCoords = 16)', barList, legendLabels,
#         'cudaAllVersionsBars16CoordsAdvTimers.png',
#         ylim=[0.1, 3], labelFontSize=10, width=(len(versionLabels) - 2) / len(labels), xxLabels=versionLabels)
# #
# # speedupPlot(dictList, 'GPU Kmeans Parallel Speedup (numCoords = 16)', 'cudaAllVersionsBars16Coords.png', labels[1:],
# #             legendLabels=legendLabels)
# barList = [
#     list(cudaNaive2.values()),
#     list(cudaTranspose2.values()),
#     list(cudaShared2.values()),
#     list(cudaAllGPU2.values()),
#     list(cudaAllGPUAdv2.values())
# ]
#
# dictList = [cudaSeq2, cudaNaive2, cudaTranspose2, cudaShared2, cudaAllGPU2, cudaAllGPUAdv2]
#
# barPlot(labels, 'GPU Kmeans Parallel Detailed Runtimes (numCoords = 2)', barList, legendLabels,
#         'cudaAllVersionsBars2CoordsAdvTimers.png',
#         ylim=[0.1, 20], labelFontSize=10, width=(len(versionLabels) - 2) / len(labels), xxLabels=versionLabels)
# barList = [
#     list(cudaNaive2.values()),
#     list(cudaTranspose2.values()),
#     list(cudaShared2.values()),
#     list(cudaAllGPU2.values()),
#     list(cudaAllGPUAdv2.values())
# ]
# speedupPlot(dictList, 'Kmeans Parallel Speedup', 'allVersionsSpeedup.png', labels[1:],
#             legendLabels=legendLabels)
