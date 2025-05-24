import matplotlib.pyplot as plt
import numpy as np

'''
QuakNet:
Mean accuaracy for QuackNet:
[0.8214, 0.92508333, 0.94064167, 0.94961667, 0.9558,0.96025, 0.963675, 0.96654167, 0.969, 0.971125]

Mean Loss for QuackNet:
[0.60771607, 0.25617624, 0.20269678, 0.17275837, 0.15279125, 0.137877, 0.12608192, 0.11634947, 0.10815185, 0.10115307]


Pytorch:
Mean accuaracy for Pytorch:
[53.59833333, 84.36466667, 88.71266667, 90.14666667, 90.99133333, 91.6, 92.11633333, 92.56966667, 92.99733333, 93.404]

Mean Loss for Pytorch:
[1.74560474, 0.58685295, 0.40313624, 0.34828176, 0.31788292, 0.29539778, 0.27666921, 0.26018988, 0.24526424, 0.23154016]


Tensorflow:
Mean Accuracy for Tensorflow:
[77.44799971580505, 89.68966603279114, 91.2766683101654, 92.22633361816406, 92.93066620826721, 93.507000207901, 93.97866725921631, 94.3886661529541, 94.75233316421509, 95.05100011825562]

Mean Loss for Tensorflow: 
[0.886668610572815, 0.36731502413749695, 0.30530876517295835, 0.2715087890625, 0.24702780246734618, 0.22744983732700347, 0.2110085219144821, 0.1968971699476242, 0.18457053303718568, 0.1736813634634018]
'''

meanAccuaracyQuackNet = [0.8214, 0.92508333, 0.94064167, 0.94961667, 0.9558,0.96025, 0.963675, 0.96654167, 0.969, 0.971125]
meanLossQuackNet = [0.60771607, 0.25617624, 0.20269678, 0.17275837, 0.15279125, 0.137877, 0.12608192, 0.11634947, 0.10815185, 0.10115307]

meanAccuaracyPytorch = [53.59833333, 84.36466667, 88.71266667, 90.14666667, 90.99133333, 91.6, 92.11633333, 92.56966667, 92.99733333, 93.404]
meanLossPytorch = [1.74560474, 0.58685295, 0.40313624, 0.34828176, 0.31788292, 0.29539778, 0.27666921, 0.26018988, 0.24526424, 0.23154016]

meanAccuaracyTensorflow = [77.44799971580505, 89.68966603279114, 91.2766683101654, 92.22633361816406, 92.93066620826721, 93.507000207901, 93.97866725921631, 94.3886661529541, 94.75233316421509, 95.05100011825562]
meanLossTensorflow = [0.886668610572815, 0.36731502413749695, 0.30530876517295835, 0.2715087890625, 0.24702780246734618, 0.22744983732700347, 0.2110085219144821, 0.1968971699476242, 0.18457053303718568, 0.1736813634634018]

epochs = list(range(1, 11))
figure, axis = plt.subplots(1, 2)

meanAccuaracyQuackNet = np.array(meanAccuaracyQuackNet) * 100

axis[0].plot(epochs, meanAccuaracyQuackNet, 'b',  marker="o", label=f'QuackNet')
axis[0].plot(epochs, meanAccuaracyPytorch, 'r',  marker="x", label=f'Pytorch')
axis[0].plot(epochs, meanAccuaracyTensorflow, 'g',  marker="s", label=f'Tensorflow')
axis[0].set_xticks(epochs)
axis[0].set_xlabel("epochs")
axis[0].set_ylabel("accauracy")
axis[0].set_title("model accuracy")
axis[0].grid(True)
axis[0].legend()

axis[1].plot(epochs, meanLossQuackNet, 'b', marker="o", label=f'QuackNet')
axis[1].plot(epochs, meanLossPytorch, 'r',  marker="x", label=f'Pytorch')
axis[1].plot(epochs, meanLossTensorflow, 'g',  marker="s", label=f'Tensorflow')
axis[1].set_xticks(epochs)
axis[1].set_xlabel("epochs")
axis[1].set_ylabel("loss")
axis[1].set_title("model loss")
axis[1].grid(True)
axis[1].legend()


plt.tight_layout()
plt.show()