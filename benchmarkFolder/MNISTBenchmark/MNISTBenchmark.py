from turtle import color
import matplotlib.pyplot as plt
import numpy as np

###################### QuackNet ######################
"""
Output:

epoch: 1/10, took: 22.06109118461609 seconds, accuracy: 82.29%, average loss: 0.5943743185859363
epoch: 2/10, took: 22.660807132720947 seconds, accuracy: 92.79%, average loss: 0.24949661380949717
epoch: 3/10, took: 27.020901679992676 seconds, accuracy: 94.23%, average loss: 0.19830215302771018
epoch: 4/10, took: 23.37468981742859 seconds, accuracy: 95.09%, average loss: 0.16979144861105816
epoch: 5/10, took: 21.828123569488525 seconds, accuracy: 95.72%, average loss: 0.1503458440102515
epoch: 6/10, took: 22.54963445663452 seconds, accuracy: 96.18%, average loss: 0.13589833647435506
epoch: 7/10, took: 21.52365756034851 seconds, accuracy: 96.52%, average loss: 0.12439909360071001
epoch: 8/10, took: 24.282421827316284 seconds, accuracy: 96.78%, average loss: 0.11505226024182873
epoch: 9/10, took: 29.059850215911865 seconds, accuracy: 96.97%, average loss: 0.10704721485662386
epoch: 10/10, took: 24.279885292053223 seconds, accuracy: 97.18%, average loss: 0.10014278934287639
accuracy: 96.21%, average loss: 0.1286862093029708

epoch: 1/10, took: 23.59324312210083 seconds, accuracy: 81.52%, average loss: 0.6347092283041816
epoch: 2/10, took: 26.33820343017578 seconds, accuracy: 92.63%, average loss: 0.2541822353693705
epoch: 3/10, took: 24.777700185775757 seconds, accuracy: 94.19%, average loss: 0.20052809360555518
epoch: 4/10, took: 24.97404170036316 seconds, accuracy: 95.12%, average loss: 0.17072894395805352
epoch: 5/10, took: 23.409579277038574 seconds, accuracy: 95.69%, average loss: 0.15067264171457578
epoch: 6/10, took: 23.381500244140625 seconds, accuracy: 96.18%, average loss: 0.13585458578027917
epoch: 7/10, took: 22.76504635810852 seconds, accuracy: 96.49%, average loss: 0.12422157141109996
epoch: 8/10, took: 23.076149225234985 seconds, accuracy: 96.77%, average loss: 0.11445171436780786
epoch: 9/10, took: 23.132285118103027 seconds, accuracy: 97.01%, average loss: 0.1062801190665104
epoch: 10/10, took: 23.6029314994812 seconds, accuracy: 97.27%, average loss: 0.0991990030467749
accuracy: 96.2%, average loss: 0.13051510788060514

epoch: 1/10, took: 23.674540758132935 seconds, accuracy: 80.60%, average loss: 0.6411607890167117
epoch: 2/10, took: 24.415658712387085 seconds, accuracy: 92.57%, average loss: 0.25531046290485454
epoch: 3/10, took: 23.95061731338501 seconds, accuracy: 94.15%, average loss: 0.2010574997574183
epoch: 4/10, took: 27.417540550231934 seconds, accuracy: 95.04%, average loss: 0.17127489077480998
epoch: 5/10, took: 24.42211627960205 seconds, accuracy: 95.67%, average loss: 0.15124863808781733
epoch: 6/10, took: 28.397368907928467 seconds, accuracy: 96.07%, average loss: 0.13658797766777295
epoch: 7/10, took: 22.180707931518555 seconds, accuracy: 96.44%, average loss: 0.1251565769044746
epoch: 8/10, took: 21.673795223236084 seconds, accuracy: 96.73%, average loss: 0.11561693748386864
epoch: 9/10, took: 21.389355659484863 seconds, accuracy: 96.98%, average loss: 0.10756234030360011
epoch: 10/10, took: 21.217774152755737 seconds, accuracy: 97.17%, average loss: 0.1007580491612182
accuracy: 96.24000000000001%, average loss: 0.1272756417172562

epoch: 1/10, took: 21.457317113876343 seconds, accuracy: 81.16%, average loss: 0.6607014022544305
epoch: 2/10, took: 21.734501361846924 seconds, accuracy: 92.73%, average loss: 0.25132668802462804
epoch: 3/10, took: 22.476791381835938 seconds, accuracy: 94.20%, average loss: 0.1998607326980374
epoch: 4/10, took: 22.487247943878174 seconds, accuracy: 95.02%, average loss: 0.17169054297707687
epoch: 5/10, took: 22.407633543014526 seconds, accuracy: 95.61%, average loss: 0.15207953872239444
epoch: 6/10, took: 22.135162115097046 seconds, accuracy: 96.08%, average loss: 0.13743768040705123
epoch: 7/10, took: 22.034822463989258 seconds, accuracy: 96.42%, average loss: 0.1257684649860944
epoch: 8/10, took: 22.280762434005737 seconds, accuracy: 96.73%, average loss: 0.11611608197102513
epoch: 9/10, took: 22.60129451751709 seconds, accuracy: 96.99%, average loss: 0.10789603483369721
epoch: 10/10, took: 22.30544137954712 seconds, accuracy: 97.19%, average loss: 0.10085453620533391
accuracy: 96.31%, average loss: 0.12536046154549305

epoch: 1/10, took: 23.46959352493286 seconds, accuracy: 81.86%, average loss: 0.5979536879810075
epoch: 2/10, took: 23.044170379638672 seconds, accuracy: 92.66%, average loss: 0.2589727638591122
epoch: 3/10, took: 22.89542269706726 seconds, accuracy: 94.15%, average loss: 0.20650116128212298
epoch: 4/10, took: 22.117268323898315 seconds, accuracy: 95.0%, average loss: 0.17624252490628578
epoch: 5/10, took: 21.762017965316772 seconds, accuracy: 95.58%, average loss: 0.15502163246621153
epoch: 6/10, took: 21.644824266433716 seconds, accuracy: 96.01%, average loss: 0.13937783310733276
epoch: 7/10, took: 21.367926597595215 seconds, accuracy: 96.35%, average loss: 0.12719056663936373
epoch: 8/10, took: 21.396846294403076 seconds, accuracy: 96.67%, average loss: 0.11714142138485377
epoch: 9/10, took: 21.772888660430908 seconds, accuracy: 96.93%, average loss: 0.10873976131017862
epoch: 10/10, took: 23.364928722381592 seconds, accuracy: 97.15%, average loss: 0.10157071198052785
accuracy: 96.32%, average loss: 0.12153321724581853

=== Final Benchmark Results ===
Average Training Loss (Epoch 10): 0.10050501794734623
Average Training Accuracy: 0.9718966666666666%
Average Test Loss: 0.12667412753842874
Average Test Accuracy: 96.256%
"""

quackNetAllAccuracies = [
    [82.29, 92.79, 94.23, 95.09, 95.72, 96.18, 96.52, 96.78, 96.97, 97.18],
    [81.52, 92.63, 94.19, 95.12, 95.69, 96.18, 96.49, 96.77, 97.01, 97.27],
    [80.60, 92.57, 94.15, 95.04, 95.67, 96.07, 96.44, 96.73, 96.98, 97.17],
    [81.16, 92.73, 94.20, 95.02, 95.61, 96.08, 96.42, 96.73, 96.99, 97.19],
    [81.86, 92.66, 94.15, 95.00, 95.58, 96.01, 96.35, 96.67, 96.93, 97.15]
]

quackNetAllLosses = [
    [0.5943743185859363, 0.24949661380949717, 0.19830215302771018, 0.16979144861105816, 0.1503458440102515, 0.13589833647435506, 0.12439909360071001, 0.11505226024182873, 0.10704721485662386, 0.10014278934287639],
    [0.6347092283041816, 0.2541822353693705, 0.20052809360555518, 0.17072894395805352, 0.15067264171457578, 0.13585458578027917, 0.12422157141109996, 0.11445171436780786, 0.1062801190665104, 0.0991990030467749],
    [0.6411607890167117, 0.25531046290485454, 0.2010574997574183, 0.17127489077480998, 0.15124863808781733, 0.13658797766777295, 0.1251565769044746, 0.11561693748386864, 0.10756234030360011, 0.1007580491612182],
    [0.6607014022544305, 0.25132668802462804, 0.1998607326980374, 0.17169054297707687, 0.15207953872239444, 0.13743768040705123, 0.1257684649860944, 0.11611608197102513, 0.10789603483369721, 0.10085453620533391],
    [0.5979536879810075, 0.2589727638591122, 0.20650116128212298, 0.17624252490628578, 0.15502163246621153, 0.13937783310733276, 0.12719056663936373, 0.11714142138485377, 0.10873976131017862, 0.10157071198052785]
]

quackNetMeanAccuracy = [81.486, 92.676, 94.184, 95.054, 95.654, 96.104, 96.444, 96.736, 96.976, 97.192]
quackNetMeanLoss = [0.626, 0.254, 0.201, 0.172, 0.152, 0.137, 0.125, 0.116, 0.108, 0.101]

quackNetTestAccuracy = 96.256
quackNetTestLoss = 0.127

###################### PyTorch ######################
"""
output:
=== Trial 1 ===
=== Training Phase ===
Epoch 1: Train Loss = 1.6342, Train Accuracy = 59.73%
Epoch 2: Train Loss = 0.5344, Train Accuracy = 85.50%
Epoch 3: Train Loss = 0.3933, Train Accuracy = 88.95%
Epoch 4: Train Loss = 0.3440, Train Accuracy = 90.11%
Epoch 5: Train Loss = 0.3135, Train Accuracy = 90.92%
Epoch 6: Train Loss = 0.2908, Train Accuracy = 91.65%
Epoch 7: Train Loss = 0.2721, Train Accuracy = 92.17%
Epoch 8: Train Loss = 0.2551, Train Accuracy = 92.70%
Epoch 9: Train Loss = 0.2396, Train Accuracy = 93.17%
Epoch 10: Train Loss = 0.2258, Train Accuracy = 93.49%

=== Inference Phase ===
Final Test Metrics - Loss: 0.2157, Accuracy: 93.74%

=== Trial 2 ===
=== Training Phase ===
Epoch 1: Train Loss = 1.7066, Train Accuracy = 55.02%
Epoch 2: Train Loss = 0.5969, Train Accuracy = 83.98%
Epoch 3: Train Loss = 0.4026, Train Accuracy = 88.80%
Epoch 4: Train Loss = 0.3452, Train Accuracy = 90.25%
Epoch 5: Train Loss = 0.3151, Train Accuracy = 91.02%
Epoch 6: Train Loss = 0.2933, Train Accuracy = 91.64%
Epoch 7: Train Loss = 0.2753, Train Accuracy = 92.12%
Epoch 8: Train Loss = 0.2599, Train Accuracy = 92.51%
Epoch 9: Train Loss = 0.2455, Train Accuracy = 92.94%
Epoch 10: Train Loss = 0.2325, Train Accuracy = 93.36%

=== Inference Phase ===
Final Test Metrics - Loss: 0.2273, Accuracy: 93.48%

=== Trial 3 ===
=== Training Phase ===
Epoch 1: Train Loss = 1.7725, Train Accuracy = 53.32%
Epoch 2: Train Loss = 0.5660, Train Accuracy = 85.11%
Epoch 3: Train Loss = 0.3978, Train Accuracy = 88.88%
Epoch 4: Train Loss = 0.3466, Train Accuracy = 90.14%
Epoch 5: Train Loss = 0.3178, Train Accuracy = 90.86%
Epoch 6: Train Loss = 0.2964, Train Accuracy = 91.45%
Epoch 7: Train Loss = 0.2795, Train Accuracy = 92.01%
Epoch 8: Train Loss = 0.2645, Train Accuracy = 92.42%
Epoch 9: Train Loss = 0.2508, Train Accuracy = 92.81%
Epoch 10: Train Loss = 0.2380, Train Accuracy = 93.21%

=== Inference Phase ===
Final Test Metrics - Loss: 0.2267, Accuracy: 93.58%

=== Trial 4 ===
=== Training Phase ===
Epoch 1: Train Loss = 1.7294, Train Accuracy = 56.27%
Epoch 2: Train Loss = 0.5678, Train Accuracy = 85.12%
Epoch 3: Train Loss = 0.4061, Train Accuracy = 88.78%
Epoch 4: Train Loss = 0.3541, Train Accuracy = 89.94%
Epoch 5: Train Loss = 0.3239, Train Accuracy = 90.75%
Epoch 6: Train Loss = 0.3008, Train Accuracy = 91.39%
Epoch 7: Train Loss = 0.2807, Train Accuracy = 91.97%
Epoch 8: Train Loss = 0.2631, Train Accuracy = 92.51%
Epoch 9: Train Loss = 0.2472, Train Accuracy = 92.94%
Epoch 10: Train Loss = 0.2324, Train Accuracy = 93.38%

=== Inference Phase ===
Final Test Metrics - Loss: 0.2213, Accuracy: 93.77%

=== Trial 5 ===
=== Training Phase ===
Epoch 1: Train Loss = 1.6759, Train Accuracy = 54.54%
Epoch 2: Train Loss = 0.5676, Train Accuracy = 84.50%
Epoch 3: Train Loss = 0.4042, Train Accuracy = 88.60%
Epoch 4: Train Loss = 0.3495, Train Accuracy = 90.14%
Epoch 5: Train Loss = 0.3180, Train Accuracy = 90.92%
Epoch 6: Train Loss = 0.2955, Train Accuracy = 91.56%
Epoch 7: Train Loss = 0.2769, Train Accuracy = 92.04%
Epoch 8: Train Loss = 0.2607, Train Accuracy = 92.50%
Epoch 9: Train Loss = 0.2466, Train Accuracy = 92.94%
Epoch 10: Train Loss = 0.2331, Train Accuracy = 93.34%

=== Inference Phase ===
Final Test Metrics - Loss: 0.2240, Accuracy: 93.33%

=== Final Benchmark Results ===
Average Training Loss (Epoch 10): 0.2324
Average Training Accuracy: 93.36%
Average Test Loss: 0.2230
Average Test Accuracy: 93.58%
"""


PyTorchAllAccuracies = [
    [59.73, 85.50, 88.95, 90.11, 90.92, 91.65, 92.17, 92.70, 93.17, 93.49],
    [55.02, 83.98, 88.80, 90.25, 91.02, 91.64, 92.12, 92.51, 92.94, 93.36],
    [53.32, 85.11, 88.88, 90.14, 90.86, 91.45, 92.01, 92.42, 92.81, 93.21],
    [56.27, 85.12, 88.78, 89.94, 90.75, 91.39, 91.97, 92.51, 92.94, 93.38],
    [54.54, 84.50, 88.60, 90.14, 90.92, 91.56, 92.04, 92.50, 92.94, 93.34]
]

PyTorchAllLosses = [
    [1.6342, 0.5344, 0.3933, 0.3440, 0.3135, 0.2908, 0.2721, 0.2551, 0.2396, 0.2258],
    [1.7066, 0.5969, 0.4026, 0.3452, 0.3151, 0.2933, 0.2753, 0.2599, 0.2455, 0.2325],
    [1.7725, 0.5660, 0.3978, 0.3466, 0.3178, 0.2964, 0.2795, 0.2645, 0.2508, 0.2380],
    [1.7294, 0.5678, 0.4061, 0.3541, 0.3239, 0.3008, 0.2807, 0.2631, 0.2472, 0.2324],
    [1.6759, 0.5676, 0.4042, 0.3495, 0.3180, 0.2955, 0.2769, 0.2607, 0.2466, 0.2331]
]

PyTorchMeanAccuracies = [55.776, 84.842, 88.802, 90.116, 90.894, 91.538, 92.062, 92.528, 92.960, 93.356]
PyTorchMeanLoss = [1.704, 0.567, 0.401, 0.348, 0.318, 0.295, 0.277, 0.261, 0.246, 0.232]

PyTorchTestAccuracy = 93.58
PyTorchTestLoss = 0.223

###################### TensorFlow ######################
"""
output:

=== Trial 1 ===
=== Training Phase ===
Epoch 1: Loss = 0.8573, Accuracy = 77.89%, Time = 2.52s
Epoch 2: Loss = 0.3637, Accuracy = 89.85%, Time = 2.30s
Epoch 3: Loss = 0.3021, Accuracy = 91.39%, Time = 2.38s
Epoch 4: Loss = 0.2686, Accuracy = 92.33%, Time = 2.50s
Epoch 5: Loss = 0.2446, Accuracy = 92.99%, Time = 2.36s
Epoch 6: Loss = 0.2258, Accuracy = 93.56%, Time = 2.49s
Epoch 7: Loss = 0.2097, Accuracy = 94.01%, Time = 2.24s
Epoch 8: Loss = 0.1953, Accuracy = 94.45%, Time = 2.55s
Epoch 9: Loss = 0.1834, Accuracy = 94.78%, Time = 2.40s
Epoch 10: Loss = 0.1727, Accuracy = 95.07%, Time = 2.19s

=== Inference Phase ===
Test Accuracy: 95.15%, Test Loss: 0.1669


=== Trial 2 ===
=== Training Phase ===
Epoch 1: Loss = 0.8411, Accuracy = 78.57%, Time = 2.93s
Epoch 2: Loss = 0.3643, Accuracy = 89.86%, Time = 2.33s
Epoch 3: Loss = 0.3064, Accuracy = 91.18%, Time = 2.38s
Epoch 4: Loss = 0.2746, Accuracy = 92.18%, Time = 2.16s
Epoch 5: Loss = 0.2515, Accuracy = 92.76%, Time = 2.55s
Epoch 6: Loss = 0.2325, Accuracy = 93.37%, Time = 2.50s
Epoch 7: Loss = 0.2169, Accuracy = 93.81%, Time = 2.29s
Epoch 8: Loss = 0.2032, Accuracy = 94.20%, Time = 1.98s
Epoch 9: Loss = 0.1907, Accuracy = 94.54%, Time = 2.33s
Epoch 10: Loss = 0.1795, Accuracy = 94.84%, Time = 2.50s

=== Inference Phase ===
Test Accuracy: 94.63%, Test Loss: 0.1796


=== Trial 3 ===
=== Training Phase ===
Epoch 1: Loss = 0.9151, Accuracy = 75.83%, Time = 2.79s
Epoch 2: Loss = 0.3746, Accuracy = 89.64%, Time = 2.41s
Epoch 3: Loss = 0.3072, Accuracy = 91.24%, Time = 2.41s
Epoch 4: Loss = 0.2722, Accuracy = 92.22%, Time = 2.09s
Epoch 5: Loss = 0.2474, Accuracy = 92.90%, Time = 2.47s
Epoch 6: Loss = 0.2274, Accuracy = 93.52%, Time = 2.58s
Epoch 7: Loss = 0.2108, Accuracy = 93.97%, Time = 2.55s
Epoch 8: Loss = 0.1965, Accuracy = 94.37%, Time = 2.55s
Epoch 9: Loss = 0.1842, Accuracy = 94.67%, Time = 2.17s
Epoch 10: Loss = 0.1730, Accuracy = 94.98%, Time = 2.11s

=== Inference Phase ===
Test Accuracy: 94.96%, Test Loss: 0.1721


=== Trial 4 ===
=== Training Phase ===
Epoch 1: Loss = 0.9143, Accuracy = 75.66%, Time = 2.64s
Epoch 2: Loss = 0.3729, Accuracy = 89.54%, Time = 2.63s
Epoch 3: Loss = 0.3109, Accuracy = 91.09%, Time = 2.51s
Epoch 4: Loss = 0.2768, Accuracy = 92.17%, Time = 2.23s
Epoch 5: Loss = 0.2515, Accuracy = 92.84%, Time = 2.23s
Epoch 6: Loss = 0.2314, Accuracy = 93.40%, Time = 2.54s
Epoch 7: Loss = 0.2145, Accuracy = 93.91%, Time = 2.39s
Epoch 8: Loss = 0.1999, Accuracy = 94.33%, Time = 2.31s
Epoch 9: Loss = 0.1871, Accuracy = 94.69%, Time = 2.23s
Epoch 10: Loss = 0.1759, Accuracy = 94.97%, Time = 2.18s

=== Inference Phase ===
Test Accuracy: 94.89%, Test Loss: 0.1754


=== Trial 5 ===
=== Training Phase ===
Epoch 1: Loss = 0.8748, Accuracy = 77.79%, Time = 2.70s
Epoch 2: Loss = 0.3665, Accuracy = 89.78%, Time = 2.53s
Epoch 3: Loss = 0.3055, Accuracy = 91.31%, Time = 2.23s
Epoch 4: Loss = 0.2726, Accuracy = 92.19%, Time = 2.38s
Epoch 5: Loss = 0.2490, Accuracy = 92.82%, Time = 2.35s
Epoch 6: Loss = 0.2305, Accuracy = 93.43%, Time = 2.15s
Epoch 7: Loss = 0.2153, Accuracy = 93.85%, Time = 1.85s
Epoch 8: Loss = 0.2018, Accuracy = 94.25%, Time = 2.35s
Epoch 9: Loss = 0.1904, Accuracy = 94.58%, Time = 2.30s
Epoch 10: Loss = 0.1794, Accuracy = 94.90%, Time = 2.22s

=== Inference Phase ===
Test Accuracy: 94.77%, Test Loss: 0.1788


Average Training Loss (Epoch 10): 0.1761
Average Training Accuracy: 94.95%
Average Test Loss: 0.1746
Average Test Accuracy: 94.88%
"""


TensorFlowAllAccuracies = [
    [77.89, 89.85, 91.39, 92.33, 92.99, 93.56, 94.01, 94.45, 94.78, 95.07],
    [78.57, 89.86, 91.18, 92.18, 92.76, 93.37, 93.81, 94.20, 94.54, 94.84],
    [75.83, 89.64, 91.24, 92.22, 92.90, 93.52, 93.97, 94.37, 94.67, 94.98],
    [75.66, 89.54, 91.09, 92.17, 92.84, 93.40, 93.91, 94.33, 94.69, 94.97],
    [77.79, 89.78, 91.31, 92.19, 92.82, 93.43, 93.85, 94.25, 94.58, 94.90]
]

TensorFlowAllLosses = [
    [0.8573, 0.3637, 0.3021, 0.2686, 0.2446, 0.2258, 0.2097, 0.1953, 0.1834, 0.1727],
    [0.8411, 0.3643, 0.3064, 0.2746, 0.2515, 0.2325, 0.2169, 0.2032, 0.1907, 0.1795],
    [0.9151, 0.3746, 0.3072, 0.2722, 0.2474, 0.2274, 0.2108, 0.1965, 0.1842, 0.1730],
    [0.9143, 0.3729, 0.3109, 0.2768, 0.2515, 0.2314, 0.2145, 0.1999, 0.1871, 0.1759], 
    [0.8748, 0.3665, 0.3055, 0.2726, 0.2490, 0.2305, 0.2153, 0.2018, 0.1904, 0.1794]
]

TensorFlowMeanAccuracies = [77.148, 89.734, 91.242, 92.218, 92.862, 93.456, 93.91, 94.32, 94.652, 94.952]      
TensorFlowMeanLoss = [0.881, 0.368, 0.306, 0.273, 0.249, 0.239, 0.213, 0.199, 0.187, 0.176]

TensorFlowTestAccuracy = 94.88
TensorFlowTestLoss = 0.175

###################### Draw Graphs ######################

epochs = list(range(1, 11))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Reduced from 15x10 to 12x8
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Tighter spacing

# 1. Training Accuracy (Top Left)
axes[0,0].plot(epochs, quackNetMeanAccuracy, 'b-', marker="o", markersize=4, label='QuackNet')
axes[0,0].plot(epochs, PyTorchMeanAccuracies, 'r--', marker="x", markersize=4, label='PyTorch')
axes[0,0].plot(epochs, TensorFlowMeanAccuracies, 'g:', marker="s", markersize=4, label='TensorFlow')
axes[0,0].set_xticks(epochs)
axes[0,0].set_title("Training Accuracy", fontsize=10)
axes[0,0].legend(fontsize=8)

# 2. Training Loss (Top Right)
axes[0,1].plot(epochs, quackNetMeanLoss, 'b-', marker="o", markersize=4, label='QuackNet')
axes[0,1].plot(epochs, PyTorchMeanLoss, 'r--', marker="x", markersize=4, label='PyTorch')
axes[0,1].plot(epochs, TensorFlowMeanLoss, 'g:', marker="s", markersize=4, label='TensorFlow')
axes[0,1].set_xticks(epochs)
axes[0,1].set_title("Training Loss", fontsize=10)
axes[0,1].legend(fontsize=8)

# 3. Test Accuracy (Bottom Left)
frameworks = ["QuackNet", "PyTorch", "TensorFlow"]
x_pos = np.arange(len(frameworks))
bars = axes[1,0].bar(x_pos, [96.256, 93.58, 94.88], width=0.5, 
                    color=["blue", "red", "green"])
axes[1,0].set_xticks(x_pos)
axes[1,0].set_xticklabels(frameworks, rotation=45, ha="right", fontsize=8)
axes[1,0].set_title("Test Accuracy", fontsize=10)
axes[1,0].set_ylim(90, 100)

# 4. Test Loss (Bottom Right)
bars = axes[1,1].bar(x_pos, [0.127, 0.223, 0.175], width=0.5,
                    color=["blue", "red", "green"])
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(frameworks, rotation=45, ha="right", fontsize=8)
axes[1,1].set_title("Test Loss", fontsize=10)
axes[1,1].set_ylim(0, 0.25)

# Shared axis labels
fig.text(0.5, 0.04, 'Epochs / Frameworks', ha='center', fontsize=9)
fig.text(0.04, 0.5, 'Accuracy (%) / Loss', va='center', rotation='vertical', fontsize=9)

plt.tight_layout(pad=2.0)  # Prevents label overlap
plt.show()