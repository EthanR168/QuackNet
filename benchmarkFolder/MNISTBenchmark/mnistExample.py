import numpy as np
import time
from quacknet.lossFunctions import CrossEntropyLossFunction

# Load the preprocessed data
train_images = np.load('benchmarkFolder/MNISTBenchmark/data/train_images.npy')  # Shape: (60000, 784)
train_labels = np.load('benchmarkFolder/MNISTBenchmark/data/train_labels.npy')  # Shape: (60000, 10)

test_images = np.load('benchmarkFolder/MNISTBenchmark/data/test_images.npy')   # Shape: (10000, 784)
test_labels = np.load('benchmarkFolder/MNISTBenchmark/data/test_labels.npy')    # Shape: (10000, 10)

from quacknet.main import Network

learningRate = 0.01

def run(epochs, steps, skipInput = True): 
    if(skipInput == False):
        inp = input("Create new weights/biases (y/n): ").lower()
    else:
        inp = "y"

    if(inp == "y"):
        n.createWeightsAndBiases()
        n.write("benchmarkFolder/MNISTBenchmark/WeightsAndBiases/weights.txt", "benchmarkFolder/MNISTBenchmark/WeightsAndBiases/biases.txt")
    else:
        n.read("benchmarkFolder/MNISTBenchmark/WeightsAndBiases/weights.txt", "benchmarkFolder/MNISTBenchmark/WeightsAndBiases/biases.txt")

    accuracies, losses = [], []
    for epoch in range(0, epochs, steps):
        start = time.time()
        accuaracy, averageLoss = n.train(train_images, train_labels, steps)
        print(f"epoch: {steps * (epoch + 1)}/{epochs*steps}, took: {(time.time() - start)} seconds, accuracy: {round(accuaracy*100,2)}%, average loss: {averageLoss}")
        n.write("benchmarkFolder/MNISTBenchmark/WeightsAndBiases/weights.txt", "benchmarkFolder/MNISTBenchmark/WeightsAndBiases/biases.txt")
        accuracies.append(accuaracy)
        losses.append(averageLoss)
    allAccuracies.append(accuracies)
    allLosses.append(losses)
    allTrainAcc.append(accuracies[len(accuracies) - 1])
    allTrainLoss.append(losses[len(losses) - 1])

    n.write()

allAccuracies, allLosses = [], []
allTestAcc = []
allTestLoss = []
allTrainAcc = []
allTrainLoss = []
for _ in range(5): # 5 runs to get mean
    n = Network(learningRate=learningRate, lossFunc="cross", optimisationFunc="batches", useBatches=True, batchSize=64)
    n.addLayer(784, "relu")
    n.addLayer(128, "relu")
    n.addLayer(64, "relu")
    n.addLayer(10, "softmax")

    run(10, 1, True)

    #### This is to get final accuracy and loss using test dataset
    totalLoss = 0
    totalAccuracy = 0
    for i in range(len(test_images)):
        layers = n.forwardPropagation(test_images[i]) #returns all layers
        lastLayer = len(layers) - 1
        totalLoss += CrossEntropyLossFunction(layers[lastLayer], test_labels[i])
        nodeIndex = np.argmax(layers[lastLayer])
        labelIndex = np.argmax(test_labels[i])
        if(nodeIndex == labelIndex):
            totalAccuracy += 1

    print(f"accuracy: {totalAccuracy / len(test_images) * 100}%, average loss: {totalLoss / len(test_images)}")
    allTestAcc.append(totalAccuracy / len(test_images) * 100)
    allTestLoss.append(totalLoss / len(test_images))

Network.drawGraphs(None, allAccuracies, allLosses)

print(f"=== Final Benchmark Results ===")
print(f"Average Training Loss (Epoch 10): {np.mean(allTrainLoss)}")
print(f"Average Training Accuracy: {np.mean(allTrainAcc)}%")
print(f"Average Test Loss: {np.mean(allTestLoss)}")
print(f"Average Test Accuracy: {np.mean(allTestAcc)}%")


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

epoch: 1/10, took: 23.674540758132935 seconds, accuracy: 80.6%, average loss: 0.6411607890167117
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
epoch: 3/10, took: 22.476791381835938 seconds, accuracy: 94.2%, average loss: 0.1998607326980374
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