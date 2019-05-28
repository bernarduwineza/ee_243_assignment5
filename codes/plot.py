"""
Plot the network characteristics
"""
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import numpy as np

test_accuracy = np.load('test_accuracy.npy')
train_accuracy = np.load('train_accuracy.npy')

train_loss = np.load('train_loss.npy')

print(np.max(train_accuracy))


#####################################################
# Plot the training and test accuracy
#####################################################

fig1, ax1 = plt.subplots()

ax1.set_xlabel('Number of Epochs', color='red')
ax1.set_ylabel('Accuracy')
ax1.plot(test_accuracy, color='red')
ax1.legend(['Test Accuracy'], loc='best')
ax1.tick_params(axis='x')

ax2 = ax1.twiny()

ax2.set_xlabel('Training Iterations', color='dodgerblue')
ax2.plot(train_accuracy, color='dodgerblue')
ax2.legend(['Training Accuracy'], loc='best')
ax2.tick_params(axis='x')

fig1.tight_layout()
plt.show()

#####################################################
# Plot the training loss amd training accuracy
#####################################################
fig2, ax_loss = plt.subplots()

ax_loss.set_xlabel('Training Iterations')
ax_loss.set_ylabel('Training Loss', color='red')
ax_loss.plot(train_loss, color='red')
ax_loss.legend(['Training Loss'], loc='best')
ax_loss.tick_params(axis='y')

ax_train = ax_loss.twinx()

ax_train.set_ylabel('Training Accuracy', color='dodgerblue')
ax_train.plot(train_accuracy, color='dodgerblue')
ax_train.legend(['Training Accuracy'])
ax_train.tick_params(axis='y')

fig2.tight_layout()
plt.show()
