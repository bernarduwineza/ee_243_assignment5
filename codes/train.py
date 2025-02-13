"""
---------------------------------------------------------------------
Training an image classifier
---------------------------------------------------------------------
For this assingment you'll do the following steps in order:
1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network (at least 4 conv layer)
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
---------------------------------------------------------------------
"""

# IMPORTING REQUIRED PACKAGES
import os
import numpy as np
import scipy.io as sio
import torch
import torchvision
import torchvision.transforms as transforms

import cnn_model

import torch.optim as optim
import torch.nn as nn
from multiprocessing import freeze_support


def main():
    # DEFINE VARIABLE
    BATCH_SIZE = 64                # YOU MAY CHANGE THIS VALUE
    EPOCH_NUM = 100                  # YOU MAY CHANGE THIS VALUE
    LR = 0.001                      # YOU MAY CHANGE THIS VALUE
    MODEL_SAVE_PATH = './Models'

    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    # DEFINING TRANSFORM TO APPLY TO THE IMAGES
    # YOU MAY ADD OTHER TRANSFORMS FOR DATA AUGMENTATION
    transform = transforms.Compose(
        [transforms.Resize(32),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ########################################################################
    # 1. LOAD AND NORMALIZE CIFAR10 DATASET
    ########################################################################

    # FILL IN: Get train and test dataset and create respective dataloader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    ########################################################################
    # 2. DEFINE YOUR CONVOLUTIONAL NEURAL NETWORK AND IMPORT IT
    ########################################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = cnn_model.ConvNet().to(device)

    ########################################################################
    # 3. DEFINE A LOSS FUNCTION AND OPTIMIZER
    ########################################################################

    # FILL IN : the criteria for ce loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    ########################################################################
    # 4. TRAIN THE NETWORK
    ########################################################################

    test_accuracy = []
    train_accuracy = []
    train_loss = []

    for epoch in range(EPOCH_NUM):  # loop over the dataset multiple times

        running_loss = 0.0
        test_min_acc = 0
        total = 0
        correct = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # FILL IN: Obtain accuracy for the given batch of data using
            # the formula acc = 100.0 * correct / total where
            # total is the toal number of images processed so far
            # correct is the correctly classified images so far

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_loss.append(running_loss/20)
            train_accuracy.append(100.0*correct/total)

            if i % 20 == 19:    # print every 20 mini-batches
                print('Train: [%d, %5d] loss: %.3f acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20, 100.0*correct/total))
                running_loss = 0.0

        # TEST LEARNT MODEL ON TESTSET
        # FILL IN: to get test accuracy on the entire testset and append
        # it to the list test_accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                # YOUR CODE HERE
                images, labels = data
                labels = labels.to(device)
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_accuracy.append(100.0*correct/total)
        test_ep_acc = test_accuracy[-1]
        print('Test Accuracy: %.3f %%' % test_ep_acc)

        # SAVE BEST MODEL
        if test_min_acc < test_ep_acc:
            test_min_acc = test_ep_acc
            torch.save(net, MODEL_SAVE_PATH + '/my_best_model.pth')
            print("Model saved...")

    np.save('test_accuracy1.npy', test_accuracy)
    sio.savemat('test_accuracy1.mat', mdict={'test_accuracy': test_accuracy})

    np.save('train_accuracy1.npy', train_accuracy)
    sio.savemat('train_accuracy1.mat', mdict={'train_accuracy': train_accuracy})

    np.save('train_loss1.npy', train_loss)
    sio.savemat('train_loss1.mat', mdict={'train_loss': train_loss})

    print('Finished Training')


if __name__ == '__main__':
    freeze_support()
    main()








