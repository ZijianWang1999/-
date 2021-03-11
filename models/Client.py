import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.dataset = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()

    def reptile(self, net):
        net.train()

        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        
        k_loss = []
        for batch_idx, (images, labels) in enumerate(self.dataset):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            k_loss.append(loss.item())
        
        return net.state_dict(), sum(k_loss)/len(k_loss)
    
    def fine_tune(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.dataset):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict()

    def test(self, net):
        net.eval()

        # testing
        test_loss = 0
        correct = 0
        total = 0
        for idx, (images, labels) in enumerate(self.dataset):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += self.loss_func(log_probs, labels).sum().item()

        test_loss /= total
        accuracy = 100.00 * correct / total
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, total, accuracy))
        return accuracy, test_loss
