import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, args, dataset=None, idxs=None, bs=None):
        self.args = args
        self.dataset = DataLoader(DatasetSplit(dataset, idxs), batch_size=bs, shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()

    def reptile(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr_meta, momentum=self.args.momentum)
        
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

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr_local, momentum=self.args.momentum)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.dataset):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

        return net.state_dict()

    def local_train(self, net):
        local_train_epoch = 150
        local_train_lr = 0.05

        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=local_train_lr, momentum=self.args.momentum)

        for iter in range(local_train_epoch):
            for batch_idx, (images, labels) in enumerate(self.dataset):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

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
            test_loss += self.loss_func(outputs, labels).sum().item()

        test_loss /= total
        accuracy = 100.00 * correct / total
        print('\nAverage loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, total, accuracy))
        return accuracy, test_loss
