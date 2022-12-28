import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import more_itertools as mit

from artificial_experiment import Experiment
from settings import BATCH_SIZE, EPOCS_NUN, ORIGIN_SEQ, WORD_LENGTH, VECTOR_SIZE


class Net(nn.Module):
    '''
    https://madebyollin.github.io/convnet-calculator/
    https://thanos.charisoudis.gr/blog/a-simple-conv2d-dimensions-calculator-logger
    '''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=64, kernel_size=4, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=6, stride=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv2_bn = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(3712, 64)
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.conv2_bn(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc1_bn(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_bn(x)
        x = self.fc3(x)
        return x


def load_data():
    batch_size, channels = BATCH_SIZE, VECTOR_SIZE
    exp = Experiment(num_iter=500, efficiency_interval=1, num_mutations=1)
    sequences = exp.create_exp(ORIGIN_SEQ)
    seq_words = exp.seq_to_words(sequences, word_length=WORD_LENGTH)
    width = len(seq_words[0])  # all sequences have the same length

    mean_channels, std_channels = exp.calculate_batch_norm(seq_words, channels=VECTOR_SIZE, width=len(seq_words[0]))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean_channels, std_channels)])

    trainloader = exp.get_batch(seq_words[:400], batch_size, channels, width, shuffle=True)
    testloader = exp.get_batch(seq_words[400:], batch_size, channels, width, shuffle=False)

    return trainloader, testloader


def train(device, net_path, trainloader):
    net = Net()
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    trainloader = mit.seekable(trainloader)
    for epoch in range(EPOCS_NUN):
        trainloader.seek(0)
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = net(inputs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if i % 1 == 0:
                print(f'Training: The loss in epoch: {epoch} after {i + 1:5d} images is {loss:.3f}')

    print('Finished Training')
    torch.save(net.state_dict(), net_path)


def test(device, net_path, testloader):
    net = Net()
    net.load_state_dict(torch.load(net_path))
    net.to(device)

    total_mse = 0
    total_images = 0
    net.eval()  # no dropout and batch normalization in evaluation step
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            total_images += BATCH_SIZE
            output = net(images)  # patch-size x classes-number
            mse = ((output - labels) ** 2).sum()
            total_mse += mse
        print(f'The mse is: {total_mse / total_images:.3f}')


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_path = r'C:\Users\user\Desktop\IBM-interview\data\protein_net.pth'
    trainloader, testloader = load_data()
    train(device, net_path, trainloader)
    print('Start test:')
    test(device, net_path, testloader)
