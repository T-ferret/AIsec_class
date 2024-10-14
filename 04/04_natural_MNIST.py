from sched import scheduler

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from small_cnn import *


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SmallCNN().to(device)

    # MINST 데이터넷 로드 및 전처리
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)

    # Optimizer 설정
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # 학습률 감소 스케줄러 추가
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    best_acc = 0.0  # 타겟 정확도 저장
    model.train()
    num_epochs = 100  # 학습할 에포크 수
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # 매 에포크마다 평가
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

        acc = 100 * correct / total
        print(f'Epoch {epoch*1}, Loss: {running_loss/len(train_loader)}, Test Accuracy: {acc:.2f}%')

        # 최고 정확도 모델 저장
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), './models/Best_SmallCNN_mnist_natural.pth')
            print(f'Best model saved with accuracy: {best_acc}%')

        model.train() # 학습 모드로 전환

    print(f'Finished Training SmallCNN, best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()