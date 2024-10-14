from sched import scheduler

import torch
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from small_cnn import *
from small_cnn2 import *


# SmallCNN 모델로부터 예측 레이블 획득 및 원래 레이블과 비교
def get_smcnn_labels_and_compare(model, dataloader, device):
    model.eval()
    labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for image, original_label in dataloader:  # 실제 레이블도 가져와서 비교
            image, original_label = image.to(device), original_label.to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            labels.extend(predicted.cpu().numpy())

            # 예측된 레이블과 원래 레이블 비교
            correct += (predicted == original_label).sum().item()
            total += original_label.size(0)

    acc = 100 * correct / total
    print(f"Accuracy of model on the selected dataset: {acc}% ({correct}/{total})")

    return torch.tensor(labels)


def get_target_results(model, data):

    model.eval()
    labels = []

    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        labels.extend(predicted.cpu().numpy())

    return torch.tensor(labels)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_loc = "..\\models\\Best_SmallCNN_mnist_natural.pth"
    model = SmallCNN()
    model.load_state_dict(torch.load(model_loc))
    model.to(device)

    # MNIST 데이터넷 로드 및 전처리
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(root='..\\data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    test_set = datasets.MNIST(root='..\\data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    # 타겟 모델의 성능 평가
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the SmallCNN on MNIST test images: {100 * correct / total}%")


    # MNIST 데이터셋 로드 및 전처리
    # train_loader에만 데이터 증강 추가
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
    ])

    transform_eval = transforms.Compose([transforms.ToTensor()])

    # MNIST 테스트 세트를 두 개의 절반으로 분할
    test_set = datasets.MNIST(root='..\\data', train=False, transform=transform_eval, download=False)
    test_set_part1, test_set_part2 = torch.utils.data.random_split(test_set, [5000, 5000])

    # train_loader는 데이터 증강 적용
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='..\\data', train=False, download=False, transform=transform_train),
        batch_size=100, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(range(5000))
    )

    # eval_loader는 원본 데이터 사용
    eval_loader = torch.utils.data.DataLoader(test_set_part2, batch_size=100, shuffle=False)

    # SmallCNN2 모델 로드 및 학습(원래 레이블 사용 or VGG11 예측 레이블 사용)
    smcnn2 = SmallCNN2().to(device)

    # Optimizer 설정
    optimizer = optim.SGD(smcnn2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # SmallCNN의 예측 레이블을 학습 데이터셋에 추가
    # pseudo_labels = get_target_results(model, train_loader, device)

    # 학습률 감소 스케줄러 추가
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    best_acc = 0.0  # 최고 정확도 저장
    smcnn2.train()
    num_epochs = 200  # 학습할 에포크 수

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            # 공격자 본인이 label을 사용하는 경우
            labels = labels.to(device)
            # 타겟 모델의 예측 결과를 label로 사용하는 경우
            # labels = pseudo_labels[i * 100:(i + 1) * 100].to(device)

            optimizer.zero_grad()
            outputs = smcnn2(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # 매 에포크마다 평가
        smcnn2.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = smcnn2(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(eval_loader)}, Eval Accuracy: {acc}%")

        # 최고 정확도 모델 저장
        if acc > best_acc:
            best_acc = acc
            torch.save(smcnn2.state_dict(), '..\\models\\best_SmallCNN2_mnist_extracted.pth')
            print(f"Best model saved with accuracy: {best_acc}%")

        smcnn2.train()  # 학습 모드로 전환

    print(f'Finished Training SmallCNN2, best accuracy: {best_acc:.2f}%')

    # 저장된 best_accuracy 모델 불러오기
    clone_model = SmallCNN2()
    clone_model.load_state_dict(torch.load("..\\models\\best_SmallCNN2_mnist_extracted.pth"))
    clone_model.to(device)

    # 모델 추출 공격 평가(정확도)
    clone_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = clone_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the extracted SamllCNN2 model on evaluation dataset: {100 * correct / total}% ({correct}/{total})')

    # 추출된 모델 평가
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, _ in eval_loader:
            images = images.to(device)

            # 두 모델의 예측 결과를 얻음
            target_outputs = model(images)
            surrogate_outputs = clone_model(images)

            _, target_predicted = torch.max(target_outputs.data, 1)
            _, surrogate_predicted = torch.max(surrogate_outputs.data, 1)

            # 두 모델의 예측이 일치하는 경우를 셈
            correct_predictions += (target_predicted == surrogate_predicted).sum().item()
            total_samples += images.size(0)

    fidelity = correct_predictions / total_samples
    print(f'Fidelity between target and surrogate models: {100 * fidelity: .2f}% ({correct_predictions}/{total_samples})')


if __name__ == '__main__':
    main()