import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from small_cnn import *
import torchattacks


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_loc = "./models/Best_SmallCNN_mnist_natural.pth"
    model = SmallCNN()
    model.load_state_dict(torch.load(model_loc))
    model.to(device)

    # mnist 데이터셋 로드 및 전처리
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    #'''
    # 타겟 모델의 성능 평가
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    print(f'Accuracy of the SmallCNN on the MNIST test images: {100 * correct / total}%')
    #'''

    bim = torchattacks.BIM(model, eps=0.3, alpha=0.02, steps=40)  # 적대적(기만) 공격 방법 정의
    print(bim)

    total = 0
    successful_attacks = 0
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)

        # 모델이 원본 이미지에 대해 올바른 예측을 했는지 확인
        logits = model(images)
        _, original_pred = torch.max(logits.data, 1)

        if original_pred.item() != labels.item():
            # 모델이 원본 이미지에 대해 이미 잘못된 예측을 했다면, 공격 건너 뛰기
            continue

        adv_exam = bim(image, label) # 적대적 예제 생성

        # 적대적 예제에 대한 모델 예측
        adv_logits = model(adv_exam)
        _, adv_pred = torch.max(adv_logits.data, 1)

        # 공격 성공 여부 확인
        if adv_pred.item() != labels.item():
            successful_attacks += 1

        total += 1

    successful_attacks = successful_attacks / total if total > 0 else 0
    print(f'Successful attacks: {successful_attacks:.4f} ({successful_attacks}/{total})')


if __name__ == '__main__':
    main()