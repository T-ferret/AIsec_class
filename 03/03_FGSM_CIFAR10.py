import warnings
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from robustbench.utils import load_model
import torchattacks


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Suppress FutureWarning from torch.load
    # warnings.filterwarnings(
    #     "ignore",
    #     message="You are using `torch.load` with `weights_only=False`",
    #     category=FutureWarning,
    # )

    # 타겟 모델 다운 & 로드, WideResNet-28-10
    wide_resnet = load_model('Standard', norm='Linf').to(device)
    wide_resnet.eval()
    print('[Model loaded]\n')

    # CIFAR-10 데이터셋 로드 및 전처리
    transform = transforms.Compose([transforms.ToTensor()])

    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # '''
    # 타겟 모델의 성능 평가
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = wide_resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the Wide-ResNet on CIFAR-10 test images: {100 * correct / total}%\n')
    # '''

    fgsm = torchattacks.FGSM(wide_resnet, eps=8/255)
    print(fgsm)  # 공격 설정 출력

    total = 0
    successful_attacks = 0

    # 공격 수행
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)

        # 모델이 원본 이미지에 대해 올바른 예측을 했는지 확인
        logits = wide_resnet(image)
        _, original_pred = torch.max(logits, 1)

        if original_pred.item() != label.item():
            # 모델이 원본 이미지에 대해 이미 잘못된 예측을 했다면, 건너뛰기
            continue

        adv_exam = fgsm(image, label)

        # 적대적 예제에 대한 모델 예측
        adv_logits = wide_resnet(adv_exam)
        _, adv_pred = torch.max(adv_logits, 1)

        # 공격 성공 여부 확인
        if adv_pred.item() != label.item():
            successful_attacks += 1

        total += 1

    success_rate = successful_attacks / total if total > 0 else 0
    print(f'Attack success rate: {success_rate:.4f} ({successful_attacks}/{total})')


if __name__ == '__main__':
    main()