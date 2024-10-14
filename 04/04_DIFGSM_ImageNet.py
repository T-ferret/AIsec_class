from torchvision.models import resnet50, ResNet50_Weights
from utils import *
import numpy as np
import torchattacks

def main():
    test_dir = '..\\..\\test_data'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    # RGB 채널에만 평균과 편차를 구하도록 reshape
    mean = np.array(preprocess.mean).reshape(1, 3, 1, 1)
    std = np.array(preprocess.std).reshape(1, 3, 1, 1)

    X_test, y_test = get_test_image(test_dir, preprocess)
    X_test = X_test.to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    org_imgs = denormalize(X_test.cpu(), mean, std).transpose(0, 3, 1, 2)
    org_imgs = torch.from_numpy(org_imgs).float().to(device)

    # 정답 출력
    print(y_test)

    # 테스트 이미지 예측
    output = model(X_test)
    target_classes = output.argmax(dim=1)

    # 예측 레이블 출력
    print('원본 이미지 예측: ', target_classes.cpu().numpy())
    pred_names = get_class_name(weights, target_classes)

    dim = torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)  # 적대적(기만) 공격 방법 정의
    dim.set_normalization_used(mean=preprocess.mean, std=preprocess.std)
    print(dim)  # 공격 설정 출력
    adv_exams = dim(org_imgs, y_test)

    # 적대적 예제 공격 성공 확인
    adv_output = model(adv_exams)
    adv_classes = adv_output.argmax(dim=1)
    print('적대적 예제 예측: ', adv_classes.cpu().numpy())
    adv_names = get_class_name(weights, adv_classes)

    # 노이즈 계산
    noises = adv_exams.cpu().numpy() - org_imgs.detach().cpu().numpy()
    noises = noises.transpose(0, 2, 3, 1)

    adv_exams = np.clip(adv_exams.cpu().numpy().transpose(0, 2, 3, 1), 0, 1)
    org_imgs = org_imgs.cpu().numpy().transpose(0, 2, 3, 1)

    visualize_attacks(org_imgs, adv_exams, noises, pred_names, adv_names)


if __name__ == '__main__':
    main()