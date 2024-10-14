import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
# from torchvision import transforms


def get_test_image(test_dir, preprocess):
    dir_list = os.listdir(test_dir)
    y_test = []
    # org_imgs = []

    for i, img_name in enumerate(dir_list):
        split_name = img_name.split('.')
        label = float(split_name[0])
        y_test.append(label)
        img_loc = os.path.join(test_dir, img_name)
        img = Image.open(img_loc).convert('RGB')

        if i == 0:
            X_test = preprocess(img).unsqueeze(0)
        else:
            X_test = torch.cat((X_test, preprocess(img).unsqueeze(0)), 0)

    return X_test, np.array(y_test)


def denormalize(images, mean, std):
    denorm_imgs = np.array(images)
    denorm_imgs = denorm_imgs * std + mean
    denorm_imgs = denorm_imgs.transpose(0, 2, 3, 1).clip(0, 1)

    return denorm_imgs


def get_class_name(weights, labels):
    label_names = []

    for label in labels:
        label_name = weights.meta['categories'][label]
        label_names.append(label_name)

    return label_names


def visualize_attacks(org_imgs, adv_exams, noises, org_names, adv_names):
    fig, axes = plt.subplots(3, 5, figsize=(15, 9), sharex=True, sharey=True)

    for i in range(len(org_imgs)):
        axes[0, i].imshow(org_imgs[i])
        axes[0, i].set_title(org_names[i])
        axes[0, i].axis('off')
        axes[1, i].imshow(noises[i], cmap='jet')
        axes[1, i].axis('off')
        axes[2, i].imshow(adv_exams[i])
        axes[2, i].set_title(adv_names[i])
        axes[2, i].axis('off')
    plt.show()