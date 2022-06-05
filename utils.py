from gan_parn import generate_fake_samples

import os
import numpy as np

from matplotlib import pyplot as plt

LATENT_DIM = 100

def cal_generate_num(dir, num_classes):
    train_symptom_dir = os.listdir(dir)
    train_symptom_dir.sort()                # The order returned from os.listdir is random...
    class_names = [train_symptom_dir[i] for i in range(num_classes)]
    
    THRESHOLD = 800
    img_num = []
    for idx, class_name in enumerate(class_names):
        img_num.append(len(os.listdir(dir + class_name)))
    
    max_idx = np.argmax(img_num)
    max_num = img_num[max_idx]

    generate_num = []
    for idx, _ in enumerate(class_names):
        if img_num[idx] < THRESHOLD:
            generate_num.append(THRESHOLD - img_num[idx])
        else:
            generate_num.append(0)
    
    return generate_num, class_names

def print_generate_num(dir, class_names, generate_num):
    print("="*100)
    print("Generate dataset Info:")
    for idx, class_name in enumerate(class_names):
        print("Class #{} = {:70} generated {:>5} images".format(idx, class_name, generate_num[idx]))
    print("\n")

def generating_dataset(generators, generate_num):
    """
    Return:
        data (list)
        lables (list)
    """
    data = []
    labels = []
    for idx, generator in enumerate(generators):
        # Why I can't generate 0 samples...
        if (generate_num[idx] == 0):
            continue
        X, _y = generate_fake_samples(generator, LATENT_DIM, generate_num[idx])
        save_plot(X, idx)           # Save the generated images in this round
        data.append(X)
        new_labels = np.linspace(idx, idx, num=generate_num[idx])
        labels.append(new_labels)
        # data = np.concatenate([data, X], axis=0)
        # labels = np.concatenate([labels, new_labels], axis=0)
    return data, labels

# create and save a plot of generated images
def save_plot(examples, idx, n=2):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i])
    # save plot to file
    filename = './image/generated_plot_class%02d.png' % (idx)
    plt.savefig(filename)
    plt.close()
