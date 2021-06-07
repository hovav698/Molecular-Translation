# for downloading the zip data:
# pip install kaggle
# import kaggle
# type on terminal kaggle competitions download -c bms-molecular-translation

import shutil
import os
import pathlib
import re
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop

#extract the zip file
def extract_data():
    filepath = 'bms-molecular-translation.zip'
    path = pathlib.Path('dataset/train')
    if not path.exists():
        shutil.unpack_archive(filepath, 'dataset')
        os.remove(filepath)

#process to raw text string that it will fit to the tokenizer
def text_spilt(form):
    form = form.split('=')[1]
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1, f"len(num_list) != 1"
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')


def load_and_reformat_image(filename):
    input_image = Image.open(filename)
    input_image = input_image.convert(mode='RGB')
    preprocess = Compose([
        Resize((256, 256), Image.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor


def get_image(image_id):
    fo_id_1 = image_id[0]
    fo_id_2 = image_id[1]
    fo_id_3 = image_id[2]
    fname = 'dataset/train/' + fo_id_1 + "/" + fo_id_2 + "/" + fo_id_3 + "/" + image_id + '.png'
    img = load_and_reformat_image(fname)

    return np.array(img)

# accuracy metric
def calc_accuracy(t, y):
    preds = torch.argmax(y, 2)
    masked_indexes = t != 0
    accuracy = torch.mean((preds[masked_indexes] == t[masked_indexes]).to(dtype=torch.float32))

    return accuracy.item()