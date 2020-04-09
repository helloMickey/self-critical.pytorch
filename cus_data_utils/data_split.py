# 数据集划分
# caption_path = '../data/caption.json'
# split_path = '../data/data_split.json'
caption_path = '../data/caption_2.json'
split_path = '../data/data_split_2.json'
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

import json
import random

def data_split():
    with open(caption_path) as f:
        captions = json.load(f)
    image_ids = list(captions.keys())

    # 将images shuffle过后的进行划分
    random.seed(0)
    random.shuffle(image_ids)
    data = []
    count = 0
    for k in image_ids:
        v = captions[k]
        sentences = v['sentence']
        for stc in sentences:
            # add (image, sentence) pair
            data.append({
                'imgid': k,
                'filename': v['filename'],
                'sentence': stc
            })
            count += 1

    train_size, val_size, test_size = int(len(data) * train_ratio), int(len(data) * val_ratio), int(
        len(data) * test_ratio)
    split_train = data[:train_size]
    split_val = data[train_size:train_size + test_size]
    split_test = data[train_size + test_size:]

    split_json = {}
    split_json['train'] = split_train
    split_json['val'] = split_val
    split_json['test'] = split_test
    with open(split_path, 'w') as f:
        f.write(json.dumps(split_json, indent=4))
    print('data split finish, train_size:{:}\tval_size:{:}\ttest_size{:}'.format(train_size, val_size, test_size))


if __name__ == '__main__':
    data_split()