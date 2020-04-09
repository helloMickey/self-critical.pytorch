import json
import nltk
import string
# caption_path = '/disk2/sharing_data/remote sensing caption/RSICD/dataset_rsicd.json'
json_path = '../data/dataset_rsicd.json'
caption_path = '../data/caption.json'
caption_redundancy_path = '../data/caption_redundancy.json'
vocab_path = '../data/vocabulary.json'


# 生成 caption.json （其中不包含重复的sentence）
def rebuild_json():
    with open(json_path) as f:
        raw = json.load(f)
    images = raw['images']
    json_dic = {}
    for item in images:
        filename = item['filename']
        imgid = item['imgid']
        sentences = item['sentences']
        stc_set = set()
        for stc in sentences:  # 去重复
            stc_set.add(stc['raw'])
        # 字典中的key对应着imageId
        json_dic[imgid] = {'filename': filename, 'sentence': list(stc_set)}
    json_dic = json.dumps(json_dic, indent=4)
    with open(caption_path, 'w') as f:
        # json重新组织
        f.write(json_dic)
        pass
    pass
# rebuild_json()

# 生成 caption_redundancy.json （包含重复的sentence）
def rebuild_json_redundancy():
    print('rebuild_json_redundancy....')
    with open(json_path) as f:
        raw = json.load(f)
    images = raw['images']
    json_dic = {}
    for item in images:
        filename = item['filename']
        imgid = item['imgid']
        sentences = item['sentences']
        stc_list = list()
        for stc in sentences:
            stc_list.append(stc['raw'])
        # 字典中的key对应着imageId
        json_dic[imgid] = {'filename': filename, 'sentence': list(stc_list)}
    json_dic = json.dumps(json_dic, indent=4)
    with open(caption_redundancy_path, 'w') as f:
        f.write(json_dic)
        pass
    pass


def get_tokened_caption_by_imageId(image_ids):
    """
    @param image_ids: list of image id [id1, id2, ...]
    @return: list of reference caption [[cap1, cap2,...],[cap1, cap2,...],[cap1, cap2,...]...]
    """
    reference = []
    # with open(caption_redundancy_path) as f:
    with open('data/caption_redundancy.json') as f:
        raw = json.load(f)
    for id in image_ids:
        caps = raw[str(id)]["sentence"]  # list type
        tokens_list = []
        for cap in caps:
            tokens = nltk.tokenize.word_tokenize(cap)
            tokens_list.append(tokens)
        reference.append(tokens_list)
    return reference

# 不包含重复的caption描述
def get_tokened_unique_captions_by_imageID(image_ids):
    """
    @param image_ids: list of image id [id1, id2, ...]
    @return: list of reference caption [[cap1, cap2,...],[cap1, cap2,...],[cap1, cap2,...]...]
    """
    reference = []
    # with open(caption_path) as f:
    with open('data/caption.json') as f:
        raw = json.load(f)
    for id in image_ids:
        caps = raw[str(id)]["sentence"]  # list type
        tokens_list = []
        for cap in caps:
            tokens = nltk.tokenize.word_tokenize(cap)
            tokens_list.append(tokens)
        reference.append(tokens_list)
    return reference

# get_tokened_caption_by_imageId([0,1,2,3])
# get_tokened_unique_captions_by_imageID([0,1,2,3])

def get_images_name_by_ids(image_ids):
    """
    @param image_ids: List [N]
    @return:  List[N]
    """
    names = []
    # with open(caption_path) as f:
    with open('data/caption.json') as f:
        raw = json.load(f)
    for id in image_ids:
        name = raw[str(id)]["filename"]  # list type
        names.append(name)
    return names
# get_images_name_by_ids([1,3,4,6])

LESS_WORD = 1 # 过滤小于 LESS_WORD
def build_vocab():
    with open(caption_path) as f:
        captions = json.load(f)
    vocab_json = {}
    word2idx = {}
    idx2word = {}
    # default word
    word2idx['<pad>'] = 0
    word2idx['<start>'] = 1
    word2idx['<end>'] = 2
    word2idx['<and>'] = 3
    word2idx['<unk>'] = 4
    idx = 5
    word_count = {}
    for k, v in captions.items():
        caps = v['sentence']
        for cap in caps:
            # 标点符号留着
            # # delete punctuation
            # for c in string.punctuation:
            #     cap = cap.replace(c, "")
            # tokenize
            tokens = nltk.tokenize.word_tokenize(cap)
            for tok in tokens:
                if tok in word_count:
                    word_count[tok] += 1
                else:
                    word_count[tok] = 1
    for key in word_count.keys():
        if not word_count[key] < LESS_WORD:
            word2idx[key] = idx
            idx += 1
            pass

    idx = 0
    for word in word2idx.keys():
        idx2word[idx] = word
        idx += 1

    vocab_json['word2idx'] = word2idx
    vocab_json['idx2word'] = idx2word
    vocab_json['size'] = idx

    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab_json, indent=4))
    pass


def load_vocab(path = 'data/vocabulary.json'):
    with open(path) as f:
        vocab = json.load(f)
    word2idx = vocab["word2idx"]
    index2word = vocab["idx2word"]
    return word2idx, index2word


def get_max_sentence_length():
    max_length = -1
    max_cap = ""
    with open(caption_path) as f:
        captions = json.load(f)
    for k, v in captions.items():
        caps = v['sentence']
        for cap in caps:
            # 标点符号留着
            # # delete punctuation
            # for c in string.punctuation:
            #     cap = cap.replace(c, "")
            # tokenize
            tokens = nltk.tokenize.word_tokenize(cap)
            cur_length = len(tokens)
            if cur_length >= max_length:
                max_length = cur_length
                max_cap = cap
    return max_length, max_cap


# rebuild_json_redundancy()
# load_vocab()
# print(get_max_sentence_length())

# if __name__ == '__main__':
#     print("rebuild json...")
#     rebuild_json()
#     print("build vocabulary...")
#     build_vocab()
#     pass