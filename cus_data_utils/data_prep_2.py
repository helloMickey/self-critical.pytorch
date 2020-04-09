import json
from collections import defaultdict
import nltk
import string

json_path = '../data/dataset_rsicd.json'
caption_path = '../data/caption_2.json'
caption_redundancy_path = '../data/caption_redundancy_2.json'
vocab_path = '../data/vocabulary_2.json'
statistic_info_path = '../data/statistic_info.json'

def caption_statistic():
    with open(caption_path) as f:
        captions = json.load(f)
    image_ids = list(captions.keys())

    statistic_info={}
    words_frequence=defaultdict(int)  # 默认值为0
    sten_length=defaultdict(int)
    for k in image_ids:
        v = captions[k]
        sentences = v['sentence']
        for stc in sentences:
            # add (image, sentence) pair
            words = nltk.tokenize.word_tokenize(stc)
            sten_length[len(words)] += 1
            for word in words:
                words_frequence[word] += 1

    temp1 = sorted(words_frequence.items(), key=lambda x: x[1], reverse=True)
    words_count = {}
    for item_tuple in temp1:
        words_count[item_tuple[0]] = item_tuple[1]

    # temp2 = sorted(sten_length.items(), key=lambda x: x[1], reverse=True)
    # sten_count = {}
    # for item_tuple in temp2:
    #     sten_count[item_tuple[0]] = item_tuple[1]
    temp2 = sorted(sten_length.keys(), reverse=True)
    sten_count = {}
    for key in temp2:
        sten_count[key] = sten_length[key]

    statistic_info['words_info'] = words_count
    statistic_info['stens_info'] = sten_count

    json_dic = json.dumps(statistic_info, indent=4)
    with open(statistic_info_path, 'w') as f:
        # json重新组织
        f.write(json_dic)
        pass
    pass


MIN_WORD_FREQ = 2
UNK_TOKEN = "UNK"  # 单词出现次数少于MIN_WORD_FREQ的替换为UNK_TOKEN
# caption字幕的最大长度和最短长度
MAX_STC_LENGHT = 20
MIN_STC_LENGTH = 5


# 生成 caption.json （其中不包含重复的sentence）
def rebuild_json():
    vocab_json = {}
    word2idx = {}
    idx2word = {}
    # default word
    word2idx['<pad>'] = 0
    word2idx['<start>'] = 1
    word2idx['<end>'] = 2
    word2idx[UNK_TOKEN] = 3
    idx = 4

    with open(statistic_info_path) as f:
        statistic_info = json.load(f)
    words_counter = statistic_info["words_info"]

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
            sent = stc['raw']
            words = nltk.tokenize.word_tokenize(sent)
            # 句子长度过滤
            if MIN_STC_LENGTH <= len(words) and len(words) <= MAX_STC_LENGHT:
                for i, w in enumerate(words):
                    # UNK 替换
                    if words_counter[w] < MIN_WORD_FREQ:
                        words[i] = UNK_TOKEN
                    else:
                        if not word2idx.__contains__(w):
                            # 计入词汇表
                            word2idx[w] = idx
                            idx += 1
                pass
                sent = ' '.join(words)
                stc_set.add(sent)

        # 字典中的key对应着imageId
        if len(stc_set) == 0:
            continue
        json_dic[imgid] = {'filename': filename, 'sentence': list(stc_set)}
    json_dic = json.dumps(json_dic, indent=4)
    with open(caption_path, 'w') as f:
        # json重新组织
        f.write(json_dic)
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

# rebuild_json()

# 生成 caption_redundancy.json （包含重复的sentence）
def rebuild_json_redundancy():
    print('rebuild_json_redundancy....')

    with open(statistic_info_path) as f:
        statistic_info = json.load(f)
    words_counter = statistic_info["words_info"]

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
            sent = stc['raw']
            words = nltk.tokenize.word_tokenize(sent)
            # 句子长度过滤
            if MIN_STC_LENGTH <= len(words) and len(words) <= MAX_STC_LENGHT:
                for i, w in enumerate(words):
                    # UNK 替换
                    if words_counter[w] < MIN_WORD_FREQ:
                        words[i] = UNK_TOKEN
                sent = ' '.join(words)
                stc_list.append(sent)
        pass
        # 字典中的key对应着imageId
        if len(stc_list) == 0:
            continue
        json_dic[imgid] = {'filename': filename, 'sentence': list(stc_list)}
    pass
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
    with open('data/caption_redundancy_2.json') as f:
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
    with open('data/caption_2.json') as f:
        raw = json.load(f)
    for id in image_ids:
        caps = raw[str(id)]["sentence"]  # list type
        tokens_list = []
        for cap in caps:
            tokens = nltk.tokenize.word_tokenize(cap)
            tokens_list.append(tokens)
        reference.append(tokens_list)
    return reference


def get_images_name_by_ids(image_ids):
    """
    @param image_ids: List [N]
    @return:  List[N]
    """
    names = []
    # with open(caption_path) as f:
    with open('data/caption_2.json') as f:
        raw = json.load(f)
    for id in image_ids:
        name = raw[str(id)]["filename"]  # list type
        names.append(name)
    return names


LESS_WORD = 2 # 过滤小于 LESS_WORD
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
    # word2idx['<and>'] = 3
    # word2idx['<unk>'] = 4
    idx = 3
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


def load_vocab(path = 'data/vocabulary_2.json'):
    with open(path) as f:
        vocab = json.load(f)
    word2idx = vocab["word2idx"]
    index2word = vocab["idx2word"]
    return word2idx, index2word

if __name__ == '__main__':
    # caption_statistic()
    # rebuild_json()
    # rebuild_json_redundancy()
    pass
