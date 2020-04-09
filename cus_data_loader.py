import argparse

import torch, torchvision
import torch.utils.data as Data
import json, os
from PIL import Image
import nltk

from misc.resnet_utils import myResnet
from misc.resnet import resnet50
from torchvision import transforms


class CusDataset(Data.Dataset):
    def __init__(self, mode, opt):
        """
        @type mode: 'train'/'val'/'test'
        """
        self.image_path = opt.cus_image_path
        self.split_path = opt.cus_split_path

        with open(opt.cus_vocab_path) as f:
            self.vocab = json.load(f)
        # self.transform = transform
        self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
        ])
        self.indexs = []
        with open(opt.cus_split_path) as f:
            raw_data = json.load(f)
        self.data = raw_data[mode]
        self.indexs = [i for i in range(len(self.data))]

    def __getitem__(self, index):
        item_dic = self.data[index]
        image_filename = item_dic['filename']
        sentence = item_dic['sentence']
        imageid = item_dic['imgid']

        words = nltk.tokenize.word_tokenize(sentence)
        caption = [] # vocab index
        """
        在句子前后增加<start> <end>，在这里都添加 0
        """
        # caption.append(self.vocab['word2idx']['<start>'])
        caption.append(self.vocab['word2idx']['<pad>'])
        caption.extend([self.vocab['word2idx'][word] for word in words])
        # caption.append(self.vocab['word2idx']['<end>'])
        caption.append(self.vocab['word2idx']['<pad>'])
        caption = torch.IntTensor(caption)
        image = Image.open(os.path.join(self.image_path, image_filename)).convert('RGB')
        image = self.transform(image)

        info = {
            'ix': index,
            'id': imageid,
            'file_path': os.path.join(self.image_path, image_filename)
        }
        label = caption
        gts = caption[1:-1]  # 不计头尾的padding
        return image, label.long(), gts, info

    # {'train': 19466, 'val': 2433, 'test': 2434}
    def __len__(self):
        return len(self.indexs)


def collate_fn(items):
        """
        Creates mini-batch tensors from the list of tuples (image, caption).
        Args:
            data: list of tuple (image, caption).
                - image: torch tensor of shape
                - caption: torch tensor of shape (?); variable length.

        Returns:
            images: torch tensor of images.
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
            image_ids: List; batch中每个image唯一的id
        """
        image_batch, label_batch, gts_batch, info_batch = zip(*items)
        # image_batch, caption_batch, imageid_batch = zip(*items)
        # Merge images (from tuple of 3D tensor to 4D tensor).
        image_batch = torch.stack(image_batch, 0)
        info_batch = list(info_batch)

        lengths = [label.size()[0] for label in label_batch]

        label_batch = torch.nn.utils.rnn.pad_sequence(label_batch, batch_first=True, padding_value=0)
        gts_batch = torch.nn.utils.rnn.pad_sequence(gts_batch, batch_first=True, padding_value=0)


        mask_batch = torch.zeros_like(label_batch)
        for i, len in enumerate(lengths):
            for j in range(len):
                mask_batch[i, j] = 1
        return image_batch, label_batch, mask_batch, gts_batch, info_batch

class CusDataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.loaders, self.iters = {}, {}
        self.max_pos, self.cur_pos = {}, {}  # 记录数据集中数据最大下标，当前下标
        for mode in ['train', 'val', 'test']:
            dataset = CusDataset(mode=mode, opt=self.opt)
            self.loaders[mode] = Data.DataLoader(
                dataset=dataset,
                # shuffle=True,
                # pin_memory=True,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                num_workers=4,
                # num_workers=0
            )
            self.iters[mode] = iter(self.loaders[mode])  # loader 迭代器
            self.cur_pos[mode] = 0
            self.max_pos[mode] = len(dataset)

        self.resnet = myResnet(resnet=resnet50(pretrained=True))
        pass

    def get_batch(self, mode):
        wrapped = False
        try:
            data = next(self.iters[mode])
        except StopIteration:
            wrapped = True
            self.cur_pos[mode] = 0
            self.iters[mode] = iter(self.loaders[mode])
            data = next(self.iters[mode])
        self.cur_pos[mode] += self.batch_size

        (image_batch, label_batch, mask_batch, gts_batch, info_batch) = data

        fc, att = self.resnet(image_batch)
        fc_feats = fc.view(-1, self.opt.fc_feat_size)  # for resnet [N, 2048]
        N, C, H, W = list(att.size())
        att_feats = att.view(N, C, H*W).permute(0, 2, 1)  # for resnet [N, 2048, H, W] => [N , 14*14, 2048]
        att_mask = None
        labels = label_batch
        masks = mask_batch
        gts = gts_batch
        infos = info_batch

        data_batch = {}
        data_batch['fc_feats'] = fc_feats
        data_batch['att_feats'] = att_feats
        data_batch['att_masks'] = att_mask
        data_batch['labels'] = labels
        data_batch['masks'] = masks
        data_batch['gts'] = gts
        # wrapped = False
        # if self.cur_pos[mode] >= self.max_pos[mode]:  # 数据集已经遍历一遍了
        #     wrapped = True
        #     self.cur_pos[mode] = 0
        data_batch['bounds'] = {
            'it_pos_now': self.cur_pos[mode],
            'it_max': self.max_pos[mode],
            'wrapped': wrapped
        }
        data_batch['infos'] = infos

        return data_batch

    def get_vocab(self):
        with open(self.opt.cus_vocab_path) as f:
            vocab = json.load(f)
        return vocab['idx2word']

    def reset_iterator(self, mode):
        self.cur_pos[mode] = 0
        # self.loaders[split].sampler._reset_iter()
        self.iters[mode] = iter(self.loaders[mode])
        pass

    # def load_state_dict(self, state_dict=None):
    #     if state_dict is None:
    #         return
    #     for split in self.loaders.keys():
    #         self.loaders[split].sampler.load_state_dict(state_dict[split])
    pass


    """
    train
    data_batch={
    'fc_feats' = [N, feat_size]  Tensor
    'att_feats' = [N, 0, 0] Tensor
    'att_mask' = None
    'labels' = [N, 5, padding_lengths] N*5*18 Tensor
    [[   0,    1,  433, 1280,  360,   32,   14,   16,   17,    1,  645,
             0,    0,    0,    0,    0,    0,    0],
         [   0, 1637, 1280, 1284,   28, 1285,   32, 3663,   17,  645,    0,
             0,    0,    0,    0,    0,    0,    0],
         [   0,    1,   20,  360,   32,    1,  645,    6,  839, 1283, 1086,
             0,    0,    0,    0,    0,    0,    0],
         [   0, 1280,  360,   98,   32,  839, 1283, 1716,   32,   14,   16,
            17,    1,  645,    0,    0,    0,    0],
         [   0,    1,   38,   39,    1, 1283, 3664,   55,  499,   35,   32,
             1,  645,    3,    1,    2,   32,    0]],
    'masks' = [N, 5, padding_lengths] (1 or 0) Tensor
        [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
          0.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
          0.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
          0.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
          0.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
          1.]],
    'gts' = list [N] value: 5*16
    [[   1   20   64   39    1 1037 1820  370    6  233  720 2690   32   14,   355    0], 
    [   1 2669  298  938  117 1874  114 1347 1348    0    0    0    0    0,     0    0], 
    [   1   20 2287   35   11    1 1562 2596    0    0    0    0    0    0,     0    0],
    [   1   54 4061 2484 2491    1    7 2596    0    0    0    0    0    0,     0    0], 
    [   1   54 4796    1 2596   32   14 1347    0    0    0    0    0    0,     0    0]]
    'bounds' = {'it_pos_now': 10, 'it_max': 5000, 'wrapped': False}
    'infos' = list [N] value: {'ix': 25317, 'id': 497668, 'file_path': 'val2014/COCO_val2014_000000497668.jpg'}
    }
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cus_image_path', type=str, default='/disk2/sharing_data/remote_sensing_caption/RSICD/RSICD_images',
                    help='')
    parser.add_argument('--cus_split_path', type=str, default='cus_data/data_split.json',
                    help='')
    parser.add_argument('--cus_vocab_path', type=str, default='cus_data/vocabulary.json',
                    help='')
    parser.add_argument('--batch_size', type=int, default=32,
                    help='')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='')

    opt = parser.parse_args()
    load = CusDataLoader(opt)

    # 1000
    for i in range(100):
        # data_t = load.get_batch('train')
        data_v = load.get_batch('val')
        # data_t = load.get_batch('test')
    load.reset_iterator('val')
    for i in range(100):
        data_v = load.get_batch('val')

# # test
# if __name__ == '__main__':
#    data_loader = data_loader(phase='train', transform=None, batch_size=5, num_workers=1)
#    for i, (images, captions, lengths) in enumerate(data_loader):
#        pass
#    pass