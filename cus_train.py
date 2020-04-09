import argparse
import os
import time
import models
import torch
from misc.loss_wrapper import LossWrapper
import misc.utils as utils
import traceback

def init_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', type=str, default='adam', help='optimization method')
    parser.add_argument('--max_epochs', type=int, default=30, help='epoch nums')

    args = parser.parse_args()
    return args


def train():
    opt = init_opt()
    ################################
    # Build dataloader
    ################################
    loader = None
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # Initialize infos
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }

    ##########################
    # Build model
    ##########################
    model = models.setup(opt).cuda()
    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    lw_model = LossWrapper(model, opt)
    # Wrap with data parallel
    dp_model = torch.nn.DataParallel(model)
    dp_lw_model = torch.nn.DataParallel(lw_model)

    ##########################
    #  Build optimizer
    ##########################
    optimizer = utils.build_optimizer(model.parameters(), opt)

    epoch = 0
    iteration = 0
    # If start self critical training
    sc_flag = False
    # If start structure loss training
    struc_flag = False
    # Assure in training mode
    dp_lw_model.train()

    # Start training
    try:
        while True:
            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break
            start = time.time()
            # Load data from train split (0)
            data = loader.get_batch('train')
            print('Read data:', time.time() - start)

            torch.cuda.synchronize()
            start = time.time()

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            optimizer.zero_grad()
            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'],
                                    torch.arange(0, len(data['gts'])), sc_flag, struc_flag)

            loss = model_out['loss'].mean()

            loss.backward()
            if opt.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' % (opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            end = time.time()

            print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, model_out['reward'].mean(), end - start))

            iteration += 1
            pass
        pass
    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(opt, model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)
    pass