# -*- coding: utf-8 -*-

import argparse
import math
import os
from collections import OrderedDict

from tqdm import tqdm
from tqdm import trange

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors

import torch
import torch.nn as nn
import torch.optim as optim

import options
import utils
from trainer import Trainer

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.transformer import Transformer


def main(args):
    device = torch.device('cuda' if args.gpu  else 'cpu')

    # construct Field objects
    SRC = data.Field(lower=True, init_token='<bos>', eos_token='<eos>')
    TGT = data.Field(lower=True, init_token='<bos>', eos_token='<eos>')
    fields = [('src', SRC), ('tgt', TGT)]

    slen_filter = lambda x: args.src_minlen <= len(x.src) <= args.src_maxlen \
                    and args.tgt_minlen <= len(x.tgt) <= args.tgt_maxlen
        
    train_data = data.TabularDataset(
        path=args.train,
        format='tsv',
        fields=fields,
        filter_pred=slen_filter,
    )

    valid_data = data.TabularDataset(
        path=args.valid,
        format='tsv',
        fields=fields,
        filter_pred=slen_filter,
    )

    # construct Vocab objects
    SRC.build_vocab(train_data, min_freq=args.src_min_freq)
    if args.src_embed_path is not None:
        vector = utils.load_vector(args.src_embed_path)
        SRC.vocab.load_vectors(vector)

    TGT.build_vocab(train_data, min_freq=args.tgt_min_freq)
    if args.tgt_embed_path is not None:
        vector = utils.load_vector(args.tgt_embed_path)
        TGT.vocab.load_vectors(vector)

    # save fields
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    utils.save_field(args.savedir, fields)
    utils.save_vocab(args.savedir, fields)

    # set iterator
    train_iter, valid_iter = data.BucketIterator.splits(
        (train_data, valid_data), 
        batch_size=args.batch_size,
        sort_within_batch=True,
        sort_key= lambda x: len(x.src),
        repeat=False,
    )

    print(f'| [src] Dictionary: {len(SRC.vocab.itos)} types')
    print(f'| [tgt] Dictionary: {len(TGT.vocab.itos)} types')
    print('')

    for iter_name, iterator in [('train', train_iter), ('valid', valid_iter)]:
        file_path = args.train if iter_name == 'train' else args.valid
        data_object = train_data if iter_name == 'train' else valid_data
        print(f' {iter_name}: {file_path}')
        for name, field in fields:
            n_tokens, n_unk = utils.get_statics(iterator, name, field)
            n_tokens -= 2 * len(data_object) # take <bos> and <eos> from n_tokens
            print(f'| [{name}] {n_tokens} tokens,', end='')
            print(f' coverage: {100*(n_tokens-n_unk)/n_tokens:.{4}}%')
        print('')

    # construct model
    model = Transformer(fields, args).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi['<pad>'])
    optimizer_fn = utils.get_optimizer(args.optimizer)
    optimizer = optimizer_fn(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    trainer = Trainer(model, criterion, optimizer, scheduler, args.clip, iteration=0)

    print('=============== MODEL ===============')
    print(model)
    print('')
    print('=============== OPTIMIZER ===============')
    print(optimizer)
    print('')

   
    epoch = 1
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    best_loss = math.inf

    while epoch < max_epoch and trainer.n_updates < max_update and args.min_lr < trainer.get_lr():
        # train
        with tqdm(train_iter, dynamic_ncols=True) as pbar:
            train_loss = 0.0
            trainer.model.train()
            for samples in pbar:
                bsz = samples.src.size(1)
                srcs = samples.src.to(device)
                tgts = samples.tgt.to(device)
                loss = trainer.step(srcs, tgts)
                train_loss += loss.item()

                # setting of progressbar
                pbar.set_description(f"epoch {str(epoch).zfill(3)}")
                progress_state = OrderedDict(
                    loss=loss.item(),
                    ppl=math.exp(loss.item()),
                    bsz=len(samples),
                    lr=trainer.get_lr(), 
                    clip=args.clip, 
                    num_updates=trainer.n_updates)
                pbar.set_postfix(progress_state)
        train_loss /= len(train_iter)

        print(f"| epoch {str(epoch).zfill(3)} | train ", end="") 
        print(f"| loss {train_loss:.{4}} ", end="")
        print(f"| ppl {math.exp(train_loss):.{4}} ", end="")
        print(f"| lr {trainer.get_lr():.1e} ", end="")
        print(f"| clip {args.clip} ", end="")
        print(f"| num_updates {trainer.n_updates} |")
        
        # validation
        valid_loss = 0.0
        trainer.model.eval()
        for samples in valid_iter:
            bsz = samples.src.size(1)
            srcs = samples.src.to(device)
            tgts = samples.tgt.to(device)
            loss = trainer.step(srcs, tgts)
            valid_loss += loss.item()
        valid_loss /= len(valid_iter)

        print(f"| epoch {str(epoch).zfill(3)} | valid ", end="") 
        print(f"| loss {valid_loss:.{4}} ", end="")
        print(f"| ppl {math.exp(valid_loss):.{4}} ", end="")
        print(f"| lr {trainer.get_lr():.1e} ", end="")
        print(f"| clip {args.clip} ", end="")
        print(f"| num_updates {trainer.n_updates} |")

        # saving model
        save_vars = {
            'epoch': epoch,
            'iteration': trainer.n_updates,
            'best_loss': valid_loss if valid_loss < best_loss else best_loss,
            'args': args,
            'weights': model.state_dict()
        }

        if valid_loss < best_loss:
            filename = os.path.join(args.savedir, 'checkpoint_best.pt') 
            torch.save(save_vars, filename)
        if epoch % args.save_epoch == 0:
            filename = os.path.join(args.savedir, f'checkpoint_{epoch}.pt') 
            torch.save(save_vars, filename)
        filename = os.path.join(args.savedir, 'checkpoint_last.pt') 
        torch.save(save_vars, filename)

        # update
        trainer.scheduler.step(valid_loss)
        epoch += 1

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('''
        An Implimentation of Transformer.
        Attention is all you need. 
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
        Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. 
        In Advances in Neural Information Processing Systems, pages 6000–6010.
    ''')

    options.train_opts(parser)
    options.model_opts(parser)
    args = parser.parse_args()
    main(args)
