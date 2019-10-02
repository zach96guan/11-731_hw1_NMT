# coding=utf-8

# 11-731 hw1
# Author: Zehao Guan (zehaog)

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""


import math
import pickle
import sys
import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence

import numpy as np
import random
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter, input_transpose
from vocab import Vocab, VocabEntry


device = 'cuda' if torch.cuda.is_available() else 'cpu'

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        self.criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        
        # initialize neural network layers...
        self.src_embed = nn.Embedding(len(vocab.src), embed_size, padding_idx=vocab.src.word2id['<pad>'])
        self.tgt_embed = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=vocab.tgt.word2id['<pad>'])

        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size)

        self.decoder_init_cell = nn.Linear(hidden_size * 2, hidden_size)
        # self.dropout = nn.Dropout(dropout_rate)

        self.key_mlp = nn.Linear(hidden_size * 2, hidden_size)
        self.value_mlp = nn.Linear(hidden_size * 2, hidden_size)
        self.output_mlp = nn.Linear(hidden_size * 2, len(vocab.tgt))
        

    def forward(self, src_sents: List[List[str]], tgt_sents: List[List[str]], mode=True):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """
        src_encodings, src_lens, dec_init_state = self.encode(src_sents)
        scores, padded_tgt = self.decode(src_encodings, src_lens, dec_init_state, tgt_sents)

        loss = self.criterion(torch.transpose(scores, 1, 2), padded_tgt[:, 1:])
        return loss


    def sents2tensors(self, sents: List[List[str]], flag: bool):
        """
        :param flag: is True if sent is source; Otherwise, return False
        """
        # src
        if flag:
            src_token = self.vocab.src.words2indices(sents)
            src_tensors = [torch.tensor(t) for t in src_token]
            return src_tensors
        # tgt
        else:
            tgt_token = self.vocab.tgt.words2indices(sents)
            tgt_tensors = [torch.tensor(t) for t in tgt_token]
            return tgt_tensors


    def encode(self, src_sents: List[List[str]]):
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """
        src_tensors = self.sents2tensors(src_sents, True)
        src_lens = [len(s) for s in src_sents]
        # (L, B, E)
        padded_src = pad_sequence(src_tensors)
        padded_src = padded_src.to(device)

        # (L, B, E)
        embed_src = self.src_embed(padded_src)
        embed_src = pack_padded_sequence(embed_src, torch.tensor(src_lens), batch_first=False)

        enc_out, (h0, c0) = self.encoder(embed_src)
        # (L, B, H)
        output, _ = pad_packed_sequence(enc_out)

        # init for decoder
        dec_init_cell = self.decoder_init_cell(torch.cat([c0[0], c0[1]], 1))
        dec_init_state = torch.tanh(dec_init_cell)

        return output, src_lens, (dec_init_state, dec_init_cell)


    def decode(self, src_encodings, src_lens, decoder_init_state, tgt_sents: List[List[str]], mode=True):
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """
        tgt_tensors = self.sents2tensors(tgt_sents, False)
        tgt_lens = [len(s) for s in tgt_sents]
        padded_tgt = pad_sequence(tgt_tensors, batch_first=True)
        padded_tgt = padded_tgt.to(device)

        h, cell = decoder_init_state
        batch_size = padded_tgt.shape[0]
        embed_size = padded_tgt.shape[1]

        # (B, L, E)
        embed_tgt = self.tgt_embed(padded_tgt)

        key = self.key_mlp(src_encodings)
        value = self.value_mlp(src_encodings)
        # (B, H)
        ctx = torch.zeros((batch_size, self.hidden_size))
        ctx = ctx.to(device)

        # get mask
        mask = torch.zeros((batch_size, key.size(0)))
        mask = mask.to(device)
        for i in range(mask.size(0)):
            for j in range(src_lens[i]):
                mask[i][j] = 1.0

        # (B, H, L)
        key = key.permute(1, 2, 0)
        # (B, L, H)
        value = value.permute(1, 0, 2)

        scores = []
        teacher_forcing_rate = 0.1

        for i in range(embed_size - 1):
            # teacher forcing
            if mode:
                if i == 0 or np.random.random() >= teacher_forcing_rate:
                    inp = embed_tgt[:, i, :]
                else:
                    inp = self.tgt_embed(torch.argmax(score_t.squeeze(1), 1))
                
            else:
                inp = self.tgt_embed(torch.argmax(score_t.squeeze(1), 1))
            
            x = torch.cat((inp, ctx), 1)
            h_t, cell_t = self.decoder(x, (h, cell))
            output = query = h_t.unsqueeze(1)

            # attention
            energy = torch.bmm(query, key)
            attn = F.softmax(energy, dim=2)
            mask_attn = F.normalize(attn * mask.unsqueeze(1), p=1, dim=2)

            ctx = torch.bmm(mask_attn, value)
            score_t = self.output_mlp(torch.cat((ctx, output), dim=2))
            scores.append(score_t)

            ctx = ctx.squeeze(1)
            h, cell = h_t, cell_t

        scores = torch.cat(scores, 1)

        return scores, padded_tgt


    def greedy_search(self, src_sent, max_decoding_time_step=70):
        """
        Given a single source sentence, perform greedy search

        Args:
            src_sent: a single tokenized source sentence
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
             value: List[str]: the decoded target sentence, represented as a list of words
        """
        src_encodings, src_lens, (h_t, cell_t) = self.encode([src_sent])
        h, cell = h_t, cell_t

        key = self.key_mlp(src_encodings)
        value = self.value_mlp(src_encodings)
        ctx = torch.zeros((1, self.hidden_size))
        ctx = ctx.to(device)

        mask = torch.zeros((1, key.shape[0]))
        mask = mask.to(device)
        for i in range(mask.shape[0]):
            for j in range(src_lens[i]):
                mask[i][j] = 1.0

        key = key.permute(1, 2, 0)
        value = value.permute(1, 0, 2)

        out = torch.tensor([1]).to(device)
        # output indices
        preds = [1]
        pos = 0

        # follow the decoder part, change mode and input size
        while pos < max_decoding_time_step and preds[-1] != self.vocab.tgt.word2id['</s>']:
            pos += 1

            inp = self.tgt_embed(out)
            x = torch.cat((inp, ctx), 1)

            h_t, cell_t = self.decoder(x, (h, cell)) 
            output = query = h_t.unsqueeze(1)

            energy = torch.bmm(query, key)
            attn = F.softmax(energy, dim=2)
            mask_attn = F.normalize(attn * mask.unsqueeze(1), p=1, dim=2)

            ctx = torch.bmm(mask_attn, value)
            score_t = self.output_mlp(torch.cat((ctx, output), 2))

            # greedy-search
            out = torch.argmax(score_t.squeeze(1), dim=1)
            preds.append(out.cpu().numpy()[0])

            ctx = ctx.squeeze(1)
            h, cell = h_t, cell_t
            # torch.cuda.empty_cache()

        hypotheses = []
        for idx in preds[1:-1]:
            hypotheses.append(self.vocab.tgt.id2word[idx])

        return hypotheses


    def evaluate_ppl(self, dev_data, batch_size=32):
        """
        Evaluate perplexity on dev sentences
    
        Args:
            dev_data: a list of dev sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """
        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`
        with torch.no_grad():
            for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
                loss = self.forward(src_sents, tgt_sents, mode=False)

                cum_loss += loss.item()
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
                cum_tgt_words += tgt_word_num_to_predict

            ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl


    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        return torch.load(model_path)


    def save(self, path: str):
        """
        Save current model to file
        """
        torch.save(model, path)


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict[str, str]):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    lr = float(args['--lr'])
    lr_decay = float(args['--lr-decay'])

    vocab = pickle.load(open(args['--vocab'], 'rb'), encoding="utf-8")

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab).to(device)

    # uniform-init
    for param in model.parameters():
        param.data.uniform_(-0.1, 0.1) 

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    
    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            batch_size = len(src_sents)
        
            optim.zero_grad()
            loss = model(src_sents, tgt_sents)
            report_loss += loss.item()
            cum_loss += loss.item()
            loss.backward()
            optim.step()
            
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=256)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    torch.save(model, model_save_path)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model = torch.load(model_save_path)
                        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr_decay)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def greedy_search(model: NMT, test_data_src: List[List[str]], max_decoding_time_step: int):
    with torch.no_grad():

        hypotheses = []
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.greedy_search(src_sent, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = torch.load(args['MODEL_PATH']).to(device)

    hypotheses = greedy_search(model, test_data_src,
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = hypotheses
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            hyp_sent = ' '.join(hyps)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()

