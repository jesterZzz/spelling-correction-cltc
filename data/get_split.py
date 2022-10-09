# encoding:utf-8

import jsonlines
from pyhanlp import HanLP
from tqdm import tqdm
import argparse


def main(config):
    tokenizer = HanLP.newSegment('viterbi').enableCustomDictionary(False)
    writer = jsonlines.open('{}.splitstc.jsonl'.format(config.tag), 'w')
    oris = [_.strip().split('\t', 1)[1] for _ in open(config.input, 'r').readlines()]
    for trg in tqdm(oris):
        stc_list = [_.toString().split('/') for _ in tokenizer.seg(trg)]
        writer.write(stc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default='train', type=str)
    parser.add_argument("--input", default='train.trg', type=str)
    args = parser.parse_args()
    main(args)
