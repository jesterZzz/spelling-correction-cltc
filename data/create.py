import random
import json
import jsonlines
from itertools import accumulate
from tqdm import tqdm
import bisect
import opencc
import argparse


class PersonalData():
    def __init__(self, config):
        self.vocab = [_.strip() for _ in open('./all_char.txt', 'r', encoding='utf-8').readlines()]
        self.weighted_conf_pron = json.load(open(r'./final_char_py_conf.json', 'r', encoding='utf-8'))
        self.weighted_conf_shape = json.load(open(r'./final_char_zx_conf.json', 'r', encoding='utf-8'))
        self.word_conf_pron = json.load(open(r'./word_py_conf_filled.json', 'r', encoding='utf-8'))
        self.word_same_means = {}
        tmp = {}
        for line in  open(r'dict_synonym.txt', 'r', encoding='utf-8').readlines():
            line = line.strip().split()[1:]
            for word in line:
                tmp.setdefault(word, []).extend(
                    [_ for _ in line if _ != word and _ not in tmp.get(word, [])])
        self.word_same_means = tmp.copy()
        for word in tmp:
            if not tmp[word]:
                self.word_same_means.pop(word)
        self.config = config
        print('char_level_all_finish')

    def is_keep(self, word):
        for char in word:
            if char not in self.vocab:
                return False
        return True

    def data_handle(self, stc_list):
        tgt_list, state_list, src_list = [], [], []
        loc = 0
        while loc < len(stc_list):
            try:
                word, nature = stc_list[loc]
            except:
                word = '/'
                nature = stc_list[loc][-1]
            src_list.append(word)
            if nature in ["nr", "nr1", "nr2", "nrf", "nrj", "ns", "nsf"]:
                loc += 1
                tgt_list.append(word)
                state_list.append(0)
                continue
            part = stc_list[loc][0]
            if self.is_keep(part):
                try:
                    if state_list[-1] == 0:
                        pro = 10
                    else:
                        pro = 20
                except:
                    pro= 10
                if not random.randint(0, pro):
                    if not random.randint(0, self.config.open_change):
                        tmp = self.replace(part)
                        tgt_list.append(tmp)
                        loc += 1
                        if tmp == part:
                            state_list.append(0)
                        else:
                            state_list.append(1)
                    else:
                        tmp = self.change(part)
                        tgt_list.append(tmp)
                        loc += 1
                        if tmp == part:
                            state_list.append(0)
                        else:
                            state_list.append(1)
                else:
                    loc += 1
                    tgt_list.append(word)
                    state_list.append(0)
            else:
                loc += 1
                tgt_list.append(word)
                state_list.append(1)

        if not random.randint(0, self.config.move_ratio):
            idx_list = [_ for _ in range(len(state_list)) if state_list[_] == 0]
            idx_idx = [_ for _ in range(len(idx_list))]
            if len(idx_list) < 2:
                return ''.join(src_list), ''.join(tgt_list)
            else:
                idx = random.choice(idx_idx)
                another = idx + random.choices([-3, -2, -1, 1, 2, 3], weights=[1, 2, 3, 3, 2, 1])[0]
                if another not in idx_idx:
                    return ''.join(src_list), ''.join(tgt_list)
                tgt_list[idx_list[idx]], tgt_list[idx_list[another]] = tgt_list[idx_list[another]], tgt_list[
                    idx_list[idx]]
        return ''.join(src_list), ''.join(tgt_list)

    def replace(self, part):
        if not random.randint(0, self.config.char_word_ratio) and len(part) > 1 and part in self.word_conf_pron:
            words, weights = list(self.word_conf_pron[part].keys()), list(self.word_conf_pron[part].values())
            return words[WeightedRandomGenerator(weights)()]
        elif len(part) > 1:
            idx = random.randint(0, len(part) - 1)
            changed_char = self.char_level_edits(part[idx])
            return part[:idx] + changed_char + part[idx + 1:]
        else:
            return self.char_level_edits(part)

    def change(self, part):
        mode = random.randint(1, 3)
        if not random.randint(0, self.config.char_word_ratio) and len(part) > 1:
            if mode == 1:
                return part * 2
            elif mode == 2:
                return ''
            elif not random.randint(0, 3):
                return random.choice(self.word_same_means.get(part, [part]))
            else:
                tmp = part
                while tmp == part:
                    tmp = ''.join(random.sample(part, len(part)))
                return tmp
        elif len(part) > 1:
            idx = random.randint(0, len(part) - 1)
            if mode == 1:
                return part[:idx] + part[idx] + part[idx] + part[idx + 1:]
            elif mode == 2:
                return part[:idx] + part[idx + 1:]
            else:
                return part[:idx] + self.char_level_edits(part[idx]) + part[idx + 1:]
        else:
            if mode == 1:
                return part * 2
            elif mode == 2:
                return ''
            else:
                return random.choice(self.word_same_means.get(part, [part]))

    def char_level_edits(self, char):
        if random.randint(0, self.config.py_wb_ratio):
            confset = self.weighted_conf_pron.get(char, {})
        else:
            confset = self.weighted_conf_shape.get(char, {})
        if confset:
            chars = list(confset.keys())
            weights = list(confset.values())
            return chars[WeightedRandomGenerator(weights)()]
        else:
            return char


class WeightedRandomGenerator(object):
    def __init__(self, weights):
        self.totals = []
        self.totals = list(accumulate(weights))

    def next(self):
        rnd = random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return self.next()


def main(config):
    random.seed(config.random_seed)
    used_stc = set()
    count = 1
    pd = PersonalData(config)
    with open('{}-s{}.src'.format(config.tag, config.random_seed), 'w', encoding='utf-8') as src_writer, open(
            '{}-s{}.trg'.format(config.tag, config.random_seed), 'w', encoding='utf-8') as trg_writer:
        with jsonlines.open(config.input, 'r') as reader:
            for stc_part in tqdm(reader):
                used_stc.add(str(stc_part))
        for stc_part in tqdm(used_stc):
            stc_part = eval(stc_part)
            num = 0
            while num < 2:
                stc, changed_stc = pd.data_handle(stc_part)
                if stc != changed_stc:
                    num = 3
                else:
                    num += 1
            src_writer.write("({}-s27-{})\t{}\n".format(config.tag, count, changed_stc))
            trg_writer.write("({}-s27-{})\t{}\n".format(config.tag, count, stc))
            count += 1


if __name__ == '__main__':
    # cc = opencc.OpenCC('t2s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--open_change", default=1, type=int)
    parser.add_argument("--tag", default='pretrain_data', type=str)
    parser.add_argument("--input", default='pd.splitstc.jsonl', type=str)
    parser.add_argument("--char_word_ratio", default=1, type=int)
    parser.add_argument("--py_wb_ratio", default=7, type=int)
    parser.add_argument("--move_ratio", default=5, type=int)
    parser.add_argument("--random_seed", default=1004, type=int)
    args = parser.parse_args()
    main(args)
