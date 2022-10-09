import argparse
import pickle
from tqdm import tqdm
import jsonlines
from transformers import BertTokenizer
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase
import random


def construct_parallel_data_para(src_path, trg_path):
    parallel_data = []

    with open(src_path, "r") as f:
        src_lines = f.readlines()
    with open(trg_path, "r") as f:
        trg_lines = f.readlines()

    assert len(src_lines) == len(trg_lines)
    print("data size: " + str(len(src_lines)))

    c_no_error_sent = 0
    for src_line, trg_line in zip(src_lines, trg_lines):
        src_items = src_line.strip().split("\t")
        if len(src_items) != 2:
            print(src_line, trg_line)
            continue
        src_sent = src_items[1]
        trg_items = trg_line.strip().split("\t")
        assert trg_items[0] == src_items[0]
        id = trg_items[0]
        trg_sent = trg_items[1]
        if src_sent == trg_sent:
            c_no_error_sent += 1
        parallel_data.append((id, src_sent, trg_sent))
    print("error-free sentences: " + str(c_no_error_sent))

    return parallel_data


def encode_parallel_data(config, parallel_data, normalizer, tokenizer, max_len, writer):
    count = 0
    for item in tqdm(parallel_data):
        data_sample = {}
        if config.normalize == "True":
            src_norm = normalizer.normalize_str(item[1])[:max_len - 2]
            trg_norm = normalizer.normalize_str(item[2])[:max_len - 2]
        else:
            src_norm = item[1][:max_len - 2]
            trg_norm = item[2][:max_len - 2]
        if len(src_norm) != len(trg_norm):
            continue

        src_token_list = list(src_norm)
        trg_token_list = list(trg_norm)
        src_token_list.insert(0, '[CLS]')
        src_token_list.append('[SEP]')
        trg_token_list.insert(0, '[CLS]')
        trg_token_list.append('[SEP]')
        data_sample['id'] = item[0]
        data_sample['src_text'] = src_norm
        data_sample['trg_text'] = trg_norm
        data_sample['input_ids'] = tokenizer.convert_tokens_to_ids(src_token_list)
        data_sample['token_type_ids'] = [0 for i in range(len(src_token_list))]
        data_sample['attention_mask'] = [1 for i in range(len(src_token_list))]
        data_sample['trg_ids'] = tokenizer.convert_tokens_to_ids(trg_token_list)
        data_sample['sequence_cls'] = [0] if src_token_list == trg_token_list else [1]
        data_sample['token_cls'] = [0 if c == data_sample['trg_ids'][k] else 1 for k, c in
                                    enumerate(data_sample['input_ids'])]

        writer.write(data_sample)
        count += 1
    open(config.save_path.replace("jsonl", "txt"), 'w').write(str(count))


def construct_parallel_data_lbl(src_path, trg_path):
    parallel_data = []

    with open(src_path, "r") as f:
        src_lines = f.readlines()
    with open(trg_path, "r") as f:
        trg_lines = f.readlines()

    assert len(src_lines) == len(trg_lines)
    print("data size: " + str(len(src_lines)))

    c_no_error_sent = 0
    for src_line, trg_line in zip(src_lines, trg_lines):
        src_items = src_line.strip().split("\t")
        if len(src_items) != 2:
            print(src_line, trg_line)
            continue
        src_sent = src_items[1]
        trg_items = trg_line.strip().split(", ")
        id = trg_items[0]
        trg_sent = list(src_sent)
        if len(trg_items) == 2:
            c_no_error_sent += 1
        else:
            for i in range(1, len(trg_items), 2):
                trg_sent[int(trg_items[i]) - 1] = trg_items[i + 1]
        trg_sent = "".join(trg_sent)
        parallel_data.append((id, src_sent, trg_sent))

    print("error-free sentences: " + str(c_no_error_sent))

    return parallel_data


def encode_predict_data(config, src_path, normalizer, tokenizer, max_len, writer):
    count = 0

    with open(src_path, "r") as f:
        src_lines = f.readlines()
    if config.target_dir:
        print(config.target_dir)
        trg_lines = open(config.target_dir, "r").readlines()
    else:
        trg_lines = []
    print("data size: " + str(len(src_lines)))

    for k, src_line in enumerate(src_lines):
        data_sample = {}
        src_items = src_line.strip().split("\t")
        if len(src_items) != 2:
            continue
        src_sent = src_items[1]
        id = src_items[0]
        if trg_lines:
            trg_sent = trg_lines[k].strip().split("\t")[1]
        else:
            trg_sent = ''

        if config.normalize == "True":
            src_norm = normalizer.normalize_str(src_sent)[:max_len - 2]
        else:
            src_norm = normalizer.normalize_str(src_sent)[:max_len - 2]

        src_token_list = list(src_norm)
        src_token_list.insert(0, '[CLS]')
        src_token_list.append('[SEP]')
        data_sample['id'] = id
        data_sample['src_text'] = src_sent
        data_sample['input_ids'] = tokenizer.convert_tokens_to_ids(src_token_list)
        data_sample['token_type_ids'] = [0 for i in range(len(src_token_list))]
        data_sample['attention_mask'] = [1 for i in range(len(src_token_list))]
        if trg_sent:
            data_sample['trg_text'] = trg_sent

        writer.write(data_sample)
        # open(config.save_path.replace("jsonl", "txt"), 'w').write(str(count))
        count += 1
    open(config.save_path.replace("jsonl", "txt"), 'w').write(str(count))


def main(config):
    print(config.__dict__)
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    normalizer = normalizers.Sequence([Lowercase()])
    writer = jsonlines.open(config.save_path, 'w')
    if config.target_dir:
        if config.is_test:
            encode_predict_data(config, config.source_dir, normalizer, tokenizer, config.max_len, writer)
        elif config.data_mode == "para":
            parallel_data = construct_parallel_data_para(config.source_dir, config.target_dir)
            encode_parallel_data(config, parallel_data, normalizer, tokenizer, config.max_len, writer)
        elif config.data_mode == "lbl":
            parallel_data = construct_parallel_data_lbl(config.source_dir, config.target_dir)
            encode_parallel_data(config, parallel_data, normalizer, tokenizer, config.max_len, writer)
        else:
            print("Wrong data mode!")
            exit()
    else:
        encode_predict_data(config, config.source_dir, normalizer, tokenizer, config.max_len, writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--is_test", default=False)
    parser.add_argument("--bert_path", default="bert-base-chinese", type=str)
    parser.add_argument("--max_len", default=128, type=int)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--data_mode", required=True, type=str)
    parser.add_argument("--normalize", default="False", type=str)
    args = parser.parse_args()
    main(args)
