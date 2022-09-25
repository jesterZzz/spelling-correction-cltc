from torch.utils.data import DataLoader
from dataset import CSC_Dataset, Padding_in_batch
from eval_char_level import get_char_metrics
from eval_sent_level import get_sent_metrics

vocab_path = "../bert_model/bert-base-chinese/vocab.txt"
vocab = []
with open(vocab_path, "r") as f:
    lines = f.readlines()
for line in lines:
    vocab.append(line.strip())


def init_dataloader(path, config, subset, tokenizer):
    sub_dataset = CSC_Dataset(path, config, subset)

    if subset == "train":
        is_shuffle = True
    else:
        is_shuffle = False

    collate_fn = Padding_in_batch(tokenizer.pad_token_id)

    data_loader = DataLoader(
        sub_dataset,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        collate_fn=collate_fn
    )

    return data_loader


def csc_metrics(pred, gold, src):
    char_metrics = get_char_metrics(pred, gold, src)
    sent_metrics = get_sent_metrics(pred, gold)
    return char_metrics, sent_metrics


def get_best_score(best_score, best_epoch, epoch, *params):
    for para, key in zip(params, best_score.keys()):
        if para > best_score[key]:
            best_score[key] = para
            best_epoch[key] = epoch
    return best_score, best_epoch


def save_decode_result_para(decode_pred, data, path):
    f = open(path, "w")
    results = []
    pred_p, targ_p, tp = 0, 0, 0
    for i, (pred_i, src) in enumerate(zip(decode_pred, data)):
        src_i = src['input_ids']
        line = ""
        pred_i = pred_i[:len(src_i)]
        pred_i = pred_i[1:-1]
        for id, ele in enumerate(pred_i):
            if vocab[ele] not in ["[UNK]", "[SEP]", "[CLS]", "[PAD]"]:
                line += vocab[ele]
            else:
                line += src['src_text'][id]
        f.write("{}\t{}\n".format(src['id'], line))
        if src['src_text'] != src['trg_text']:
            targ_p += 1
        if src['src_text'] != line:
            pred_p += 1
        if src['trg_text'] == line != src['src_text']:
            tp += 1
    f.close()
    p = tp / pred_p if pred_p > 0 else 0
    r = tp / targ_p if targ_p > 0 else 0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    print("Percision: {}, Recall: {}, F1: {}".format(p, r, f))


def save_decode_result_lbl(decode_pred, data, path):
    with open(path, "w") as fout:
        for pred_i, src in zip(decode_pred, data):
            src_i = src['input_ids']
            line = src['id'] + ", "
            pred_i = pred_i[:len(src_i)]
            no_error = True
            for id, ele in enumerate(pred_i):
                if id == 0:
                    continue
                if id == len(pred_i) - 1:
                    continue
                if ele != src_i[id]:
                    if vocab[ele] != "[UNK]":
                        no_error = False
                        line += (str(id) + ", " + vocab[ele] + ", ")
            if no_error:
                line += '0'
            line = line.strip(", ")
            fout.write(line + "\n")


def save_decode_result_topk(decode_pred, data, path):
    f = open(path, "w")
    for i, (pred_i, src) in enumerate(zip(decode_pred, data)):
        src_i = src['input_ids']
        line = ""
        pred_i = pred_i[:len(src_i)]
        pred_i = pred_i[1:-1]
        if len(pred_i) != len(src['trg_text']):
            print(src['src_text'], src['trg_text'])
        for id, eles in enumerate(pred_i):
            if vocab[list(eles.keys())[0]] == src['trg_text'][id]:
                line += src['trg_text'][id]
            elif src['trg_text'][id] in set('—‘’“”qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPLKJHMNBVGFCDXSAZ'):
                line += src['trg_text'][id]
            else:
                candidates = [vocab[ele] for ele in eles]
                candidates = {v: k for v, k in zip(candidates, eles.values())}
                line += '({},{})'.format(src['src_text'][id], src['trg_text'][id]) + str(candidates)

        f.write("{}\t{}\n".format(src['id'], line))

    f.close()


def save_decode_result_test(decode_pred, data, path):
    f = open(path, 'w')
    for i, (pred_i, src) in enumerate(zip(decode_pred, data)):
        src_i = src['input_ids']
        line = ""
        pred_i = pred_i[:len(src_i)]
        pred_i = pred_i[1:-1]
        for id, ele in enumerate(pred_i):
            if src['src_text'][id] in set('—‘’“”qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPLKJHMNBVGFCDXSAZ'):
                line += src['src_text'][id]
            else:
                line += vocab[ele]
        f.write("{}\t{}\n".format(src['id'], line))
    f.close()
