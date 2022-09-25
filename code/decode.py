import torch
from transformers import BertTokenizer
from utils import *
from model import BERT_Model
from tqdm import tqdm
import os
import argparse

class Decoder:
    def __init__(self, config):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.test_loader = init_dataloader(config.test_path, config, "test", self.tokenizer)
        self.model = BERT_Model(config, self.test_loader.dataset)
        self.model.to(self.device)
        self.config = config
        self.top_num = config.top_num
        self.action_mode = config.action_mode

    def __forward_prop(self, dataloader, topk):
        collected_outputs = []
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            _, logits = self.model(**batch)
            logits=torch.nn.functional.normalize(logits, dim=-1)
            ppbs, outputs = torch.topk(logits, topk, dim=-1)
            ppbs = ppbs.tolist()
            outputs = outputs.tolist()
            for ppbs_i, outputs_i in zip(ppbs, outputs):
                tmp = [{k: v for k, v in zip(o, p)} for o, p in zip(outputs_i, ppbs_i)]
                collected_outputs.append(tmp)
        return collected_outputs

    def decode(self):
        model = self.model
        outputs = []
        print(self.config.model_path)
        for model_name in eval(self.config.model_path):
            model.load_state_dict(torch.load(model_name))
            model.eval()
            with torch.no_grad():
                output_tmp = self.__forward_prop(dataloader=self.test_loader, topk=self.top_num)
                if not outputs:
                    outputs = output_tmp
                else:
                    for sk, stc in enumerate(output_tmp):
                        for ck, char in enumerate(stc):
                            for can in char:
                                outputs[sk][ck][can] = char[can] + outputs[sk][ck].get(can, 0)
        if self.action_mode == 'test':
            for sk, stc in enumerate(outputs):
                tmp = []
                for char in stc:
                    best_char, pro = '', 0
                    for can in char:
                        if char[can] > pro:
                            pro = char[can]
                            best_char = can
                    tmp.append(best_char)
                outputs[sk] = tmp
            save_decode_result_test(outputs, self.test_loader.dataset.data, self.config.save_path)
        elif self.action_mode == 'eval':
            for sk, stc in enumerate(outputs):
                tmp = []
                for char in stc:
                    best_char, pro = '', 0
                    for can in char:
                        if char[can] > pro:
                            pro = char[can]
                            best_char = can
                    tmp.append(best_char)
                outputs[sk] = tmp
            save_decode_result_para(outputs, self.test_loader.dataset.data, self.config.save_path)
        elif self.action_mode == 'check':
            for sk, stc in enumerate(outputs):
                outputs[sk] = [dict(sorted([(c, p) for c, p in char.items()],
                                           key=lambda x:x[1], reverse=True)[:self.top_num]) for char in stc]
            save_decode_result_topk(outputs, self.test_loader.dataset.data, self.config.save_path)


def main(config):
    decoder = Decoder(config)
    decoder.decode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--label_ignore_id", default=0, type=int)
    parser.add_argument("--top_num", default=1, type=int)
    parser.add_argument("--action_mode", required=True, type=str)
    args = parser.parse_args()
    main(args)
