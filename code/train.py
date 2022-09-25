from utils import *
from transformers import BertTokenizer, AdamW, get_scheduler
from tqdm import tqdm
from model import BERT_Model
import torch
import os
import argparse
from random import seed


class Trainer:

    def __init__(self, config):
        self.config = config
        self.fix_seed(config.seed)
        print(config.__dict__)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.train_dataloader = init_dataloader(config.train_path, config, "train", self.tokenizer)
        self.valid_dataloader = init_dataloader(config.dev_path, config, "dev", self.tokenizer)
        self.model = BERT_Model(config, config.freeze_bert, config.tie_cls_weight)
        if config.finetune_base:
            checkpoints = torch.load(config.finetune_base)
            self.model.load_state_dict(checkpoints)
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = self.set_scheduler()
        self.best_score = {"valid-c": 0, "valid-s": 0}
        self.best_epoch = {"valid-c": 0, "valid-s": 0}
        self._scaler = torch.cuda.amp.GradScaler()

    def fix_seed(self, seed_num):
        torch.manual_seed(seed_num)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed(seed_num)

    def set_scheduler(self):
        num_epochs = self.config.num_epochs
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        return lr_scheduler

    def __forward_prop(self, dataloader, back_prop=True):
        loss_sum = 0
        collected_outputs = []
        if back_prop:
            for k, batch in enumerate(tqdm(dataloader)):
                with torch.cuda.amp.autocast():
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    loss, logits = self.model(**batch)
                    loss = loss / self.config.accumulation_steps
                    loss_sum += loss.item()
                    self._scaler.scale(loss).backward()
                    if (k + 1) % self.config.accumulation_steps == 0:
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        if (k + 1) % self.config.evaluation_steps == 0:
                            self.__save_ckpt(k+1)
        else:
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, logits = self.model(**batch)
                loss_sum += loss.item()
                outputs = torch.argmax(logits, dim=-1)
                for outputs_i in outputs:
                    collected_outputs.append(outputs_i)
        epoch_loss = loss_sum / len(dataloader)
        return epoch_loss, collected_outputs

    def __save_ckpt(self, epoch):
        save_path = self.config.save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        path = os.path.join(save_path, self.config.tag + f"-epoch-{epoch}.pt")
        torch.save(self.model.state_dict(), path)

    def train(self):
        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            train_loss, _ = self.__forward_prop(self.train_dataloader, back_prop=True)
            self.__save_ckpt(epoch)
            self.model.eval()
            with torch.no_grad():
                valid_loss, valid_output = self.__forward_prop(self.valid_dataloader, back_prop=False)
            print(f"train_loss: {train_loss}, valid_loss: {valid_loss}")
            if not os.path.exists(self.config.save_path + '/tmp/'):
                os.makedirs(self.config.save_path + '/tmp/')
            save_decode_result_lbl(valid_output, self.valid_dataloader.dataset.data,
                                   self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".lbl")
            char_metrics, sent_metrics = csc_metrics(
                self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".lbl",
                self.config.lbl_path,
                self.config.src_path)
            get_best_score(self.best_score, self.best_epoch, epoch,
                           char_metrics["Correction"]["F1"], sent_metrics["Correction"]["F1"])
            #self.__save_ckpt(epoch)
            print(f"curr epoch: {epoch} | curr best epoch {self.best_epoch}")
            print(f"best socre:{self.best_score}")
            print(f"no improve: {epoch - max(self.best_epoch.values())}")
            if (epoch - max(self.best_epoch.values())) >= self.config.patience:
                break


def main(config):
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", required=True, type=str)
    parser.add_argument("--train_path", required=True, type=str)
    parser.add_argument("--dev_path", required=True, type=str)
    parser.add_argument("--src_path", required=True, type=str)
    parser.add_argument("--lbl_path", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--label_ignore_id", default=0, type=int)
    parser.add_argument("--num_epochs", default=30, type=int)
    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--evaluation_steps", default=100000, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--freeze_bert", default=False, type=bool)
    parser.add_argument("--tie_cls_weight", default=False, type=bool)
    parser.add_argument("--weight", default=5., type=float)
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--seed", default=2021, type=int)
    parser.add_argument("--finetune_base", type=str)
    args = parser.parse_args()
    main(args)
