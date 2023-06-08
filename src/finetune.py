import argparse
import glob
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup, AdamW
from nltk.translate import bleu_score, nist_score
from utils import (
    lmap,
    save_json,
    Seq2SeqDataset,
    set_seed
)
from callbacks import get_checkpoint_callback, get_early_stopping_callback


set_seed(42) 
logger = logging.getLogger(__name__)


class DefinitionModule(pl.LightningModule):
    loss_names = ["loss"]
    metric_names = ["bleu", 'nist']
    val_metric = 'losst'

    def __init__(self, hparams, **kwargs):
        super(DefinitionModule, self).__init__()
        self.hparams = hparams
        self.output_dir = self.hparams.output_dir
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name_or_path)

        self.metrics_save_path = Path(self.hparams.output_dir) / "metrics.json"
        self.epoch_count = 0
        self.metrics = defaultdict(list)
        
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
        )
        self.resume_ckpt = self.hparams.resume_ckpt
    
        self.sample = self.hparams.sample
        self.beams_penalty = self.hparams.beams_penalty
        self.beams_group = self.hparams.beams_group
        self.num_beams = self.hparams.num_beams
        self.type_path = ''

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
            "test_val": self.hparams.n_test,            
        }
        

        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
            "test_val": self.hparams.test_max_target_length,            
        }
       
        
        
        self.num_workers = self.hparams.num_workers
        self.decoder_start_token_id = None
        self.dataset_class = Seq2SeqDataset


    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        clean_text = []
        for g in generated_ids:
            gen_text = self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            clean_text.append(gen_text)
        return lmap(str.strip, clean_text)


    def _step(self, batch: dict,) -> Tuple:

        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, target_ids, word_spans = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"], batch['word_spans']
        
        decoder_input_ids = self.model._shift_right(target_ids)
        lm_labels = target_ids
        
        outputs = self(source_ids, attention_mask=source_mask, 
                    decoder_input_ids=decoder_input_ids, 
                    use_cache=False, 
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True
                )
       
        encoder_hiddens = outputs.encoder_last_hidden_state
        decoder_hiddens = outputs.decoder_hidden_states[-1]
            
        
       
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        lm_logits = outputs.logits
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1))
       
        #! contrastive loss
        if self.hparams.contrastive_ratio:
            batch_word_embed = []
            if self.hparams.pooling_method == 'average':
                for idx in range(encoder_hiddens.shape[0]):
                    start_idx, end_idx = word_spans[idx]
                    cur_word_embed = torch.mean(encoder_hiddens[idx,start_idx:end_idx,:], dim=0)
                    batch_word_embed.append(cur_word_embed)

                batch_word_embed = torch.stack(batch_word_embed)#! (batch size, 512)
                batch_definition_embed = torch.mean(decoder_hiddens, dim=1) #! (batch size, 512)
                
            elif self.hparams.pooling_method == 'max':
                for idx in range(encoder_hiddens.shape[0]):
                    start_idx, end_idx = word_spans[idx]
                    cur_word_embed = torch.max(encoder_hiddens[idx,start_idx:end_idx,:], dim=0)[0]
                    batch_word_embed.append(cur_word_embed)

                batch_word_embed = torch.stack(batch_word_embed) #! (batch size, 512)
                batch_definition_embed = torch.max(decoder_hiddens, dim=1)[0] #! (batch size, 512)
                
            batch_word_embed = F.normalize(batch_word_embed, p=2, dim=1)
            batch_definition_embed = F.normalize(batch_definition_embed, p=2, dim=1)
            cosine_sim = torch.matmul(batch_word_embed, batch_definition_embed.T)

            cosine_sim *= 20
            labels = torch.arange(0, batch_word_embed.shape[0], dtype=torch.int64).to('cuda')
            contrastive_loss = F.cross_entropy(input=cosine_sim, target=labels)
            return (loss, contrastive_loss)
        else:
            return (loss, )
   
    def training_step(self, batch, batch_idx) -> Dict:
        def_gen_loss, contrastive_loss = self._step(batch)
        loss = def_gen_loss * (1 - self.hparams.contrastive_ratio) + contrastive_loss * self.hparams.contrastive_ratio
       
        logs = {'loss':loss}
        return {"loss": loss, "log": logs}
    
    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        loss_tensors = self._step(batch)
       
        #! only ppl loss
        loss_tensors = (loss_tensors[0], )
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target)            
        return base_metrics
    
    def validation_step(self, batch, batch_idx) -> Dict:
        metrics = {}
        def_gen_metrics = self._generative_step(batch)
        def_gen_loss, def_gen_bleu, def_gen_nist = def_gen_metrics['loss']
        metrics['def_gen_loss'] = def_gen_loss
        metrics['def_gen_bleu'] = def_gen_bleu
        metrics['def_gen_nist'] = def_gen_nist
        return metrics
    
   
    def validation_epoch_end(self, outputs) -> Dict:
        self.epoch_count += 1
        metrics = {}
        def_gen_loss_list = []
        for metric_dict in outputs:
            def_gen_loss_list.append(metric_dict['def_gen_loss'])
           
        avg_def_gen_loss = torch.stack(def_gen_loss_list).mean().item()
        metrics['avg_def_gen_loss'] = avg_def_gen_loss
        metrics["epoch_count"] = self.epoch_count
        if self.hparams.use_warmup:
            metrics['cur_learning_rate'] = float(self.scheduler.get_lr()[0])
        else:
            metrics['cur_learning_rate'] = self.optimizer.state_dict()['param_groups'][0]['lr']
        metrics['val_avg_loss'] = avg_def_gen_loss
        self.save_metrics(metrics, 'val')
        return {"log": metrics, "val_loss": metrics['val_avg_loss']}

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)
        
 
    def test_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        
        output = {} 
        generated_ids = self.model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                num_beams=self.num_beams,
                repetition_penalty=1.2,
                length_penalty=0.8,        
                no_repeat_ngram_size=1,
                early_stopping=True,
                use_cache=True,
                num_return_sequences=1  
            )
        
        preds = self.ids_to_clean_text(generated_ids)
        targets = self.ids_to_clean_text(y)
        inputs = self.ids_to_clean_text(source_ids)
        
        output['def_gen_inputs'] = inputs
        output['def_gen_preds'] = preds
        output['def_gen_targets'] = targets
        output['words'] = batch["target_word"]
        
        return output
        
    
    def test_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(
            args.output_dir, "{}_def_gen_predictions.txt".format(self.hparams.test_dataset.split('_')[-1]))
        
        with open(output_test_predictions_file, "w",encoding='utf-8') as p_writer:
            for output_batch in outputs:
                for inp, pred, tgt, word in zip(output_batch["def_gen_inputs"], output_batch["def_gen_preds"], output_batch["def_gen_targets"], output_batch['words']):
                    p_writer.writelines(json.dumps({'inp':inp, 'pred':pred, 'tgt':tgt, 'word':word}) + '\n')
            p_writer.close()
        return None
            

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            max_target_length=max_target_length,
            task=self.hparams.task,
            sample=self.hparams.sample,  
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers, 
            sampler=None,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        if self.hparams.shuffle == 'yes':
            dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        else:
            dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=False)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        self.type_path = self.hparams.test_dataset
        return self.get_dataloader(self.type_path, batch_size=self.hparams.eval_batch_size)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        self.optimizer = optimizer
        if self.hparams.use_warmup:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
            )
            gen_scheduler = {'scheduler':scheduler, 'interval':'step'}
            self.scheduler = gen_scheduler['scheduler']
 
            self.opt = optimizer
            return [optimizer], [gen_scheduler]
        else:
            self.opt = optimizer
            return optimizer
    
    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)

        parser.add_argument(
            "--max_source_length",
            default=200,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=150,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=150,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=150,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )     

        parser.add_argument("--logger_name", type=str, choices=["default"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        return parser
    
    
def main(args) -> DefinitionModule:
    
    Path(args.output_dir).mkdir(exist_ok=True)
    model: DefinitionModule = DefinitionModule(args) 
            
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = True 
    else:
        logger = False
    ck = False
    
    if args.resume_ckpt:
        ck = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))[-1]
        
    es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    
   
    if args.ckpt_path:
        ckpt = list(sorted(glob.glob(os.path.join(args.ckpt_path, "*.ckpt"), recursive=True)))[-1]
        state_dict = torch.load(ckpt)['state_dict']
        model.load_state_dict(state_dict)
        print(f'loading from {ckpt} done')
        
    trainer: pl.Trainer = generic_train(
        model,
        args,
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        early_stopping_callback=es_callback,
        resume_from_checkpoint=ck,
        logger=logger,
    )   

    model.hparams.tpu_cores = None
    save_json(model.hparams, args.output_dir+"hparams.json")
    

    if not args.do_predict:
        return model
    
    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
   
    trainer.test(model)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DefinitionModule.add_model_specific_args(parser, os.getcwd())
    parser.add_argument(
            "--use_warmup",
            type=int,
            default=0,
            help="use_warmup or not",
        )
 
    parser.add_argument(
            "--val_check_inter",
            type=int,
            default=None,
            help="every n batch do validation",
        ) 

    parser.add_argument(
            "--ckpt_path",
            type=str,
            default=None,
            help="checkpoint path",
        ) 

    parser.add_argument(
            "--shuffle",
            type=str,
            default='yes',
            help="yes or not for train_dataloader shuffle",
        )  

    parser.add_argument(
            "--pooling_method",
            type=str,
            default='average',
            help="average or max",
        )   

    parser.add_argument(
            "--contrastive_ratio",
            type=float,
            default=0.2,
            help="contrastive task loss ratio",
        )   

    parser.add_argument(
            "--task",
            type=str,
            default='def-gen-with-contras',
            help="def-gen or def-gen-with-contras",
        )
     
    parser.add_argument(
            "--resume_ckpt",
            action="store_true",
            default=False,
            help="resume training",
        )  

    parser.add_argument(
            "--test_dataset",
            type=str,
            default='test',
            help="generate prediction for test set or validation set ",
        )   
    parser.add_argument(
            "--sample",
            type=int,
            default=None,
            help="how many samples are used",
        )       
    parser.add_argument(
            "--beams_penalty",
            type=float,
            default=1.0,
            help="penalty for diverse beam search",
        ) 
    parser.add_argument(
            "--beams_group",
            type=int,
            default=1,
            help="how many group of beams",
        )    
    parser.add_argument(
            "--num_beams",
            type=int,
            default=5,
            help="The number of beam search",
        )      
    args = parser.parse_args()

    main(args)
