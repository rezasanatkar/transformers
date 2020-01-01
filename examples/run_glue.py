# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer,
                                  AlbertConfig,
                                  AlbertForSequenceClassification, 
                                  AlbertTokenizer,
                                )

from transformers import AdamW, get_linear_schedule_with_warmup
#AdamW and get_linear_schedule_with_warmup are defined at optimization.py

from transformers import glue_compute_metrics as compute_metrics
#glue_compute_metrics is defined at transformers/data/metrics/__ini__.py

from transformers import glue_output_modes as output_modes
#glue_output_modes is defined at transformers/data/processors/glue.py and is the following:
#glue_output_modes = {
#    "cola": "classification",
#    "mnli": "classification",
#    "mnli-mm": "classification",
#    "mrpc": "classification",
#    "sst-2": "classification",
#    "sts-b": "regression",
#    "qqp": "classification",
#    "qnli": "classification",
#    "rte": "classification",
#    "wnli": "classification",
#}


from transformers import glue_processors as processors
#glue_processors is defined at transformers/data/processors/glue.py
#glue_processors is a python dict with string key name of tasks and values being processor classes that are able to read the train and dev datasets of
#those tasks and restun python lists of train and dev InputExample


from transformers import glue_convert_examples_to_features as convert_examples_to_features
#glue_convert_examples_to_features is defined at transformers/data/processors/glue.py

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
}


def set_seed(args):
    #args.seed is not set for glue script and it will be defaulted to its default value of 42.
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #in above, we set the seed values for the three potential modules that rely on random generators
    
    if args.n_gpu > 0:
        #for linux machine, n_gpu will be equal to 2 and for mac, it will be equal to 0. 
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    #train_dataset is TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels).
    #model is an instance of BertForSequenceClassification
    #tokenizer is an instance of BertTokenizer. Note that the tokenizer is already used to create train_dataset and therefore, it won't be used to create
    #train data, but it will be passed to the evaluation job.
    
    """ Train the model """
    #for glue script, local_rank is -1
    if args.local_rank in [-1, 0]:
        #SummaryWriter is the summary writer for tensorboard
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    #in glue script, per_gpu_train_batch_size is set to 8.
    #for the linux machine, args.n_gpu will be 2. Therefore, for mac, args.train_batch_size will be equal to 8 and for linux machine,
    #args.train_batch_size will be equal to 16.

    #for glue script, local_rank is equal to -1
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    #RandomSampler is part of torch.utils.data which takes an object of torch.utils.data.Dataset and return a shuffled Dataset version of it.
    #DistributedSampler is used for distributed sampling as part of DistributedDataParallel
    
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    #args.trin_batch_size will be 8 for mac and 16 for linux machine

    # ** why use of sampler instead of shuffle = True for the above DataLoader?
    #in the case of cpu trainig or single-machine multi-gpu training using DataParallel, it seems that using sampler=train_sampler is equivalent with
    #shuffle = True. However, for multi-machine multi-gpu training using DistributedDataParallel, we cannot use shuffle = True and we need to use
    #DistributedSampler. That is the reason, we cannot use shuffle = True in above. 
    
    if args.max_steps > 0:#max_steps is not set in glue script and it will back-off to its -1 default value
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        #this if for bert mrpc
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        #for mrpc, len(train_dataloader) is equal to 459 which makes sense since the number of mrpc training sequences is equal to 3668 and batch size is
        #equal to 8. Therefore, 459 * 8 = 3672.
        #for bert mrpc, gradient_accumulation_steps is equal to 1 and num_train_epochs is equal to 3.0

    # ** what the method named_parameters() return?
    #it returns python generater of tuples (parameter_name (str), torch.nn.parameter.Parameter) for each paramter of the model. Therefore, it is different
    #from the parameters() method that only returns a list of torch.nn.parameter.Parameter objects.

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #for glue, args.weight_decay is equal to 0.0
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # ** what is the objective of optimizer_grouped_parameters?
    #the mian objective it to partition the parameters (learnable paramters of the model) into two groups such that for the first group weight_decay
    #regularization is used but for the second groupd weight_decay regularization is not applied. In particular, they don't want to use weight_decay
    #regularization for all the bias weights as well as the alpha parameters of LayerNorms.

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #AdamW is defined at optimization.py. learning_rate is equla to 2e-5 and adam_epsilon is equal to 1e-8
    #AdamW implements Adam optimizer with the weight decay fix. This fix ensures that the weight decay is uncoupled from the running statistics of
    #Adam so that weight decay will not be scaled according to the running statistics of Adam. 
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    #get_linear_schedule_with_warmup is implemented at optimization.py
    #for glue bert mrpc, args.warmup_steps is set to zero and t_total is equal to 459 * 3 where 3 is num_train_epochs and 459 is the number of minibatches
    #of size 8 for mrpc dataset. The total number of sequences in mrpc traning dataset is 3672 = 459 * 8

    # ** what returns by get_linear_schedule_with_warmup?
    #it increases the learning rate linearly from 0 to lr over the steps [0, warmup_steps] and then deacrease the learning rate from lr to 0 over the
    #steps [warmup_steps, num_training_steps]

    #args.fp16 is False for bert glue script.
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    # for linux machine, n_gpu will be equal to 2 and for mac, it will be equal to 0
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    #for glue birt mrpc script, local_rank is equal to -1
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    #train_dataset is an instance of TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels) and its length is equal to the
    #number of sequences in train dataset of mrpc which is equal to 3672.
    
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    #num_train_epochs is equal to 3 for glue script
    
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    #per_gpu_train_batch_size is equal to 8
    
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    #the above will be equal to 8 since local_rank is -1 and gradient_accumulation_steps is equal to 1

    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps) #this is equal to 1
    
    logger.info("  Total optimization steps = %d", t_total)
    #t_total is equal to 459 * 3 where 3 is num_train_epochs and 459 is the number of minibatches of size 8 for mrpc dataset

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    #the tensor buffers associated with gradient being computed in backprob are accumulative and are not being cleared each time that the gradients are
    #computed per each minibatch. In particular, if you don't set these tensor buffers to be zero using zero_grad method of nn.Module, the gradients are
    #being accumulated (added) over and over. 
    
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    #trange is being imported from tdqm. trange(10) is equivalent to tqdm(range(10)) which gives you an iteratable with progress bar.
    #the above tqdm shows progress over epochs which is equal to 3 for glue script
    
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        #this internal tqdm progress bar is over the the examples in train dataset. The disable option will disable tqdm wrapper if either local_rank
        #is -1 which means that we don't use DistributedDataParallel and local_rank 0 means that this is the master node (?) for DistributedDataParallel
        
        for step, batch in enumerate(epoch_iterator):
            
            model.train()
            
            # ** what is the functionality of train method of nn.Module?
            #it puts the instance of nn.Module is train mode which only impacts certain field modules of this module like Dropout and BatchNorm.
            #When DropOut module is put in train mode, it starts dropping the nodes with the dropout probabilities. However, I don't think so that
            #train() and eval() methods have impact on LayerNorm.

            #batch is a python list where contains the following tensors: all_input_ids, all_attention_mask, all_token_type_ids, all_labels
            #all_input_ids, all_attention_masks, all_token_type_ids are tensors of size(8, 128) while all_lables is a tensor of size(8)
            
            batch = tuple(t.to(args.device) for t in batch)

            # ** why do we need to move the input tensors to args.device?
            #it is because, we already moved all the parameters and gradient tensors of the model to args.device using model.to(args.device). Therefore,
            #we need also to move the input tensors to the same device using to() method of tensor objects as above. args.device is equal to cpu for mac
            #and cuda:0 for linux machine 
            
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                #for bert mrpc, since model_type is 'bert', 'token_type_ids' will be added to the inputs dict
                
            outputs = model(**inputs)
            #I believe ** before inputs makes the python dict inputs to be passed to the model as named arguments.
            
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            #loss will be a scalar tensor

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
                #the above line is very interesting! in order to garuntee that the gradients are aggregated among different workers while multi-gpu
                #training, it seems that it is required to compute mean of loss among different workers. 
                

            #gradient_accumulation_steps is equal to 1 for glue script
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            #args.fp16 is False for glue script
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                #nice! this is for bert mrpc. finally, we invoke backward method on loss which computes the graients with respect to the parameters of the
                #model
                loss.backward()

            tr_loss += loss.item()
            #since loss is a scalar tensor, the item method returns the value of this tensor as python float object. 


            # ** what is the objective of gradient_accumulation_steps?
            #it is used if we don't want to update the weights of the model every batch but every few batch. In other words, we want to rely on
            #gradient accumulation feature of torch, to accumulare gradients for gradient_accumulation_setps steps before updating the weights. 
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                if args.fp16:#this is False for glue script
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    #max_grad_norm is equal to 1.0 for glue script
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # ** what is the functionality of clip_grad_norm_?
                #it takes the parameters of the model and access their computed gradients (note loss.backward() is already invoked) using p.grad.data
                #where p is a parameter of the model. Then, it computes the norm 2 of the gradients using p.data.grad.norm(2). Finally, all the
                #norm 2 of gradients being added after being squared. Finally, the square root of the sum is computed. If the total norm is less than 1.0,
                #then the gradients are not being impacted. Otherwise, all the gradients are being scaled down so that the total gradient norm is equal to
                #1.0. In summary, we compute the norm 2 of all the gradient tensors after being concatenated. 
                    
                optimizer.step()
                #optimizer.step() updates the parameters of the model using the computed grqadients filled by loss.backward()
                
                scheduler.step()  # Update learning rate schedule
                #updating the learning_rate
                
                model.zero_grad()
                #make sure that grdient tensors are reset to zero.
                
                global_step += 1

                #logging_steps is equal to 50
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics

                    #for glue script since we rely on DataParallel, local_rank is equal to -1
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        #resuls will be a python dict of acc, f1 and f1_and_acc where acc is the fraction of test examples that are correctly predicted.
                        #f1 is the f1 score is f1 score assuming binary classification where label 1 is the target label. f1_and_acc is the average
                        #f1 and acc
                        
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            #tb_writer is summary writer for tensorboard

                            
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    #save_steps is 50
                    
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    #output_dir is results/MRPC
                    
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    #save_pretrained is defined in modeling_utils.py and it is a method of PreTrainedModel class.
                    #this method saves the model in a format that can be used by from_pretrained method of PreTrainedModel class to reload the model.
                    
                    
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    #for mrpc, task_name is mrpc
    
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    #for bert mrpc, eval_outputs_dirs will be results/mrpc
    
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        #for mrpc, eval_task is mrpc and eval_output_dir is results/mrpc
        
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        #in above eval_dataset will be TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels) where
        #all_input_ids, all_attention_mask, all_token_type_ids are 2-dimensioanl tensors of size [num_test_sequence, 128] and
        #all_lables is a one dimensional tensor of size [num_test_sequence]].
        #TensorDataset is a standard torch dataset from torch.utils.data and each example is retrieved by indexing tensors along
        #their first dimensions


        #local_rank for DistributedParallel will be -1 in glue dataset
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        #n_gpu for linux machine is 2 and 0 for mac. Therefore, eval_batch_size will be equal to 16 for linux machine and 8 for mac 
        
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        #if local_rank is -1, it means that we are not relying on DistributedDataParallel. Therefore, we have either no disributed traninig or a single
        #machine multi gpu distributed training. If it is DistibutedDataParallel, for the evaluation dataset, dataloader, its sampler must be aware of the
        #situation and distribute the examples of dataset among different machines. Therefore, we need an instance of DistributedSampler to be passed as
        #sampler argument to DataLoader. However, if we don't rely on DistributedDataParallel, then for train dataset RandomSampler is good enough and
        #for evaluation dataset, SequnentialSampler which is determenistic is perfect. The reason that here for DistributedDataParallel in the case
        #of evaluation dataset, we still use DistributedSampler which is a random sampler and not deterministic, is that there is no a determnistic
        #sampler like SequentialSampler for DistributedDataParallel
        
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        #as you can see eval dataloader is shuffle disabled

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
            #for linux n_gpu is 2 and model is already being passed once in the train method through torch.nn.DataParallel. Therefore, I think it is fine
            #to pass an instacne of nn.Module several time through torch.nn.DataParallel

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        #lenth of eval dataloder will be equal to the number of sequences in test mrpc dataset.
        
        logger.info("  Batch size = %d", args.eval_batch_size)
        #eval_batch_size will be equal to 8 for mac and 16 for linux machine
        
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            #model.eval() will put the DropOut and BatchNorm layers of the model in evaluation mode
            
            batch = tuple(t.to(args.device) for t in batch)
            #args.device is equal to 'cpu' for mac and 'cuda:0' for linux machine. Since the model's parameters and gradients tensors have already being
            #transferred to args.device which is 'cuda:0' for linux machine, we need also to transfer the input tensors to the same device. 

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                
                #tmp_eval_loss will be the cross entropy loss for binary mrpc taks and it should be already a scalar tensor. However, for
                #DistributedDataParallel, it might be a 1d tensor with the dimension being equal to the number of machines. Therefore, in below, before
                #invoking item method, first, we invoke mean method.

                eval_loss += tmp_eval_loss.mean().item()
                
            nb_eval_steps += 1

            #preds for the first batch will be None!
            if preds is None:
                preds = logits.detach().cpu().numpy()
                #the signficance of detach() method is that eventhough logits grad is True, for the graph that is constructed on top of logits.detach(), 
                #gradients will not be tracked.
                
                out_label_ids = inputs['labels'].detach().cpu().numpy()

                #in both above cases, cpu method is required in the case if these two tensors reside on gpus.
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0) #axis 0 here is across rows
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        #nb_eval_steps is the number of evaluation batch steps

        #for bert mrpc, output_mode is classification. 
        if args.output_mode == "classification":
            #preds will be a ndarray of shape(num_text_examples, 2)
            preds = np.argmax(preds, axis=1)
            #axis 1 here is across column
            #after above np.argmax, preds will be 1d array of size(num_test_examples)
            
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        #both preds and out_label_ids are 1d np arrays of size(num_test_examples)
        result = compute_metrics(eval_task, preds, out_label_ids)
        #compute_metrics is glue_compute_metrics which is defined at transformers/data/metrics/__ini__.py
        #compute_metrics returns a python dict of acc, f1 and acc_and_f1 where acc is the fraction of examples that their labels are correctly predicted.
        #f1 is the f1 score of treating this problem as a binary classification problem where the target lablel is 1 and acc_and_f1 is simply
        #average of acc and f1

        results.update(result)
        #both result and results are python dict where the items of result is used to update results dict. 

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        #eval_ouptput_dir is results/MRPC/
        
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    #task is mrpc and tokenizer is instance of BertTokenizer that internally loads the vocab for bert-base-uncased
    
    #for the standard glue script, local_rank is always -1
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    #here, processor will be an object of transformers.data.processors.glue.MrpcProcessor which is defined at transformers/data/processors/glue.py
    #this object is able to read the train and dev dataset of MRPS and restun python lists of train and dev InputExample

    output_mode = output_modes[task]
    #output_mode will be  "classification" for MRPC

    
    # Load data features from cache or dataset file

    #args.data_dir is glue_data/MRPC which is passed as argument to glue.sh
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    #cached_features_file for bert-base-uncased and MRPC will be cached_train_bert-base-uncased_128_mrpc
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        #for the first time that you execute this script, this feature file for mrpc based on the tokenizr of bert-base-uncased doesn't exist
        
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        label_list = processor.get_labels()
        #label_list for MRPC is ['0', '1']
        
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

            
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        #examples is a list of transformers.data.processors.utils.InputExample
        #the first example is the following:
        #{
        #  "guid": "train-1",
        #  "label": "1",
        #  "text_a": "Amrozi accused his brother , whom he called \" the witness \" , of deliberately distorting his evidence .",
        #  "text_b": "Referring to him as only \" the witness \" , Amrozi accused his brother of deliberately distorting his evidence ."
        #}
        
        features = convert_examples_to_features(examples,# a list of InputExample
                                                tokenizer, #an instance of BertTokenizer
                                                label_list=label_list, #it is ['0', '1']
                                                max_length=args.max_seq_length, #this is 128 for MRPS eventhough for BERT is 512
                                                output_mode=output_mode, #output_mode is 'classification'
                                                pad_on_left=bool(args.model_type in ['xlnet']), # pad on the left for xlnet => False for BERT
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                #for bert pad_token is '[PAD]' and convert_tokens_to_ids will return the vocab index for the token '[PAD]'
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,#so, it will be zero
        )

        #features will be a list of InputFeatures that is defined at transformers/transformers/data/processors/utils.py and is a class with
        #the following fields: input_ids, attention_mask, token_type_ids and label, where input_ids is a list vocab indices that is padded with 0 to
        #have the length of 128, attention_mask is also a list of length 128 with the entries 1 for non-pad tokens and entries 0 for pad tokens,
        #token_type_ids will be also a list of length 128 with the format [0 0 0 .... 0 1 1 1 ... 1 0 0 . . . 0] where the first 0 entries are corresponding
        #[cls] text_a [sep] and 1 entries are corresponding to text_b [sep] and the final 0 entries are corresponding to pad tokens. label also could be
        #either 0 or 1. 

        #for glue standard example. local_rank will be -1. This means that we will be saving the generated features list 
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate: #not True for bert mrpc
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    #all_input_ids will be a 2 dimensional tensor of size [3668, 128]

    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    #all_attention_mask will be a 2 dimensional tensor of size [3668, 128]
    
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    #all_token_type_ids will be a 2 dimensional tensor of size [3668, 128]
    
    if output_mode == "classification": #bert mrps is classification
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        #all_labels will be a 1 dimensional tensor of size [3668]
        
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
 
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    #TensorDataset is a standard torch dataset from torch.utils.data which takes a number of tensors as above assuming all of them have same size
    #first dimention (batch dimension) and each example will be retrieved by indexing tensors along the first dimension.
    
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    #all the MODEL_CLASSES are the versions of sequence encoder for sequenceClassification like BertForSequenceClassification
    #note you might have several models with a same model_type for example, for the model type BertForSequenceClassification, you can have several
    #different variance of BERT like the base one with smaller number of layers possibly and the large one with the larger numebr of layers.

    #{'bert': (<class 'transformers.configuration_bert.BertConfig'>, <class 'transformers.modeling_bert.BertForSequenceClassification'>, <class 'transformers.tokenization_bert.BertTokenizer'>),

    #'xlnet': (<class 'transformers.configuration_xlnet.XLNetConfig'>, <class 'transformers.modeling_xlnet.XLNetForSequenceClassification'>, <class 'transformers.tokenization_xlnet.XLNetTokenizer'>),

    #'xlm': (<class 'transformers.configuration_xlm.XLMConfig'>, <class 'transformers.modeling_xlm.XLMForSequenceClassification'>, <class 'transformers.tokenization_xlm.XLMTokenizer'>),

    #'roberta': (<class 'transformers.configuration_roberta.RobertaConfig'>, <class 'transformers.modeling_roberta.RobertaForSequenceClassification'>, <class 'transformers.tokenization_roberta.RobertaTokenizer'>),

    #'distilbert': (<class 'transformers.configuration_distilbert.DistilBertConfig'>, <class 'transformers.modeling_distilbert.DistilBertForSequenceClassification'>, <class 'transformers.tokenization_distilbert.DistilBertTokenizer'>),

    #'albert': (<class 'transformers.configuration_albert.AlbertConfig'>, <class 'transformers.modeling_albert.AlbertForSequenceClassification'>, <class 'transformers.tokenization_albert.AlbertTokenizer'>)}
    
    #in particaulr, the below model_name_or_path defines what variant of the model_type you want to use. For example, if you choose the model_type to be
    #bert, its variant could be one of the followings: 'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased', 'bert-base-multilingual-uncased', 'bert-base-multilingual-cased', 'bert-base-chinese', 'bert-base-german-cased', 'bert-large-uncased-whole-word-masking', 'bert-large-cased-whole-word-masking', 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-base-cased-finetuned-mrpc', 'bert-base-german-dbmdz-cased', 'bert-base-german-dbmdz-uncased'

    #for the other model_types, you have the following variants:

    #( 'xlnet-base-cased', 'xlnet-large-cased', 'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024', 'xlm-mlm-enro-1024', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024', 'xlm-clm-enfr-1024', 'xlm-clm-ende-1024', 'xlm-mlm-17-1280', 'xlm-mlm-100-1280', 'roberta-base', 'roberta-large', 'roberta-large-mnli', 'distilroberta-base', 'roberta-base-openai-detector', 'roberta-large-openai-detector', 'distilbert-base-uncased', 'distilbert-base-uncased-distilled-squad', 'distilbert-base-multilingual-cased')
    
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    #task_name must be one of the 9 taks of GLUE:
    #['cola', 'mnli', 'mnli-mm', 'mrpc', 'sst-2', 'sts-b', 'qqp', 'qnli', 'rte', 'wnli']
    #in particular, processors is glue_processors

    #processors is a dict of string key name of tasks to their processor classes that read train and dev datasets and retutn python lists of train and
    #test InputExamples
    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    #in transformer example, the two above options were not used which means that if you follow the standard steps, you don't need to be worries about
    #these two options
    
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    #cache_dir is not used in the transformer glue example as well
    
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    #max_seq_length is probably one of the most important options of this script. When training encoders as languge models, you know that all the sequence
    #examples have same lenght of the context window since the text corpus is huge. However, for sequence classification taks, each sentence could have
    #different lenght and the question is that how to deal with differenec sequence lenghts in a single minibatch. In tensorflow, the way that the sequnce
    #with different lenghts are addressed by considering the length of each sequence therefore, to start propagating the gradients from the sequnce lenght
    #to the start of sequnce and also use the hidden state corresponding to the sequnce lenght as the basis for classification task.

    #however, here, it seems that they don't care about the lenght of the sequence. For example, if a sequnce is longer than max_sequence_lenght, it will
    #be truncated and if it shorter than max_seq_lenght, it will be padded by a special token

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    #very simple, do you want to train the model (fine-tune for the NLU task?
    
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    #do you want to run the evaluation over the dev dataset (not test dataset)?

    parser.add_argument("--feature_pyramid", action='store_true',
                        help="Whether to use feature pyramid.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    #this is the option that will be useful if you want to run evaluation job over dev dataset at each time that you save the checkpoint of the model being
    #trained
    
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    #for almost all the encoders, you can find two versions: one that is trained on cased sentences which means that sentences have both lower case and
    #upper case letters, and uncased once which means that all the letters are being lower cased before training the model. Therefore, if the model that
    #you are using an encoder that has trained on uncased sentences (all the sentences being lower cased), then you need to specify it here

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    #this is an important feature in the case if you get out of memory runtime error while running training jobs
    
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    #the same batch_size for evaluations jobs
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    #I don't think this is relevant option for transformers since they don't have any recurrent structures and therefore there is no notion of
    #backpropagation of gradients over time. transformer GLUE job doesn't specify this, therefore, the default value of 1 will be used
    
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    #it seems that the only option for optimizer is Adam. The learning rate for GLUE transformer task is set to be 2e-5
    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    #the weight decay is not used for transformer GLUE job which means that weight_decay is equal to 0
    
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    #this is not set for GLUE transformer task as well
    
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    #this is not set as well which means that the max_grad_norm is 1.0
    
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    #this is set to be 3 which means that we only perform 3 epochs of training over NLU tasks
    
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    #this is not set in transformer GLUE task which means that there is no linear warmup. If this parameter is set, it determines the numeber of steps
    #(in terms of number of minibatches) over which the learning_rate will be increaseed from 0 to the nominal value for the learning rate

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    #this means to log every 50 minibatches being processed
    
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    #this means which how many minibatches interval to save the checkpoints for weights
    
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    #this specifies whether we want to evaluate each saved checkpoint or not
    
    
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    #why to not use cuda?
    
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    #do you want to overwrite the content of output directory?
    
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    #do you want to overwrite the cached training and evaluation sets?
    
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    #what is the random seed for initilization?
    #the above default seed will be used for glue script.

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    #whether to use fp16 for mixed precision operation to speed up training?

    
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    

    
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    #this is interesting since it specifies the rank of this worker in distributed training schema
    
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    
    
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    #we usually don't use server debugging

    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #in above, I changed "cuda" to "cuda:0"
        #in above, if we are asked no_cuda, then device will be "cpu", otherwise if gpu is avaiable, it will be "cuda:0"
        #the glue standard script will trigger this option which happens because we don't specifiy local_rank option.
        #in general, if you don't set local_rank, you are asking this script to use DataParallel to do distributed trainig which is a single-machine,
        #multi-gpu distributed training and if you set the local_rank, then you are asking from torch to use DistributedDataParallel which is
        #multi-machine, multi-gpu distributed training. setting "cuda:0" is totally fine and still DataParallel uses both GPUs to run the trainig job
        args.n_gpu = torch.cuda.device_count()
        #args.n_gpu will be 2
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    
    processor = processors[args.task_name]()
    #the above will create an instance of MRPCProcessor which return lists of test and train InputExample's
    
    args.output_mode = output_modes[args.task_name]
    #the output_modes for all the 9 GLUE tasks are "classification" which makes sense since GLUE is a (pair) sentence classification

    
    label_list = processor.get_labels()
    #label_list will be ["0", "1"] for MRPC dataset
    num_labels = len(label_list) #=> 2

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    #barrier method blocks this process till all the processes reach this point. This means that except the first process with rank 0 that gets
    #a free pass, all the other processes need to wait
    
    args.model_type = args.model_type.lower()
    #in this example, this is simply bert
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    #for bert, the model class is the following: 'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),

    #in below, config_class is BertConfig that is defined at transormers/configuration_bert.py. BertConfig is a config class that contains the configuration
    #of a BertModel. args.model_name_or_path is bert-base-uncased.
    #I think the config_class is supposed to represent the architecture of the model so that it the model could be reconstructed. This model config file
    #which only contains the meta data representing the model architecture and not the weights of the network either could be passed to this class in a form
    #of json file, or could be downloaded if you use one of the standard existing models like bert-base-uncased.

    #in the case of bert-base-uncased, the cloud config json file in s3 is the following:
    #{
    #attention_probs_dropout_prob: 0.1,
    #hidden_act: "gelu",
    #hidden_dropout_prob: 0.1,
    #hidden_size: 768,
    #initializer_range: 0.02, => std for initialization of the weights with normal distribution
    #intermediate_size: 3072,
    #max_position_embeddings: 512, => this must be the context window
    #num_attention_heads: 12,
    #num_hidden_layers: 12,
    #type_vocab_size: 2,
    #vocab_size: 30522
    #}

    
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          enable_feature_pyramid = args.feature_pyramid)

    #from_pretrained method will download the config json file if doesn't already exist in cache folder or if force_download is enabled. Then, after either
    #dowloading it from cloud or finding it in cache, it will create a config object using that json config file and return it.
    #the returned config object for bert-base-uncased for mrpc task is the following where we have overriden its fine_tunning attributed by passing it
    #via from_pretrained method to be arg.task_name

    #{
    #"attention_probs_dropout_prob": 0.1,
    #"finetuning_task": "mrpc",
    #"hidden_act": "gelu",
    #"hidden_dropout_prob": 0.1,
    #"hidden_size": 768,
    #"initializer_range": 0.02,
    #"intermediate_size": 3072,
    #"is_decoder": false,
    #"layer_norm_eps": 1e-12,
    #"max_position_embeddings": 512,
    #"num_attention_heads": 12,
    #"num_hidden_layers": 12,
    #"num_labels": 2,
    #"output_attentions": false,
    #"output_hidden_states": false,
    #"output_past": true,
    #"pruned_heads": {},
    #"torchscript": false,
    #"type_vocab_size": 2,
    #"use_bfloat16": false,
    #"vocab_size": 30522
    #}

    #tokenizer_class is BertTokenizer which is defined at tokenization_bert.py 
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    #in glue.sh task, we don't specify args.tokenizer_name, therefore, args.model_name_or_path that is equal to
    #bert-base-uncased will be passed as the first argument to this method. The other arguement for this method is args.do_lower_case
    #which is enabled for glue.sh . The cache_dir option is not provided inside glue.sh which causes None to be passed instead.
    #This results in this method to rely on the default cache folder which is the following:
    #/home/mohammad.sanatkar/.cache/torch/transformers

    #tokenizer will be an object of bert tokenizer that first download the vocab file for bert-base-unacased from s3 amazon
    #cloud and cache it. Then, it will read this file and create a vocab corresponding to byte-pair encoding of bert-base-uncased
    #also, this tokenizer knows about what are the special tokens it needs to add to the begining of the sencteces and the end
    #of sentences as well as how to separate two sentecens in the case of pair sequence classification. Also, it has access
    #internally to word segmenter that has the responsibilty to use a greedy first longest match to segment each word into its
    #forming byte-pair encoding segments. 


    #model_class for bert-base-uncased is BertForSequenceClassification which is implemented at modeling_bert.py
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    #in above, bert-base-uncased is a torch model, therefore, from_tf will be Falae (no tensorflow model)
    #also, cache_dir will be None which results in this class to rely on the default cache directory
    #the config arguments is a config object that describes the architecture of bert-base-uncased like the activation type,
    #number of attention-block layers, number of attention heads, hidden size, fine-tunning task, ...

    # ** what does from_pretrained perform?
    #it is a static method of PreTrainedModel in modeling_utils.py which instantiates an object of the child class (for this example,
    #BertForSequenceClassification) and initilize its weights from a pretrained checkpoint that either was already cached or downloaded from S3.
    #Note that not all the layers of BertForSequenceClassification is initialized from this checkpoint. In particular, the fully-connected single
    #layer head that does the binary classification of the pooled embedding is not initialized. Also, another significant point is that this
    #returned model is in eval mode which means that dropout layers are not active. 

    #in glue standard script, local_rank as one of the arguements is not specified which will be backoff-ed to -1.
    #in particualr, the existing script relies on DataParallel which will be distributed training with single machine and several gpus training. This
    #option happens when local_rank is equal to -1
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    #args.device will be equal to cpu for laptop and cuda:0 for linux machine
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    #args.do_train is enabled in script glue.sh for mrpc based on bert-base-uncased
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        #args.task_name is mrpc, tokenizer is instance of BertTokenizer

        #in above train_dataset will be TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels) where
        #all_input_ids, all_attention_mask, all_token_type_ids are 2-dimensioanl tensors of size [3668, 128] and all_lables is a one dimensional tensor
        #of size [3668]. TensorDataset is a standard torch dataset from torch.utils.data and each example is retrieved by indexing tensors along
        #their first dimensions
        
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        #tr_loss is the average loss over all the training steps and not the final loss. In particualr, tr_loss is the average loss acorss all the
        #minibatched of all epochs.
        #the above train method, will save the model checkpoints in results/MRPC folder. Also, it writes the evaluation results in a txt file called
        #eval_results.txt in the the folder results/MRPC with three different metrics: acc, f1 and acc_f1. Note that this eval_results.txt will be
        #overwritten each time that we run the evaluation job.
        
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        #save_pretrained is defined in modeling_utils.py and it is a method of PreTrainedModel class.
        #this method saves the model in a format that can be used by from_pretrained method of PreTrainedModel class to reload the model.

        #output_dir is results/MRPC
        
        tokenizer.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        #In the below, we reload the model from the saved finetuend model in order to ensure that the evaluation has been performred on the save model
        #to avoid any discrepency.
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)
        #args.device is cuda:0 for linux machine ans cpu for mac

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        #for glue script, do_eval is enabled.
        
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        #args.output_dir is results/MRPC
        
        #for glue script, eval_all_checkpoints is disabled
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            #WEIGHTS_NAME is equal to "pytorch_model.bin"
            
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            #for glue script, since checkpoints is [results/MRPC], its len is equal to 1 and therefore, global_step will be equal to "".
            #Note, torch records the checkpoints in the format of checkpoint-250 where 250 is the global step.
            
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            #for glue script, prefix will be equal to ""
            
            model = model_class.from_pretrained(checkpoint)
            #in above, checkpoint is equal to results/MRPC
            
            model.to(args.device)
            
            result = evaluate(args, model, tokenizer, prefix=prefix)
            #resuls will be a python dict of acc, f1 and f1_and_acc where acc is the fraction of test examples that are correctly predicted.
            #f1 is the f1 score is f1 score assuming binary classification where label 1 is the target label. f1_and_acc is the average
            #f1 and acc

            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
