""" Finetuning the library models for Multi-Label Task with Basic Model (Bert, XLM, XLNet)."""
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import csv

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_fscore_support

from transformers import WEIGHTS_NAME
from transformers import (BertConfig, BertTokenizer, BertForSequenceClassification)
from transformers import (RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, RobertaForMultipleChoice)

from transformers import AdamW, get_linear_schedule_with_warmup

from utils.dataset_utils_mber import (Motive2Emotion_Processor, motive2emotion_convert_examples_to_features)

from utils.log_wrapper import creat_logger

from opt_finetune import set_args, set_seed

logger = None
LOGGER_NAME = "MBER"
processor = None
PROCESSORS = {
    'mber.m2e': Motive2Emotion_Processor,
}

CONVERTER = {
    "mber.m2e": (motive2emotion_convert_examples_to_features, None),
}

# TaskName_NetworkName
MODEL_CLASSES = {
    "mber.m2e_basic_bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "mber.m2e_basic_roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def accuracy(out, labels, return_prob=True):
    outputs = np.argmax(out, axis=1)
    # if return_prob:
    pred = torch.from_numpy(outputs)
    prob = torch.nn.functional.softmax(torch.from_numpy(out), dim=-1)
    # return np.sum(outputs == labels)
    return np.sum(outputs == labels), pred, prob

def accuracy_f1(out, labels, return_prob=True):
    outputs = np.argmax(out, axis=1)
    # if return_prob:
    p, r, f1, _ = precision_recall_fscore_support(labels, outputs, 1, pos_label=1, average='weighted')
    pred = torch.from_numpy(outputs)
    prob = torch.nn.functional.softmax(torch.from_numpy(out), dim=-1)
    # return np.sum(outputs == labels)
    return np.sum(outputs == labels), pred, prob, p, r, f1


def load_and_cache_examples(args, task, tokenizer, do_mode="dev", output_mode="classification"):
    """Load data and construct dataset process"""
    if args.local_rank not in [-1,0] and do_mode=='train':
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    processor = PROCESSORS[args.task_name.split('_')[0]]()

    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        do_mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task),
        str(args.cls_segment_id)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if do_mode == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        elif do_mode == 'test':
            examples = processor.get_test_examples(args.data_dir)
        elif do_mode == 'train':
            examples = processor.get_train_examples(args.data_dir)
        else:
            raise KeyError(do_mode)

        features = CONVERTER[args.task_name.split("_")[0]][0](
            examples=examples,
            label_list=label_list,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            pad_on_left=bool(args.model_type in ['xlnet']),
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and do_mode=='train':
        torch.distributed.barrier()

    field_selector = CONVERTER[args.task_name.split("_")[0]][1] ### TODO: remove, None
    ### TODO: if change the task, may need to modify the TensorDataset contain
    all_input_ids   = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask  = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    if do_mode != 'test' or args.have_test_label:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    else:
        all_label_ids = torch.tensor([-1 for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset, features


def get_batch_inputs(args, batch, have_test_label=True):
    inputs = {
        'input_ids':      batch[0],
        'attention_mask': batch[1],
        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
        'labels':         batch[3] if have_test_label else None,
    }
    return inputs


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.tfboard_log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) \
        if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else: # use this
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)

    ### override warmup_steps
    if args.max_steps > 0 and args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(t_total * (args.warmup_proportion / 100))

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info(f"Logging file: [{args.log_file}]")
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() \
                       if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  WarmUp steps = %d", warmup_steps)
    logger.info("  Learning Rate = {}".format(args.learning_rate))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch_id in train_iterator:
        epoch_tr_loss = 0
        epoch_nb_steps = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = get_batch_inputs(args, batch, have_test_label=True)
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if args.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            epoch_tr_loss += loss.item()
            epoch_nb_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', loss.item(), global_step)

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, prefix=global_step)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        logger.info('Epoch: {}, Epoch Loss: {}'.format(epoch_id, epoch_tr_loss / epoch_nb_steps))

        if args.do_eval and args.evaluate_epoch:
            ### evaluate after each epoch
            results = evaluate(args, model, tokenizer,
                               prefix=global_step,
                               train_loss=(epoch_tr_loss / epoch_nb_steps))
            for key, value in results.items():
                tb_writer.add_scalar('epoch_eval_{}'.format(key), value, epoch_id)
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=None, have_test_label=True, train_loss=-1.0):
    eval_task = args.task_name
    eval_output_dir = args.output_dir
    eval_dataset, eval_features = load_and_cache_examples(args, eval_task, tokenizer, do_mode='dev')

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    ### Eval!
    logger.info(f"Logging file: [{args.log_file}]")
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = get_batch_inputs(args, batch, have_test_label=have_test_label)
            outputs = model(**inputs)

            if have_test_label:
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            else:
                logits = outputs[0]

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            if have_test_label:
                out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            if have_test_label:
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    result = {
        'global_step': prefix,
        'train_loss': train_loss
    }

    answer = os.path.join(eval_output_dir, "subtaskA_answers.csv")

    output_eval_file = os.path.join(eval_output_dir, "{}-{}".format(args.result_eval_file, prefix))
    logger.info("Eval resulat write to: {}".format(output_eval_file))
    if have_test_label:
        eval_accuracy, eval_pred, eval_prob = accuracy(preds, out_label_ids)

        pred_out = eval_pred.tolist()
        pred_prob = eval_prob.tolist()
        ground_truth = out_label_ids.tolist()

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / len(ground_truth)

        result['eval_loss'] = eval_loss
        result['eval_accuracy'] = eval_accuracy

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            ### record prediction results
            writer.write("format: id, ground_truth label, system prediction\n")
            for i, (gt, pred, prob) in enumerate(zip(ground_truth, pred_out, pred_prob)):
                qid = eval_features[i].example_id
                result_out = "\t".join(["{:.4f}".format(pi) for pi in prob])
                write_out = "{}\t{}\t{}\t{}\t\n".format(qid, gt, pred, result_out)
                writer.write(write_out)

        # output submission csv format
        with open (answer, "w") as writer:
            ans_writer = csv.writer(writer, delimiter = ',')
            for i in range(len(pred_out)):
                ans_writer.writerow([eval_features[i].example_id.split("-")[-1], pred_out[i]])

    else:
        pred_out = torch.from_numpy(np.argmax(preds, axis=1)).tolist()
        pred_prob = torch.nn.functional.softmax(torch.from_numpy(preds), dim=-1).tolist()
        with open(output_eval_file, "w") as writer:
            writer.write("format: id, system prediction prob.\n")
            for i, (pred, prob) in enumerate(pred_out, pred_prob):
                qid = eval_features[i].example_id
                result_out = "\t".join(["{:.4f}".format(pi) for pi in prob])
                write_out = "{}\t{}\t{}\n".format(qid, pred, result_out)
                writer.write(write_out)

        # output submission csv format
        with open (answer, "w") as writer:
            ans_writer = csv.writer(writer, delimiter = ',')
            for i in range(len(pred_out)):
                ans_writer.writerow([eval_features[i].example_id.split("-")[-1], pred_out[i]])

    return result


def evaluate_f1(args, model, tokenizer, prefix=None, have_test_label=True, train_loss=-1.0,do_mode='dev'):
    eval_task = args.task_name
    eval_output_dir = args.output_dir
    eval_dataset, eval_features = load_and_cache_examples(args, eval_task, tokenizer, do_mode)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    ### Eval!
    logger.info(f"Logging file: [{args.log_file}]")
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = get_batch_inputs(args, batch, have_test_label=have_test_label)
            outputs = model(**inputs)

            if have_test_label:
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            else:
                logits = outputs[0]

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            if have_test_label:
                out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            if have_test_label:
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    result = {
        'global_step': prefix,
        'train_loss': train_loss
    }

    answer = os.path.join(eval_output_dir, "subtaskA_answers.csv")

    output_eval_file = os.path.join(eval_output_dir, "{}-{}".format(args.result_eval_file, prefix))
    logger.info("Eval resulat write to: {}".format(output_eval_file))
    if have_test_label:
        eval_accuracy, eval_pred, eval_prob, p, r, f1 = accuracy_f1(preds, out_label_ids)

        pred_out = eval_pred.tolist()
        pred_prob = eval_prob.tolist()
        ground_truth = out_label_ids.tolist()

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / len(ground_truth)

        result['eval_loss'] = eval_loss
        result['eval_accuracy'] = eval_accuracy
        result['p'] = p
        result['r'] = r
        result['f1'] = f1

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            ### record prediction results
            writer.write("format: id, ground_truth label, system prediction\n")
            for i, (gt, pred, prob) in enumerate(zip(ground_truth, pred_out, pred_prob)):
                qid = eval_features[i].example_id
                result_out = "\t".join(["{:.4f}".format(pi) for pi in prob])
                write_out = "{}\t{}\t{}\t{}\t\n".format(qid, gt, pred, result_out)
                writer.write(write_out)

        # output submission csv format
        with open (answer, "w") as writer:
            ans_writer = csv.writer(writer, delimiter = ',')
            for i in range(len(pred_out)):
                ans_writer.writerow([eval_features[i].example_id.split("-")[-1], pred_out[i]])

    else:
        pred_out = torch.from_numpy(np.argmax(preds, axis=1)).tolist()
        pred_prob = torch.nn.functional.softmax(torch.from_numpy(preds), dim=-1).tolist()
        with open(output_eval_file, "w") as writer:
            writer.write("format: id, system prediction prob.\n")
            for i, (pred, prob) in enumerate(pred_out, pred_prob):
                qid = eval_features[i].example_id
                result_out = "\t".join(["{:.4f}".format(pi) for pi in prob])
                write_out = "{}\t{}\t{}\n".format(qid, pred, result_out)
                writer.write(write_out)

        # output submission csv format
        with open (answer, "w") as writer:
            ans_writer = csv.writer(writer, delimiter = ',')
            for i in range(len(pred_out)):
                ans_writer.writerow([eval_features[i].example_id.split("-")[-1], pred_out[i]])

    return result

def main():
    parser = argparse.ArgumentParser()
    args = set_args(parser)

    global logger
    logger = creat_logger(LOGGER_NAME, to_disk=True, log_file=args.log_file)

    if os.path.exists(args.output_dir) \
            and os.listdir(args.output_dir) \
            and args.do_train \
            and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.task_name = args.task_name.lower()
    args.network_name = args.network_name.lower()
    args.model_type = args.model_type.lower()

    ### prepare task data
    processor = PROCESSORS[args.task_name.split('_')[0]]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    ### Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[f"{args.task_name}_{args.network_name}_{args.model_type}"]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=False,
                                        config=config)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    # Distributed and parallel training
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    ### Training
    if args.do_train:
        train_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer,
                                                   do_mode='train',
                                                   output_mode=args.output_mode)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging ??
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate_f1(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    # Testing
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list (os.path.dirname (c) for c in
                                sorted (glob.glob (args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger ("pytorch_transformers.modeling_utils").setLevel (logging.WARN)  # Reduce logging ??
        logger.info ("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split ('-')[-1] if len (checkpoints) > 1 else ""
            model = model_class.from_pretrained (checkpoint)
            model.to (args.device)
            result = evaluate_f1 (args, model, tokenizer, prefix=global_step,
                               have_test_label=args.have_test_label,
                               do_mode='test')
            result = dict (('{}_'.format ('test') + k + '_{}'.format (global_step), v) for k, v in result.items ())
            results.update (result)

    results_str = "\n".join(["{}:\t{}".format(k,v) for k,v in results.items()])
    logger.info("Final Results:\n{}".format(results_str))

    return results

if __name__ == '__main__':
    main()
