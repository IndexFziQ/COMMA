import os
import sys
import argparse
import logging
import json
# import yaml
import numpy as np
import random
import torch

logger = logging.getLogger("MBER")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def bool_flag(v):
    if v.lower() in {"on", "true", "yes", "t", "y", "1"}:
        return True
    elif v.lower() in {"off", "false", "no", "f", "n", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def set_args(parser, additional=False, **kwargs):
    path_config(parser)
    run_config(parser)
    ### add by func
    add_func = kwargs.get("add_func", None)
    if add_func is not None:
        for each in add_func:
            logger.info(f'Args add: [{each}]')
            eval(each)(parser)
    args = parser.parse_args()

    return args


def path_config(parser):
    path_group = parser.add_argument_group("Path information and required dirs")
    path_group.add_argument("--data_dir",
                            default='./data',
                            type=str,
                            required=False,
                            help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    path_group.add_argument("--train_file", default='./data/train.tsv', type=str)
    path_group.add_argument("--dev_file",   default=None, type=str)
    path_group.add_argument("--test_file",  default=None, type=str)
    path_group.add_argument("--use_newd", default=False, type=bool_flag)
    path_group.add_argument("--split_dev", default=False, type=bool_flag)
    path_group.add_argument("--output_dir",
                            default='./output',
                            type=str,
                            required=False,
                            help="The output directory where the model checkpoints will be written.")
    path_group.add_argument("--log_file", default="log.out", type=str)
    path_group.add_argument("--tfboard_log_dir", default="event.out", type=str)
    path_group.add_argument("--result_eval_file", default="result.eval.txt", type=str)
    path_group.add_argument("--result_test_file", default="result.test.txt", type=str)
    path_group.add_argument("--result_trial_file", default="result.trial.txt", type=str)

    path_group.add_argument("--model_type", default="gpt2", type=str)
    path_group.add_argument("--model_name_or_path",
                            default='./model/pytorch-gpt2',
                            type=str,
                            required=False,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                                "bert-base-multilingual-cased, bert-base-chinese.")
    path_group.add_argument("--tokenizer_name_or_path",
                            default='./model/pytorch-gpt2',
                            type=str,
                            required=False,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                                "bert-base-multilingual-cased, bert-base-chinese.")
    path_group.add_argument("--config_name",
                            default=None,
                            type=str,
                            help="Pretrained config name or path if not the same as model_name")
    path_group.add_argument('--overwrite_output_dir', action='store_true',
                            help="Overwrite the content of the output directory")
    path_group.add_argument('--overwrite_cache', action='store_true',
                            help="Overwrite the cached input feature sets.")


def run_config(parser):
    run_group = parser.add_argument_group("Run configs")
    run_group.add_argument("--task_name", default='gpt2',
                           type=str,
                           required=False)
    run_group.add_argument("--network_name", default=None,
                           type=str)
    ### Run parameters
    run_group.add_argument("--max_seq_length", default=128,
                           type=int,
                           help="The maximum total input sequence length after WordPiece tokenization. \n"
                                "Sequences longer than this will be truncated, and sequences shorter \n"
                                "than this will be padded.")
    run_group.add_argument("--do_lower_case",
                           action='store_true',
                           help="Set this flag if you are using an uncased model.")
    run_group.add_argument("--cls_segment_id", default=0,
                           type=int)

    ### Run Mode
    run_group.add_argument("--do_train",
                           action='store_true',
                           help="Whether to run training.")
    run_group.add_argument("--do_eval",
                           action='store_true',
                           help="Whether to run eval on the dev set.")
    run_group.add_argument("--do_test",
                           action='store_true',
                           help="Whether to run test on the test set.")
    run_group.add_argument("--do_trial",
                           action='store_true',
                           help="Whether to run test on the unofficial dev set.")
    run_group.add_argument("--have_test_label",
                           action='store_true',
                           help="Used when testing")

    ### Train parameters
    run_group.add_argument("--train_batch_size",
                           default=8,
                           type=int,
                           help="Total batch size for training.\n"
                                "Discarded.")
    run_group.add_argument("--per_gpu_train_batch_size",
                           default=8,
                           type=int,
                           help="Batch size per GPU/CPU for training.")
    run_group.add_argument("--eval_batch_size",
                           default=8,
                           type=int,
                           help="Total batch size for eval. \n"
                                "Discarded.")
    run_group.add_argument("--per_gpu_eval_batch_size",
                           default=8,
                           type=int,
                           help="Batch size per GPU/CPU for evaluation.")
    run_group.add_argument("--learning_rate",
                           default=1e-5,
                           type=float,
                           help="The initial learning rate for Adam.")
    run_group.add_argument("--adam_epsilon",
                           # default=1e-8,
                           default=1e-6,
                           type=float,
                           help="Epsilon for Adam optimizer.")
    run_group.add_argument("--num_train_epochs",
                           default=1.0,
                           type=float,
                           help="Total number of training epochs to perform.")
    run_group.add_argument("--max_steps",
                           default=-1,
                           type=int,
                           help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    run_group.add_argument("--warmup_proportion",
                           default=0.1,
                           type=float,
                           help="Proportion of training to perform linear learning rate warmup for. "
                                "E.g., 0.1 = 10%% of training.")
    run_group.add_argument("--warmup_steps",
                           default=0, type=int,
                           help="Linear warmup over warmup_steps.")
    run_group.add_argument("--weight_decay",
                           # default=0.0,
                           default=0.01,
                           type=float,
                           help="Weight deay if we apply some.")
    run_group.add_argument('--gradient_accumulation_steps',
                           type=int,
                           default=1,
                           help="Number of updates steps to accumulate before performing a backward/update pass.")
    run_group.add_argument("--max_grad_norm",
                           default=1.0, # default is 1.0
                           type=float,
                           help="Max gradient norm.")
    run_group.add_argument('--loss_scale',
                           type=float, default=0,
                           help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                                "0 (default value): dynamic loss scaling.\n"
                                "Positive power of 2: static loss scaling value.\n")
    run_group.add_argument('--lm_coef', type=float, default=0.9,
                           help="parameter to balance lm loss and task loss for GPT/GPT2")
    run_group.add_argument('--add_loss_coef', type=float, default=1.0,
                           help="parameter to balance main loss and additional loss for Task")

    ### Environment
    run_group.add_argument("--no_cuda",
                           action='store_true',
                           help="Whether not to use CUDA when available")
    run_group.add_argument("--local_rank",
                           default=-1,
                           type=int,
                           help="local_rank for distributed training on gpus")
    run_group.add_argument('--seed',
                           default=42,
                           type=int,
                           help="random seed for initialization")
    run_group.add_argument('--fp16',
                           action='store_true',
                           help="Whether to use 16-bit float precision instead of 32-bit")
    run_group.add_argument('--fp16_opt_level',
                           type=str, default='O1',
                           help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")
    ### Others
    run_group.add_argument('--logging_steps', type=int, default=100,
                           help="Log every X updates steps.")
    run_group.add_argument('--save_steps', type=int, default=0,
                           help="Save checkpoint every X updates steps.")
    run_group.add_argument("--eval_all_checkpoints", action='store_true',
                           help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    run_group.add_argument("--evaluate_during_training", action='store_true',
                           help="Rul evaluation during training at each logging step.")
    run_group.add_argument("--evaluate_epoch", action='store_true',
                           help="Rul evaluation during training at each logging step.")

    ### Task specific
    run_group.add_argument("--output_mode",
                           default="classification",
                           type=str)
    run_group.add_argument("--num_choices", default=2,
                           type=int)
    run_group.add_argument("--have_passage",
                           action='store_true',
                           help="if example have context passage")


def add_args(now_args, additional_args):
    now_args, additional_args = vars(now_args), vars(additional_args)
    for k,v in additional_args.items():
        if k not in now_args:
            now_args[k] = v
            logger.info("Update additional config {}: {}".format(k,v))
        else:
            if v != now_args[k]:
                logger.info("Warn: additional config {}: {}/{} exist.".format(k, now_args[k], v))
    return argparse.Namespace(**now_args)


def check_args_version(load_args, now_args):
    load_args, now_args = vars(load_args), vars(now_args)
    for k, v in now_args.items():
        if k not in load_args:
            load_args[k] = v
            logger.info("Update load checkpoint config {}: {}".format(k,v))
    return argparse.Namespace(**load_args)


def override_args(old_args, new_args):
    KEEP_CONFIG = {}
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in KEEP_CONFIG:
                logger.info("Overriding saved {}: {} --> {}".format(k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info("Keeping saved {}: {}".format(k, old_args[k]))
    return  argparse.Namespace(**old_args)


def task_lm_finetune(parser):
    lm_task_group = parser.add_argument_group('Task configs: lm fine-tune')
    ### for lm finetuning
    lm_task_group.add_argument("--mlm",
                               action='store_true',
                               help="Train with masked-language modeling loss instead of language modeling.")
    lm_task_group.add_argument("--mlm_probability",
                               type=float,
                               default=0.15,
                               help="Ratio of tokens to mask for masked language modeling loss")
    lm_task_group.add_argument("--block_size",
                               default=-1,
                               type=int,
                               help="Optional input sequence length after tokenization."
                                    "The training dataset will be truncated in block of this size for training."
                                    "Default to the model max input length for single sentence inputs "
                                    "(take into account special tokens).")
    lm_task_group.add_argument('--save_total_limit',
                               type=int,
                               default=None,
                               help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, '
                                    'does not delete by default')
    lm_task_group.add_argument("--cache_dir", default="", type=str,
                               help="Optional directory to store the pre-trained models downloaded from s3 "
                                    "(instread of the default one)")