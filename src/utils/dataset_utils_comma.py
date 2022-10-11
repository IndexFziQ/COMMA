import os
import logging
import json
from .dataset_utils_base import DataProcessor
from .dataset_utils_base import InputExample_MBER, InputFeatures_MBER

logger = logging.getLogger("MBER")

class Motive2Emotion_Processor(DataProcessor):
    "Processor for the SemEval Task 4 subtask A datset."

    def get_train_examples(self, data_dir, file_name=None):
        input_file = os.path.join (data_dir, "train.tsv")
        return self._create_examples (
            records=self._read_tsv (input_file),
            set_type='train'
        )

    def get_dev_examples(self, data_dir, file_name=None):
        input_file = os.path.join (data_dir, "dev.tsv")
        return self._create_examples (
            records=self._read_tsv (input_file),
            set_type='dev'
        )

    def get_test_examples(self, data_dir, file_name=None):
        input_file = os.path.join (data_dir, "test.tsv")
        return self._create_examples (
            records=self._read_tsv (input_file),
            set_type='test'
        )

    def get_labels(self):
        return ["1", "2", "3", "4", "5", "6", "7", "8"]

    def _create_examples(self, records, set_type='train'):
        """Creates examples for the training and trial sets."""
        examples = []
        for (i, line) in enumerate(records):
            record = line

            storyid_linenum = record['storyid_linenum']
            human = record['char']
            context = record['history']
            event = record['behavior']
            desc = json.loads(record['srl_behavior'])
            behavior = desc['description']
            factors = record['factors']
            label_motive = record['motive']
            emotion_expect = record['emotion_expect']
            label_emotion = record['emotion_result']


            examples.append (
                InputExample_MBER (
                    storyid_linenum=storyid_linenum,
                    human=human,
                    context=context,
                    event=event,
                    behavior=behavior,
                    factors=factors,
                    label_motive=label_motive,
                    emotion_expect=emotion_expect,
                    label_emotion=label_emotion,
                )
            )
        return examples

class Emotion2Motive_Processor(DataProcessor):
    "Processor for the SemEval Task 4 subtask A datset."

    def get_train_examples(self, data_dir, file_name=None):
        input_file = os.path.join(data_dir, "train.tsv")
        return self._create_examples(
            records=self._read_tsv(input_file),
            set_type='train'
        )

    def get_dev_examples(self, data_dir, file_name=None):
        input_file = os.path.join(data_dir, "dev.tsv")
        return self._create_examples(
            records=self._read_tsv(input_file),
            set_type='dev'
        )

    def get_test_examples(self, data_dir, file_name=None):
        input_file = os.path.join(data_dir, "test.tsv")
        return self._create_examples(
            records=self._read_tsv(input_file),
            set_type='test'
        )

    def get_labels(self):
        return ["1", "2", "3", "4", "5"]

    def _create_examples(self, records, set_type='train'):
        """Creates examples for the training and trial sets."""
        examples = []
        for (i, line) in enumerate(records):
            record = line

            storyid_linenum = record['storyid_linenum']
            human = record['char']
            context = record['history']
            event = record['behavior']
            desc = json.loads (record['srl_behavior'])
            behavior = desc['description']
            factors = record['factors']
            label_motive = record['motive']
            emotion_expect = record['emotion_expect']
            label_emotion = record['emotion_result']

            examples.append (
                InputExample_MBER (
                    storyid_linenum=storyid_linenum,
                    human=human,
                    context=context,
                    event=event,
                    behavior=behavior,
                    factors=factors,
                    label_motive=label_motive,
                    emotion_expect=emotion_expect,
                    label_emotion=label_emotion,
                )
            )
        return examples

class Behave2Motive_Processor(DataProcessor):
    "Processor for the SemEval Task 4 subtask A datset."

    def get_train_examples(self, data_dir, file_name=None):
        input_file = os.path.join (data_dir, "train.tsv")
        return self._create_examples (
            records=self._read_tsv (input_file),
            set_type='train'
        )

    def get_dev_examples(self, data_dir, file_name=None):
        input_file = os.path.join (data_dir, "dev.tsv")
        return self._create_examples (
            records=self._read_tsv (input_file),
            set_type='dev'
        )

    def get_test_examples(self, data_dir, file_name=None):
        input_file = os.path.join (data_dir, "test.tsv")
        return self._create_examples (
            records=self._read_tsv (input_file),
            set_type='test'
        )

    def get_labels(self):
        return ["1", "2", "3", "4", "5"]

    def _create_examples(self, records, set_type='train'):
        """Creates examples for the training and trial sets."""
        examples = []
        for (i, line) in enumerate(records):
            record = line

            storyid_linenum = record['storyid_linenum']
            human = record['char']
            context = record['history']
            event = record['behavior']
            desc = json.loads (record['srl_behavior'])
            behavior = desc['description']
            factors = record['factors']
            label_motive = record['motive']
            emotion_expect = record['emotion_expect']
            label_emotion = record['emotion_result']

            examples.append (
                InputExample_MBER (
                    storyid_linenum=storyid_linenum,
                    human=human,
                    context=context,
                    event=event,
                    behavior=behavior,
                    factors=factors,
                    label_motive=label_motive,
                    emotion_expect=emotion_expect,
                    label_emotion=label_emotion,
                )
            )
        return examples

class Motive2Emotion_Processor(DataProcessor):
    "Processor for the SemEval Task 4 subtask A datset."

    def get_train_examples(self, data_dir, file_name=None):
        input_file = os.path.join (data_dir, "train.tsv")
        return self._create_examples (
            records=self._read_tsv (input_file),
            set_type='train'
        )

    def get_dev_examples(self, data_dir, file_name=None):
        input_file = os.path.join (data_dir, "dev.tsv")
        return self._create_examples (
            records=self._read_tsv (input_file),
            set_type='dev'
        )

    def get_test_examples(self, data_dir, file_name=None):
        input_file = os.path.join (data_dir, "test.tsv")
        return self._create_examples (
            records=self._read_tsv (input_file),
            set_type='test'
        )

    def get_labels(self):
        return ["1", "2", "3", "4", "5", "6", "7", "8"]

    def _create_examples(self, records, set_type='train'):
        """Creates examples for the training and trial sets."""
        examples = []
        for (i, line) in enumerate(records):
            record = line

            storyid_linenum = record['storyid_linenum']
            human = record['char']
            context = record['history']
            event = record['behavior']
            desc = json.loads(record['srl_behavior'])
            behavior = desc['description']
            factors = record['factors']
            label_motive = record['motive']
            emotion_expect = record['emotion_expect']
            label_emotion = record['emotion_result']


            examples.append (
                InputExample_MBER (
                    storyid_linenum=storyid_linenum,
                    human=human,
                    context=context,
                    event=event,
                    behavior=behavior,
                    factors=factors,
                    label_motive=label_motive,
                    emotion_expect=emotion_expect,
                    label_emotion=label_emotion,
                )
            )
        return examples

def motive2emotion_convert_examples_to_features(examples, label_list, tokenizer,
                                          max_length=512,
                                          pad_on_left=False,
                                          pad_token=0,
                                          pad_token_segment_id=0,
                                          mask_padding_with_zero=True,
                                          is_training=True):
    """convert_examples_to_features function for huggingface-transformers-v2.2"""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for example_index, example in enumerate(examples):
        char_name = example.human
        human_needs = ['physiological', 'stability', 'love', 'esteem', 'spiritual growth']
        appendix = human_needs[int(example.label_motive) - 1]
        inputs = tokenizer.encode_plus(
            # w/ motive
            # text='<need> '+"%s has %s need." % (char_name, appendix)+ '</need> ',
            # text_pair='<act> ' + example.behavior + '</act> ',
            # w/o motive
            text=example.behavior,
            text_pair=appendix,
            add_special_tokens=True,  # for [CLS] and [SEP]
            max_length=max_length,
        )
        if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
            logger.info('Attention! you are cropping tokens (swag task is ok). '
                        'If you are training ARC and RACE and you are poping question + options,'
                        'you need to try to use a bigger max seq length!')

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

        label_id = label_map[example.label_emotion] if example.label_emotion is not None else -1

        if example_index < 1:
            logger.info("*** Example ***")
            logger.info(f"example_id: {example.storyid_linenum}")
            logger.info(f"tokens: {tokens}")
            logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
            logger.info(f"input_mask: {' '.join(map(str, attention_mask))}")
            logger.info(f"segment_ids: {' '.join(map(str, token_type_ids))}")
            logger.info(f"label: {label_id}")

        if example_index % 2000 == 0:
            logger.info(f"convert: {example_index}")

        features.append(
            InputFeatures_MBER(
                example_id=example.storyid_linenum,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id
            )
        )

    return features

def emotion2motive_convert_examples_to_features(examples, label_list, tokenizer,
                                          max_length=512,
                                          pad_on_left=False,
                                          pad_token=0,
                                          pad_token_segment_id=0,
                                          mask_padding_with_zero=True,
                                          is_training=True):
    """convert_examples_to_features function for huggingface-transformers-v2.2"""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for example_index, example in enumerate(examples):
        char_name = example.human
        # human_needs = ['physiological', 'stability', 'love', 'esteem', 'spiritual growth']
        # appendix = human_needs[int(example.label_motive) - 1]
        # emotion_p = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]
        emotions_set = 'default'
        if example.label_emotion == '1':
            emotions_set = 'joy'
        elif example.label_emotion == '2':
            emotions_set = 'trust'
        elif example.label_emotion == '3':
            emotions_set = 'fear'
        elif example.label_emotion == '4':
            emotions_set = 'surprise'
        elif example.label_emotion == '5':
            emotions_set = 'sadness'
        elif example.label_emotion == '6':
            emotions_set = 'disgust'
        elif example.label_emotion == '7':
            emotions_set = 'anger'
        elif example.label_emotion == '8':
            emotions_set = 'anticipation'

        inputs = tokenizer.encode_plus(
            # w/ motive
            # text='<act> ' + example.behavior + '</act> ',
            # text_pair='<e_reaction> ' + "%s's emotional reaction is %s ." % (char_name, emotions_set)+ '</e_reaction> ',
            # w/o motive
            text= example.event,
            text_pair=emotions_set,
            add_special_tokens=True,  # for [CLS] and [SEP]
            max_length=max_length,
        )
        if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
            logger.info('Attention! you are cropping tokens (swag task is ok). '
                        'If you are training ARC and RACE and you are poping question + options,'
                        'you need to try to use a bigger max seq length!')

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

        label_id = label_map[example.label_motive] if example.label_motive is not None else -1

        if example_index < 1:
            logger.info("*** Example ***")
            logger.info(f"example_id: {example.storyid_linenum}")
            logger.info(f"tokens: {tokens}")
            logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
            logger.info(f"input_mask: {' '.join(map(str, attention_mask))}")
            logger.info(f"segment_ids: {' '.join(map(str, token_type_ids))}")
            logger.info(f"label: {label_id}")

        if example_index % 2000 == 0:
            logger.info(f"convert: {example_index}")

        features.append(
            InputFeatures_MBER(
                example_id=example.storyid_linenum,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id
            )
        )

    return features

def behave2motive_convert_examples_to_features(examples, label_list, tokenizer,
                                          max_length=512,
                                          pad_on_left=False,
                                          pad_token=0,
                                          pad_token_segment_id=0,
                                          mask_padding_with_zero=True,
                                          is_training=True):
    """convert_examples_to_features function for huggingface-transformers-v2.2"""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for example_index, example in enumerate(examples):
        char_name = example.human
        inputs = tokenizer.encode_plus(
            # w/ motive
            # text='<e_expect> ' + "%s's emotional expectation is %s ." % (char_name, example.emotion_expect) + '</e_expect> ',
            # text_pair = '<act> ' + example.event + '</act> ',
            text = example.emotion_expect,
            text_pair=example.behavior,
            # w/o motive
            # text=example.context + example.event,
            # text_pair="",
            add_special_tokens=True,  # for [CLS] and [SEP]
            max_length=max_length,
        )
        if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
            logger.info('Attention! you are cropping tokens (swag task is ok). '
                        'If you are training ARC and RACE and you are poping question + options,'
                        'you need to try to use a bigger max seq length!')

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

        label_id = label_map[example.label_motive] if example.label_motive is not None else -1

        if example_index < 1:
            logger.info("*** Example ***")
            logger.info(f"example_id: {example.storyid_linenum}")
            logger.info(f"tokens: {tokens}")
            logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
            logger.info(f"input_mask: {' '.join(map(str, attention_mask))}")
            logger.info(f"segment_ids: {' '.join(map(str, token_type_ids))}")
            logger.info(f"label: {label_id}")

        if example_index % 2000 == 0:
            logger.info(f"convert: {example_index}")

        features.append(
            InputFeatures_MBER(
                example_id=example.storyid_linenum,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id
            )
        )

    return features
