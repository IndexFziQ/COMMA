import csv
import copy
import json


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputExample_EA(object):

    """
    A training/test example for human needs prediction.
    storyid_linenum,context,sentence,char,plutchik,label
    Args:
        story_id:       Unique story id for the example.
        human：         character of the story
        linenum:        the number of the event in a story
        context:        the pro-sentences to the current event
        event:          current event
        maslow:         -- 5 labels
        reiss:          -- 19 labels
        plutchik:       -- 8 labels
    """
    def __init__(self, storyid_linenum, human,
                 context=None, event=None,
                 maslow=None, reiss=None, plutchik=None,
                 label_motive=None, label_emotion=None, next_motive=None):
        self.storyid_linenum = storyid_linenum
        self.human = human
        self.context = context
        self.event = event
        self.maslow = maslow
        self.reiss = reiss
        self.plutchik = plutchik
        self.label_motive = label_motive
        self.label_emotion = label_emotion
        self.next_motive = next_motive

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputExample_MBER(object):

    """
    A training/test example for human needs prediction.
    storyid_linenum,context,sentence,char,plutchik,label
    Args:
        story_id:       Unique story id for the example.
        human：         character of the story
        linenum:        the number of the event in a story
        context:        the pro-sentences to the current event
        event:          current event
        behavior:       SRL annotation
        maslow:         -- 5 labels
        reiss:          -- 19 labels
        plutchik:       -- 8 labels
    """
    def __init__(self, storyid_linenum, human,
                 context=None, event=None, behavior=None,
                 maslow=None, reiss=None, plutchik=None, factors=None,
                 label_motive=None, emotion_expect=None, label_emotion=None, next_motive=None):
        self.storyid_linenum = storyid_linenum
        self.human = human
        self.context = context
        self.event = event
        self.behavior = behavior
        self.maslow = maslow
        self.reiss = reiss
        self.plutchik = plutchik
        self.factors = factors
        self.label_motive = label_motive
        self.emotion_expect = emotion_expect
        self.label_emotion = label_emotion
        self.next_motive = next_motive

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, example_id, input_ids, attention_mask, token_type_ids, label_id):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id


class InputExample_MCinQA(object):
    """A single training/dev/test example for Multiple Choice RC without paragraph (only question and candidate) task"""

    def __init__(self, guid, question, choices=[], label=None):
        self.guid = guid
        self.question = question
        self.choices = choices  # list in order
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures_EA(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, example_id, input_ids, attention_mask, token_type_ids, label_id):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id


class InputFeatures_MBER(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, example_id, input_ids, attention_mask, token_type_ids, label_id):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id


def select_field_EA(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


class InputFeatures_MCBase(object):
    """A single set of features of an example in Multiple-Choices RC Tasks."""
    def __init__(self, example_id, choices_features, label_id):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label_id = label_id


def select_field_MC(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_txt(cls, input_file, quotechar=None):
        "Read a text file"
        lines = []
        with open (input_file, "r") as f:
            for line in f.readlines ():
                line = line.strip ('\n').strip ('\t')
                lines.append (line)
        return lines

    @classmethod
    def _read_csv_with_delimiter(cls, input_file, delimiter=','):
        lines = []
        with open(input_file, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for line in reader:
                lines.append(line)
        return lines

    @classmethod
    def _read_csv(cls, input_file):
        "Read a csv file"
        lines = []
        with open(input_file, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                lines.append(row)
        return lines
