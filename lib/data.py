# coding: utf-8
from __future__ import division

import os

DATA_PATH = "../data"

# path to text file in the format:
# word1 0.123 0.123 ... 0.123
# word2 0.123 0.123 ... 0.123 etc...
# e.g. glove.6B.50d.txt
PRETRAINED_EMBEDDINGS_PATH = None

END = "</S>"
UNK = "<UNK>"

SPACE = "_SPACE"

MAX_WORD_VOCABULARY_SIZE = 100000
MIN_WORD_COUNT_IN_VOCAB = 2
MAX_SEQUENCE_LEN = 50

TRAIN_FILE = os.path.join(DATA_PATH, "train")
DEV_FILE = os.path.join(DATA_PATH, "dev")
TEST_FILE = os.path.join(DATA_PATH, "test")

# Stage 2
TRAIN_FILE2 = os.path.join(DATA_PATH, "train2")
DEV_FILE2 = os.path.join(DATA_PATH, "dev2")
TEST_FILE2 = os.path.join(DATA_PATH, "test2")

WORD_VOCAB_FILE = os.path.join(DATA_PATH, "vocabulary")

PUNCTUATION_VOCABULARY = [SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK",
                          "!EXCLAMATIONMARK", ":COLON", ";SEMICOLON", "-DASH"]
PUNCTUATION_MAPPING = {}

# Comma, period & question mark only:
# PUNCTUATION_VOCABULARY = {SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK"}
# PUNCTUATION_MAPPING = {"!EXCLAMATIONMARK": ".PERIOD", ":COLON": ",COMMA",
# ";SEMICOLON": ".PERIOD", "-DASH": ",COMMA"}

EOS_TOKENS = {".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"}
# punctuations that are not included in vocabulary nor mapping, must be
# added to CRAP_TOKENS
CRAP_TOKENS = {"<doc>", "<doc.>"}
PAUSE_PREFIX = "<sil="
