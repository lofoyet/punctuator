# coding: utf-8
from __future__ import division

import os
import string

import numpy as np
import theano
import theano.tensor as T

import models
import data


PROJ_DIR = os.path.dirname(os.path.dirname(__file__))


class Punctuator(object):
    """Punctuator class, used to add missing puntuation."""

    puncs = set(string.punctuation)

    def __init__(
        self,
        tokenize_func,
        model_path=None,
    ):
        """Init."""
        default_model_path = os.path.join(
            PROJ_DIR, "resources", "Demo-Europarl-EN.pcl"
        )
        self.x = T.imatrix('x')
        self.model_path = model_path or default_model_path
        self.tokenize_func = tokenize_func  # cannot use on PySpark

    def load(self):
        """Load pretrain model."""
        print "Loading model."
        self.net, _ = models.load(self.model_path, 1, self.x)
        self._predict = theano.function(inputs=[self.x], outputs=self.net.y)
        self.word_vocabulary = self.net.x_vocabulary
        self.punctuation_vocabulary = self.net.y_vocabulary
        self.reverse_word_vocabulary = \
            {v: k for k, v in self.net.x_vocabulary.items()}
        self.reverse_punctuation_vocabulary = \
            {0: u'?',
             1: u'!',
             2: u' ',
             3: u',',
             4: u'-',
             5: u':',
             6: u';',
             7: u'.'}

        # {v: k for k, v in self.net.y_vocabulary.items()}

    def _tokenize_and_insert_punctuation(self, text):
        """Add missing puncation, no need to break long paragraph."""
        if not isinstance(text, unicode) or len(text) == 0:
            print "Wrong input"
            return
        tokens = self._tokenize(text)
        i = 0
        results = []
        while True:
            subsequence = tokens[i:i + data.MAX_SEQUENCE_LEN]
            if len(subsequence) == 0:
                break
            converted_array = self._convert(subsequence)
            y = self._predict(converted_array)
            sub_results, step = self._parse(y, subsequence)
            results += sub_results
            i += step
            if subsequence[-1] == data.END:
                break

        print results
        add_space_after_punc = []
        for token in results:
            if token in self.puncs:
                add_space_after_punc.append(token)
                add_space_after_punc.append(" ")
            else:
                add_space_after_punc.append(token)
        return add_space_after_punc

    def punctuate(self, text):
        """Add missing puncation, no need to break long paragraph."""
        inserted = self._tokenize_and_insert_punctuation(text)
        print inserted
        joined = u"".join(inserted)
        return joined

    def _tokenize(self, text):
        # tokenize text into words
        tokens = []
        for word in self.tokenize_func(text):
            if word not in self.punctuation_vocabulary:
                tokens.append(word)
        tokens += [data.END]
        return tokens

    def _convert(self, tokens):
        """Convert to input x."""
        converted = \
            [self.word_vocabulary.get(w, self.word_vocabulary[data.UNK])
                for w in tokens]
        return to_array(converted)

        # predict within each max len

    def _parse(self, y, tokens):
        """Parse y results."""
        results = tokens[0:1]
        last_eos_idx = 0
        punctuations = []
        for y_t in y:
            p_i = np.argmax(y_t.flatten())
            punctuation = self.reverse_punctuation_vocabulary[p_i]
            punctuations.append(punctuation)
            if punctuation in data.EOS_TOKENS:
                # we intentionally want the index of next element
                last_eos_idx = len(punctuations)

        if tokens[-1] == data.END:
            step = len(tokens) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(tokens) - 1

        for j in range(step):
            if punctuations[j] != data.SPACE:
                results.append(punctuations[j])
            else:
                results.append(u" ")
            if j < step - 1:
                results.append(tokens[j + 1])
        return results, step


def to_array(arr, dtype=np.int32):
    """Minibatch of 1 sequence as column."""
    return np.array([arr], dtype=dtype).T


def convert_punctuation_to_readable(punct_token):
    """Convert punctuation to readble format."""
    if punct_token == data.SPACE:
        return " "
    else:
        return punct_token[0]


if __name__ == "__main__":
    pass
    # if len(sys.argv) > 1:
    #     model_file = sys.argv[1]
    # else:
    #     sys.exit("Model file path argument missing")

    # show_unk = False
    # if len(sys.argv) > 2:
    #     show_unk = bool(int(sys.argv[2]))

    # with codecs.getwriter('utf-8')(sys.stdout) as f_out:
    #     while True:
    #         text = raw_input("\nTEXT: ").decode('utf-8')
    #         punctuate(predict, word_vocabulary, punctuation_vocabulary,
    #                   reverse_punctuation_vocabulary, reverse_word_vocabulary, text, f_out, show_unk)
