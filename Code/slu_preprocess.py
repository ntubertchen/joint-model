import numpy as np
from collections import defaultdict

class slu_data():
    def __init__(self):
        train_nl = open('Data/train/seq.in', 'r')
        valid_nl = open('Data/valid/seq.in', 'r')
        test_nl = open('Data/test/seq.in', 'r')
        self.nl_dict = defaultdict()
        self.rev_nl_dict = defaultdict()
        self.build_dict(train_nl, valid_nl, test_nl)

    def build_dict(self, train_nl, valid_nl, test_nl):
        f = open('temp', 'w')
        for line in train_nl:
            temp_nl = line.strip('\n').split('***next***')[:-1] # temp_nl contains many sentences
            nl = self.clean_nl(temp_nl)
            f.write(' '.join(nl)+'\n')

    def clean_nl(self, temp_nl):
        ret = list()
        for sentence in temp_nl:
            # remove some puntuation marks
            temp = sentence.replace('~', '').strip(' ')
            # restore abbreviations to their original forms
            if '\'m' in temp:
                temp = temp.replace('\'m', ' am')
            if '\'re' in temp:
                temp = temp.replace('\'re', ' are')
            if '\'ll' in temp:
                temp = temp.replace('\'ll', ' will')
            if '\'s' in temp:
                temp = temp.replace('\'s', ' is')
            if '\'d' in temp:
                temp = temp.replace('\'d', ' would')
            if '\'ve' in temp:
                temp = temp.replace('\'ve', ' have')
            if 'don\'t' in temp:
                temp = temp.replace('don\'t', 'do not')
            if 'doesn\'t' in temp:
                temp = temp.replace('doesn\'t', 'does not')
            if 'hasn\'t' in temp:
                temp = temp.replace('hasn\'t', 'has not')
            if 'haven\'t' in temp:
                temp = temp.replace('daven\'t', 'have not')
            if 'wouldn\'t' in temp:
                temp = temp.replace('wouldn\'t', 'would not')
        
            ret.append(temp) 

        return ret

slu_data()
