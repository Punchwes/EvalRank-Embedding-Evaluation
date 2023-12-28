#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-20 15:39:12
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import logging
import random
import spacy
from collections import defaultdict


nlp = spacy.load("en_core_web_lg")

class Sent_ranking_dataset_loader:
    ''' dataset loader for sentence ranking task '''

    def __init__(self, config) -> None:
        ''' class initialization '''
        
        self.pos_pairs = [] # positive sentence pairs
        self.all_sents = [] # background sentences
        self.hard_negative = False

        logging.info('')
        logging.info('*** Prepare pos sentence pairs for ranking evaluation ***')
        
        self.rank_data_path = '../data/sent_evalrank/'

        with open(self.rank_data_path + 'pos_pair.txt', 'r') as f: 
            lines = f.readlines()
            for line in lines:
                sent1, sent2 = line.strip().split('\t')
                self.pos_pairs.append([sent1, sent2])
        logging.info('{} positive pairs collected from STSB dataset'.format(len(self.pos_pairs)))


        logging.info("")
        logging.info("Loading Background Sentences for Ranking")
        
        self.build_basic_sents()

        with open(self.rank_data_path + 'background_sent.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line not in self.all_sents:
                    self.all_sents.append(line)

        logging.info('{} sentences as background sentences'.format(len(self.all_sents)))

        if self.hard_negative:
            self.neg_dict = self.add_hard_negative()
        else:
            self.neg_dict = None

    def build_basic_sents(self):
        ''' build basic background sentences from pos pairs '''

        for item in self.pos_pairs:
            if item[0] not in self.all_sents: self.all_sents.append(item[0])
            if item[1] not in self.all_sents: self.all_sents.append(item[1])

        logging.info('{} sentences as background sentences'.format(len(self.all_sents)))

    def add_hard_negative(self):
        neg_dict = {}
        def insert_word(pos, token, tokenList):
            """
            here we want to insert word, however, I think this will be mainly used for negation
            we probably will negate the root verb all the time
            """
            tokenList.insert(pos, token)
            return " ".join(tokenList)

        def swap_word(tokenList):
            """
            We here try to swap words, but we want to swap words with the same entity
            """
            token_idx = list(range(len(tokenList)))
            swap_idx = random.sample(token_idx, 2)
            tokenList[swap_idx[1]], tokenList[swap_idx[0]] = tokenList[swap_idx[0]], tokenList[swap_idx[1]]
            return " ".join(tokenList)

        def replace_antonym():
            ### we can use spacy wordnet
            return
        def replace_hypernym():
            ### we can use spacy wordnet
            return

        def merge_entity(token_ent_label_iob, ent_labels, ents, tags, tokenized_sent):
            grouped_list = []
            span_idx = 0
            for idxx, teli in enumerate(token_ent_label_iob):
                if teli == 'O':
                    grouped_list.append(idxx)
                elif teli == "B":
                    grouped_list.append([idxx])
                    span_idx = len(grouped_list) - 1
                elif teli == "I":
                    grouped_list[span_idx].append(idxx)

            merged_entity = []
            merged_words = []
            for idxxx, gl in enumerate(grouped_list):
                if type(gl) is list:
                    merged_entity.append(ent_labels[0])
                    ent_labels.pop(0)
                    merged_words.append(ents[0])
                    ents.pop(0)
                else:
                    merged_entity.append(tags[gl])
                    merged_words.append(tokenized_sent[gl])

            assert len(merged_entity) == len(merged_words)

            # type_dict = defaultdict(list)
            # for idxxx, me in enumerate(merged_entity):
            #     type_dict[me].append(merged_words[idxxx])

            return merged_words, merged_entity

        def process_sentence(sentence, nlp):
            parsed_sent = nlp(sentence)
            tokenized_sent = [token.text for token in parsed_sent]
            tags = [token.tag_ for token in parsed_sent]
            ents = [ent.text for ent in parsed_sent.ents]
            ent_labels = [ent.label_ for ent in parsed_sent.ents]
            token_ent_label_iob = [token.ent_iob_ for token in parsed_sent]
            root = [(token, idx) for idx, token in enumerate(parsed_sent) if token.dep_ == "ROOT"][0] ## we want root

            merged_word, merged_entity = merge_entity(token_ent_label_iob, ent_labels, ents, tags, tokenized_sent)
            assert root[0] in merged_word

            return merged_word, merged_entity, root

        for pair in self.pos_pairs:
            sent1, sent2 = pair
            # sent1_merged_word, sent1_merged_entity, sent1_root = process_sentence(sent1)
            # sent2_merged_word, sent2_merged_entity, sent2_root = process_sentence(sent2)
            """
            okay, here we can:
             1)make a swap based on entity;
             2)add negation to the root;
             3)replace verb/adj with antonym;
             4)replace verb/adj with hypernym;
             5)what else???
            """

            sent1_swap =  swap_word(sent1.split())
            sent1_neg = insert_word(3, "not", sent1.split())
            neg_dict[sent1] = [sent1_swap, sent1_neg]

            sent2_swap =  swap_word(sent2.split())
            sent2_neg = insert_word(3, "not", sent2.split())
            neg_dict[sent2] = [sent2_swap, sent2_neg]

            self.all_sents.append(sent1_swap)
            self.all_sents.append(sent1_neg)

            self.all_sents.append(sent2_swap)
            self.all_sents.append(sent2_neg)

            """
            okay, here we write them to local file,
            we need to write the neg_dict && the new all_sents file;
            we can separate these bits of functions into a new one called augment_dataset.py
            """

        return neg_dict

    def build_indexs(self):
        """
        run this function after init
        """
        rank_dict = {}

        for pair in self.pos_pairs:

            sent1_remove_idx = []
            sent2_remove_idx = []

            sent1, sent2 = pair

            sent1_remove_idx.append(sent1) ### here we remove the index of itself
            sent2_remove_idx.append(sent2) ### here we remove the index of itself

            if self.hard_negative:
                sent1_negs = self.neg_dict[sent1]
                sent2_negs = self.neg_dict[sent2]
                sent1_hard_negative_idx = sent1_negs ## the hard-negatives derived from sentence1, we do not want them in the comparison with sentence1
                sent2_hard_negative_idx = sent2_negs ## the hard-negatives derived from sentence2

                sent1_remove_idx.extend(sent1_hard_negative_idx)
                sent2_remove_idx.extend(sent2_hard_negative_idx)

            #### so for the rank_dict, we store the sentences that need to be compared to the pivot sentence;
            rank_dict[sent1] = [x for x in self.all_sents if x not in sent1_remove_idx]
            rank_dict[sent2] = [x for x in self.all_sents if x not in sent2_remove_idx]

        return rank_dict












