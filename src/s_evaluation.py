#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2022-03-20 17:23:19
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import sys
import logging

import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

import senteval
from nltk.util import ngrams
import sklearn
import random


# Set params for SentEval (for fast prototyping)
params_senteval = {'task_path': './data/', 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}


'''
# Set params for SentEval (for better performance)
params_senteval.update({'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10})
params_senteval['classifier'] = {'nhid': 50, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
'''


class Sent_emb_evaluator:
    ''' run evaluation by similarity and ranking '''

    def __init__(self, config, sent_pairs_data, sent_emb_model) -> None:
        ''' initialization for sentence embedder '''

        self.config = config

        self.eval_by_ranking        = 'ranking' in config.eval_type
        self.eval_by_similarity     = 'similarity' in config.eval_type
        self.eval_by_classification = 'classification' in config.eval_type
        self.eval_by_jaccard        = 'jaccard' in config.eval_type
        self.eval_by_bow            = 'bow' in config.eval_type

        self.sent_pairs_data = sent_pairs_data
        self.sent_emb_model  = sent_emb_model

    def eval(self):
        ''' main function for evaluation '''

        bow_sim = None
        jaccard_sim = None
        res_rank = None
        sent_sim = None
        res_cls = None

        if self.eval_by_bow:
            logging.info("")
            logging.info('*** Evaluation on sentence BOW task ***')
            bow_sim = self.eval_for_bow_sim()

        if self.eval_by_jaccard:
            logging.info("")
            logging.info('*** Evaluation on sentence Jaccard task ***')
            jaccard_sim = self.eval_for_jaccard()

        if self.eval_by_ranking:
            logging.info('')
            logging.info('*** Evaluation on sentence ranking task ***')
            res_rank = self.eval_for_ranking()

        if self.eval_by_similarity:
            import senteval
            logging.info('')
            logging.info('*** Evaluation on sentence similarity tasks ***')
            sent_sim = self.eval_for_similarity()

        if self.eval_by_classification:
            import senteval
            logging.info('')
            logging.info('*** Evaluation sentence classification tasks ***')
            res_cls = self.eval_for_classification()
        
        return bow_sim, jaccard_sim, sent_sim, res_rank, res_cls


    def prepare_nonorm(self, params, samples):
        ''' batcher for preparation '''        

        samples = [' '.join(sent) if sent != [] else '.' for sent in samples]
        self.sent_emb_model.embedder_all(samples, normalization=False)


    def batcher(self, params, batch):
        ''' obtain original sentence embedding given a batch '''

        batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
        embedding = self.sent_emb_model.embed(batch)

        return embedding

    def jaccard_similarity(self, str1, str2, n):
        str1_bigrams = list(ngrams(str1.split(), n))
        str2_bigrams = list(ngrams(str2.split(), n))

        intersection = len(list(set(str1_bigrams).intersection(set(str2_bigrams))))
        union = (len(set(str1_bigrams)) + len(set(str2_bigrams))) - intersection

        return float(intersection) / union

    def eval_for_bow_sim(self):

        bow_sim = None

        total_number_pairs = 0

        temp_all_sents = self.sent_pairs_data.all_sents
        temp_all_sents = list(set(temp_all_sents))

        vectorizer = sklearn.feature_extraction.text.CountVectorizer()
        bow = vectorizer.fit_transform(temp_all_sents)

        pos_sim = []
        neg_sim = []
        neg_sim_max = []
        times_neg_higher = []
        for pair_idx, pair in enumerate(tqdm(self.sent_pairs_data.pos_pairs)):

            s1, s2 = pair

            if len(s1.split()) <=7:
                total_number_pairs += 1
                continue

            s1_idx_in_all_sents = temp_all_sents.index(s1)
            s2_idx_in_all_sents = temp_all_sents.index(s2)
            remain_idx = [iddx for iddx, in_sent in enumerate(temp_all_sents) if (iddx!=s1_idx_in_all_sents and iddx!=s2_idx_in_all_sents)]

            s1_bow = bow[s1_idx_in_all_sents]
            s2_bow = bow[s2_idx_in_all_sents]
            # remain_in_all_sents = [in_sent for iddx, in_sent in enumerate(temp_all_sents) if (iddx!=s1_idx_in_all_sents and iddx!=s2_idx_in_all_sents)]
            all_remain_bow = bow[remain_idx]

            positives = sklearn.metrics.pairwise.cosine_similarity(s1_bow, s2_bow)
            negatives = sklearn.metrics.pairwise.cosine_similarity(s1_bow, all_remain_bow)

            max_neg = np.max(negatives)
            pos_sim.append(positives.item())
            neg_sim.append(np.mean(negatives))

            neg_sim_max.append(max_neg)
            if max_neg >= positives.item():
                times_neg_higher.append(True)
            else:
                times_neg_higher.append(False)


        bow_sim = {"positive pair": np.mean(pos_sim), "positive pair max": np.max(pos_sim),
                   "negative pair": np.mean(neg_sim), "negative pair max": np.max(neg_sim_max),
                   "negative pair mean over max": np.mean(neg_sim_max),
                   "Times negative higher/equal to positive": sum(times_neg_higher)/len(times_neg_higher),
                   "total number of positive pairs compared": len(self.sent_pairs_data.pos_pairs)-total_number_pairs
        }

        logging.info('Experimental results on bow sim')
        logging.info("\n"+str(bow_sim))

        return bow_sim


    def eval_for_jaccard(self):

        jaccard_sim = None

        all_neg_avg_jaccard_sims = []
        all_pos_avg_jaccard_sims = []
        temp_all_sents = self.sent_pairs_data.all_sents

        for pair_idx, pair in enumerate(tqdm(self.sent_pairs_data.pos_pairs)):

            s1, s2 = pair

            """
            here we calculate the jaccard similarity
            """

            s1_idx_in_all_sents = temp_all_sents.index(s1)
            s2_idx_in_all_sents = temp_all_sents.index(s2)
            remain_in_all_sents = [in_sent for iddx, in_sent in enumerate(temp_all_sents) if (iddx!=s1_idx_in_all_sents and iddx!=s2_idx_in_all_sents)]
            pos_jaccard = self.jaccard_similarity(s1, s2, 1)
            neg_jaccards = []
            for rias in remain_in_all_sents:
                neg_jaccards.append(self.jaccard_similarity(s1, rias, 1))
            avg_neg_jaccard = np.mean(neg_jaccards)

            all_neg_avg_jaccard_sims.append(avg_neg_jaccard)
            all_pos_avg_jaccard_sims.append(pos_jaccard)

        jaccard_sim = {'positive pair': np.mean(all_pos_avg_jaccard_sims),
                    'negative pairs': np.mean(all_neg_avg_jaccard_sims)}
        logging.info('Experimental results on jaccard')
        logging.info("\n"+str(jaccard_sim))

        return jaccard_sim

    def insert_word(self, pos, token, tokenList):
        tokenList.insert(pos, token)
        return " ".join(tokenList)
    def swap_word(self, tokenList):
        token_idx = list(range(len(tokenList)))
        swap_idx = random.sample(token_idx, 2)
        tokenList[swap_idx[1]], tokenList[swap_idx[0]] = tokenList[swap_idx[0]], tokenList[swap_idx[1]]
        return " ".join(tokenList)


    def eval_for_ranking(self):
        ''' evaluate the sentence embeddings with ranking task '''

        hits_max_bound = 15
        res_rank       = None

        """
        here we try to take one swap and add not for s1
        also, I think we need to remove the identical sentence..... in a clever way;;;;;
        we could preserve an index list of all corresponding paired sentences;;;;
        
        """
        rank_dict = self.sent_pairs_data.build_indexs()

        # all_sentsss = []
        # counting = 0
        # for sent_idx, sent in enumerate(self.sent_pairs_data.all_sents):
        #     if sent_idx%2==0: #jump every other token
        #         continue
        #     s1_tokens = sent.split()
        #     all_sentsss.append(self.swap_word(s1_tokens))
        #     all_sentsss.append(self.insert_word(3, "not", s1_tokens))
        #     counting +=1
        #     if counting%1000==0:
        #         print (counting)
        # self.sent_pairs_data.all_sents = self.sent_pairs_data.all_sents + all_sentsss

        # pre-compute all embeddings
        logging.info("Pre-compute all embeddings")
        self.sent_emb_model.embedder_all(self.sent_pairs_data.all_sents, normalization=self.config.normalization, centralization=True)

        # embedding
        # sents_embs = self.sent_emb_model.embed(self.sent_pairs_data.all_sents) ## original

        ranks      = []
        idx_count = 0

        for pair_idx, pair in enumerate(tqdm(self.sent_pairs_data.pos_pairs)):
            # if pair_idx%2!=0: #jump every other token
            #     continue

            s1, s2 = pair
            s1_emb  = self.sent_emb_model.embed([s1])
            s2_emb  = self.sent_emb_model.embed([s2])

            """
            Okay, here we do not use all sents, we use indexes to access sentences we want
            we use this even without adding hard-negatives, because we remove the same sentence in the process as well
            """
            sents_embs = self.sent_emb_model.embed(rank_dict[s1])

            if self.config.dist_metric == 'cos':
                pos_score         = np.dot(s1_emb, s2_emb.T).squeeze()
                background_scores = np.dot(sents_embs, s1_emb.T) ### original one

                background_scores = np.squeeze(background_scores)
                index_rank = np.argsort(background_scores)[::-1]
                background_scores = np.sort(background_scores)[::-1] ### 从大到小排序

                """
                OKAY, here we need to check with the lexical overlap between sentences extra... 
                We can do a full analysis of this and then propose our solution.
                """

            elif self.config.dist_metric == 'l2':
                pos_score         = 1 / (np.linalg.norm(s1_emb - s2_emb) + 1)
                background_scores = 1 / (np.linalg.norm((sents_embs - s1_emb),axis=1) + 1)
                background_scores = np.sort(background_scores)[::-1]

            else:
                sys.exit("Distance Metric NOT SUPPORTED: {}".format(self.config.dist_metric))

            #### Problem/Error: this calculation is a little bit problematic here, because the pos_score sometimes are unexpectedly lower than the same bg_score
            #### so we guess we need to remove the identical sentence; after removing identical sentence
            rank = len(background_scores) - np.searchsorted(background_scores[::-1], pos_score, side='right')
            if rank == 0: rank = 1
            ranks.append(int(rank))
            idx_count += 1

        MR  = np.mean(ranks)
        MRR = np.mean(1. / np.array(ranks))

        hits_scores = []
        for i in range(hits_max_bound): hits_scores.append(sum(np.array(ranks)<=(i+1))/len(ranks))

        res_rank = {'MR'    : MR,
                    'MRR'   : MRR}

        for i in range(hits_max_bound): res_rank['hits_'+str(i+1)]  = hits_scores[i]

        table = PrettyTable(['Scores', 'Emb'])
        table.add_row(['MR', MR])
        table.add_row(['MRR', MRR])
        for i in range(hits_max_bound): 
            if i in [0,2]:
                table.add_row(['Hits@'+str(i+1), res_rank['hits_'+str(i+1)]])
        logging.info('Experimental results on ranking')
        logging.info("\n"+str(table))

        return res_rank
        

    def eval_for_similarity(self):
        ''' perform evaluation on similarity tasks '''

        sent_sim = None
        
        transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness', 'STR', 'STSB_Supervised', 'SICKRelatedness_Supervised']

        se = senteval.engine.SE(params_senteval, self.batcher, self.prepare_nonorm)
        results = se.eval(transfer_tasks)

        # report result
        table = PrettyTable(['Embs', 'DATASET', 'Pearson', 'Spearman', 'Kendall'])
        for dataset, values in results.items():
            if dataset in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                table.add_row([self.config.sent_emb_model, dataset+'_ALL', results[dataset]['all']['pearson']['wmean'], results[dataset]['all']['spearman']['wmean'], results[dataset]['all']['kendall']['wmean']])
            elif dataset in ['STSBenchmark', 'SICKRelatedness', 'STR', 'STSB_Supervised', 'SICKRelatedness_Supervised']:
                table.add_row([self.config.sent_emb_model, dataset, results[dataset]['pearson'], results[dataset]['spearman'], results[dataset]['kendall']])
        sent_sim = results

        logging.info('Experimental results on similarity for original sentence embeddings (weighted-average)')
        logging.info("\n"+str(table))

        return sent_sim
        

    def eval_for_classification(self):
        '''
            evaluate the sentence embedding with classificaition / downstream tasks
        '''
        results = None

        transfer_tasks = ['SCICITE', 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKEntailment']

        # evaluation for original embedding and report result
        se = senteval.engine.SE(params_senteval, self.batcher, self.prepare_nonorm)
        results = se.eval(transfer_tasks)

        # results
        logging.info("Classification results on sentence embedding")
        table = PrettyTable(['Dataset', 'SentEmb'])
        for dataset in transfer_tasks:
            table.add_row([dataset, results[dataset]['acc']])
        logging.info("\n"+str(table))

        return results



