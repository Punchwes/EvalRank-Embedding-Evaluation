import logging
import random
import spacy
from collections import defaultdict
from nltk.corpus import wordnet as wn
from transformers import pipeline
import string
import json
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

nlp = spacy.load("en_core_web_lg")
unmasker = pipeline('fill-mask', model='xlm-roberta-base', top_k=10)

def antonyms_for(word):
    """
    It seems that verbs are not guaranteed to have antonyms...... then how can we find out one to replace????
    Emmmm, tricky.
    """

    antonyms = set()
    for ss in wn.synsets(word):
        for lemma in ss.lemmas():
            if lemma.antonyms():
                any_pos_antonyms = [ antonym.name() for antonym in lemma.antonyms() ]
                for antonym in any_pos_antonyms:
                    antonym_synsets = wn.synsets(antonym)
                    as_pos = [ss.pos() for ss in antonym_synsets]
                    if wn.ADJ not in as_pos and wn.VERB not in as_pos: ### why this.....
                        continue
                    antonyms.add(antonym)
            else:
                continue
    return list(antonyms)


def add_hard_negative(all_sents, pos_pairs):
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
        we will not swap punctuation with others;
        """
        temp_tokenList = tokenList.copy()
        if temp_tokenList[-1] in string.punctuation:
            token_idx = list(range(len(temp_tokenList)-1))
        else:
            token_idx = list(range(len(temp_tokenList)))
        random.seed(2023)
        swap_idx = random.sample(token_idx, 2)
        temp_tokenList[swap_idx[1]], temp_tokenList[swap_idx[0]] = temp_tokenList[swap_idx[0]], temp_tokenList[swap_idx[1]]
        return " ".join(temp_tokenList)

    def lm_replace_word(tokenList):
        """
        we can choose to randomly mask or mask either verb or adj
        We can think of ways to generate high quality ones; ensure meaning changes;
        
        """
        temp_tokenList = tokenList.copy()
        if temp_tokenList[-1]  in string.punctuation:
            token_idx = list(range(len(temp_tokenList)-1))
        else:
            token_idx = list(range(len(temp_tokenList)))
        random.seed(2023)
        replace_idx = random.sample(token_idx, 1)[0]
        word_to_replace = tokenList[replace_idx]

        temp_tokenList[replace_idx] = "<mask>"

        masked_sent = " ".join(temp_tokenList)
        candidates = unmasker(masked_sent)
        for candidate in candidates:
            if candidate["token_str"].lower() != word_to_replace.lower() and candidate["token_str"].lower() not in ["an", "a", "the"] and candidate["token_str"].lower() not in string.punctuation:
                temp_tokenList[replace_idx] = candidate["token_str"]
                return " ".join(temp_tokenList)

        temp_tokenList[replace_idx] = "unknown"
        return " ".join(temp_tokenList)

    def replace_antonym(sent1_merged_word, sent1_merged_entity):
        ### we can use spacy wordnet
        ### we replace the verb or adj with its antonym; We just replace one word
        ### some adj and verbs do not have antonyms - how can we handle this;
        ###

        the_antonym = ""
        the_antonym_index = ""

        jj_idx_list = [idx_ for idx_, me in enumerate(sent1_merged_entity) if me in ["JJ", "JJR", "JJS"]]
        v_idx_list = [idx_ for idx_, me in enumerate(sent1_merged_entity) if (
                    me in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] and sent1_merged_word[idx_] not in ["was", "am", "are",
                                                                                                   "were", "is"])]
        if len(jj_idx_list) != 0:
            for jil in jj_idx_list:
                adj_word = sent1_merged_word[jil]
                adj_word_antonyms = antonyms_for(adj_word)
                if len(adj_word_antonyms) !=0:
                    adj_word_antonym = adj_word_antonyms[0]
                    the_antonym = adj_word_antonym
                    the_antonym_index = jil
                    break

        elif len(v_idx_list) != 0:
            for vil in v_idx_list:
                verb_word = sent1_merged_word[vil]
                verb_word_antonyms = antonyms_for(verb_word)
                if len(verb_word_antonyms) != 0:
                    verb_word_antonym = verb_word_antonyms[0]
                    the_antonym = verb_word_antonym
                    the_antonym_index = vil
                    break

        assert the_antonym != ""
        assert the_antonym_index != ""

        sent1_merged_word[the_antonym_index] = the_antonym

        replaced_sent = " ".join(sent1_merged_word)

        return replaced_sent

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
        root = [(token, idx) for idx, token in enumerate(parsed_sent) if token.dep_ == "ROOT"][0]  ## we want root

        merged_word, merged_entity = merge_entity(token_ent_label_iob, ent_labels, ents, tags, tokenized_sent)
        try:
            assert root[0].text in merged_word
        except:
            merged_word = tokenized_sent
            merged_entity = tags
            assert root[0].text in merged_word

        return merged_word, merged_entity, root, tokenized_sent

    for pair in pos_pairs:
        sent1, sent2 = pair

        sent1_merged_word, sent1_merged_entity, sent1_root, tokenized_sent1 = process_sentence(sent1, nlp)
        sent2_merged_word, sent2_merged_entity, sent2_root, tokenized_sent2 = process_sentence(sent2, nlp)

        """
        we first need to fix the seed (2023);
        okay, here we can:
         1)make a swap based on entity;
         2)add negation to the root; this might be a little bit limited
         3)replace verb/adj with antonym; (randomly choose a noun/verb/adj), however, not every verb/adj has antonym
         4)replace verb/adj with hypernym; (randomly choose a noun/verb/adj), however, not every verb/adj has hypernym
         5)lm random replace
         6)what else???
         
         We need to produce three versions - all, and each one separate;
        """

        sent1_swap = swap_word(sent1_merged_word) ## need to add more to enable swap with the same entity
        sent1_neg = insert_word(sent1_root[1], "not", tokenized_sent1) ### add not to root verb;
        sent1_replace = lm_replace_word(sent1_merged_word) ###
        # sent1_antonym = replace_antonym(sent1_merged_word, sent1_merged_entity)
        neg_dict[sent1] = [sent1_swap, sent1_neg, sent1_replace] #### it is important that we keep the order the same;

        sent2_swap = swap_word(sent2_merged_word) ### swap words
        sent2_neg = insert_word(sent2_root[1], "not", tokenized_sent2) ### add not to root verb;
        sent2_replace = lm_replace_word(sent2_merged_word) ### LLM replace word
        # sent2_antonym = replace_antonym(sent2_merged_word, sent2_merged_entity) ### replace with antonym
        neg_dict[sent2] = [sent2_swap, sent2_neg, sent2_replace] #### it is important that we keep the order the same;

        all_sents.append(sent1_swap)
        all_sents.append(sent1_neg)
        all_sents.append(sent1_replace)

        all_sents.append(sent2_swap)
        all_sents.append(sent2_neg)
        all_sents.append(sent2_replace)

        """
        okay, here we write them to local file,
        we need to write the neg_dict && the new all_sents file;
        we can separate these bits of functions into a new one called augment_dataset.py
        """

    return all_sents, neg_dict



def build_basic_sents(pos_pairs):
    ''' build basic background sentences from pos pairs '''
    all_sents = []

    for item in pos_pairs:
        if item[0] not in all_sents: all_sents.append(item[0])
        if item[1] not in all_sents: all_sents.append(item[1])

    logging.info('{} sentences as background sentences'.format(len(all_sents)))

    return all_sents

pos_pairs = [] # positive sentence pairs
hard_negative = True
if not hard_negative:
    neg_dict = None

logging.info('')
logging.info('*** Prepare pos sentence pairs for ranking evaluation ***')

rank_data_path = '../data/sent_evalrank/'

with open(rank_data_path + 'pos_pair.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        sent1, sent2 = line.strip().split('\t')
        pos_pairs.append([sent1, sent2])
logging.info('{} positive pairs collected from STSB dataset'.format(len(pos_pairs)))

logging.info("")
logging.info("Loading Background Sentences for Ranking")

all_sents = build_basic_sents(pos_pairs)
with open(rank_data_path + 'background_sent.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line not in all_sents:
            all_sents.append(line)

logging.info('{} sentences as background sentences'.format(len(all_sents)))

# if hard_negative:
all_sents, neg_dict = add_hard_negative(all_sents, pos_pairs)


"""
Okay, here we can write the all_sents and neg_dict to local;
Okay, I think here we should come up with two styles, both should remove the index of the identical sentence when comparing;
"""

output_path = "../data/sent_evalrank/augmented_all_sents.txt"
output_neg_dict_path = "../data/sent_evalrank/neg_dict.json"

with open(output_neg_dict_path, 'w') as f:
    json.dump(neg_dict, f, indent=4)
with open(output_path, 'w') as f:
    for sent in all_sents:
        f.write(sent + "\n")









