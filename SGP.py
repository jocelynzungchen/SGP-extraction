from collections import defaultdict, Counter
import spacy
import json
import joblib
from data_process import to_crf_format, get_pattern_instance, sim_pattern_form, revert_to_simple_form, sent2features
from get_ch_patterns import get_ch_pat_instance
from phrase_table import PHRASE_TABLE

nlp = spacy.load('en_core_web_lg')
merge_nps = nlp.create_pipe("merge_noun_chunks")
nlp.add_pipe(merge_nps)


class SGP(object):

    def __init__(self):
        self.load_data()
        # verb_dict[en_verb][en_pat][ch_pat] = [(en_instance, ch_instance), ...]
        self.verb_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # verb_trans[en_verb][en_pat][ch_pat][ch_trans] = count
        self.verb_trans = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        self.i = 0

    def load_data(self):
        with open('data/cobuild_all_patterns.txt') as f1, open('data/crf_model_passive.pkl', 'rb') as f2:
            for line in f1:
                self.cobuild_all_patterns = eval(line)
            self.crf = joblib.load(f2) # load crf model

        self.cobuild_prep = {'about', 'across', 'after', 'against', 'among', 'around', 'as', 'at', 'between', 'by', 
        'for', 'from', 'in', 'into', 'like', 'not', 'of', 'off', 'on', 'onto', 'out', 'over', 'round', 'so', 
        'though', 'through', 'to', 'together', 'toward', 'towards', 'under', 'with'}

    
    def process_file(self, file_path, align_file_path, ch_pos_file_path, break_number = 10):
        print(file_path, align_file_path, ch_pos_file_path)
        with open(file_path) as infile, open(align_file_path) as align_file, open(ch_pos_file_path) as ch_pos_file:
            self.i = 0
            for line, alignment, ch_pos in zip(infile, align_file, ch_pos_file):
                line = line.split('|||')
                en_sent = self._clean_en_sent(line[0].strip())
                ch_sent = line[1].strip().replace('裏', '裡')
                if len(ch_pos.split()) != len(ch_sent.split()): # ignore ckip parsing error
                    continue
                sent_align = self._get_sent_align(alignment)
                self.extract_SGP(en_sent, ch_sent, sent_align, ch_pos)
                sent_align.clear()
                if self.i>break_number:
                    break

    def extract_SGP(self, en_sent, ch_sent, sent_align, ch_pos):
        # step1: identify Englisg pattern instance
        try:
            sent = to_crf_format(en_sent)
        except:
            return
        test_pred = self.crf.predict([sent2features(sent)])
        instances = get_pattern_instance(sent, test_pred[0])

        # for every English pattern instance: find the Chinese counterpart
        for instance in instances:
            en_pat = sim_pattern_form([ p for p, w, num in instance])
            # TODO: en_pat 是否再做處理（例如 extract 出來的 pattern 過長，只取部分會在 COBUILD patterns 裡面）
            if revert_to_simple_form(en_pat) != en_pat or en_pat not in self.cobuild_all_patterns: # ignore the passive form # TODO: cope with the passive form
                continue

            # step2: use alignment file to obtain headword and Chinese pattern instance
            headword_en_index, headword_ch_index, ch_indices = self._get_headword_index_ch_indices(instance, ch_sent, sent_align)
            if not ch_indices or not headword_en_index: # or not headword_ch_index
                continue

            # -----------------
            headword_ch_index.sort()
            headword_ch_new = []
            for j in range(len(headword_ch_index)):
                if headword_ch_index[j] < headword_ch_index[j-1]+3: # 間距不超過 3 個字
                    headword_ch_new += [headword_ch_index[j]]
            # -----------------

            # step3: for the obtained Chinese pattern instance: do some addiotnal process (eg. filter and re-align prep)
            ch_pat, ch_instance = get_ch_pat_instance(ch_indices, ch_pos.split(), ch_sent, headword_ch_new)
            if not ch_pat:
                continue
            en_instance = ' '.join([ w if p != 'V' else '['+w+']' for p, w, num in instance])

            # step4: save the result
            # TODO: 是否要處理 'wh' 和 'that' 在 en_pat.split()
            en_sent_new = ' '.join([ word if k not in headword_en_index else '['+word+']' for k, word in enumerate(en_sent.split())])
            ch_sent_new = ' '.join([ word if k not in headword_ch_index else '['+word+']' for k, word in enumerate(ch_sent.split())])
            headword_en = ' '.join([ nlp(en_sent.split()[k])[0].lemma_ for k in headword_en_index ])
            headword_ch = ' '.join([ ch_sent.split()[k] for k in headword_ch_index ])
            if (en_instance, ch_instance, en_sent_new, ch_sent_new) not in self.verb_dict[headword_en][en_pat][ch_pat]:
                self.verb_dict[headword_en][en_pat][ch_pat] += [(en_instance, ch_instance, en_sent_new, ch_sent_new)]
                self.verb_trans[headword_en][en_pat][ch_pat][headword_ch] += 1
                self.i += 1


    def _clean_en_sent(self, en_sent): # for Cambridge sent
        en_sent = en_sent.replace("'ve", " have") # 針對 spacy 會把 I've 斷成 I 've
        if en_sent[0] == '[':
            return en_sent[en_sent.index(']')+1:]
        return en_sent


    # sent_align[en_token_index] = [ch_token_indices]
    def _get_sent_align(self, alignment):
        sent_align = defaultdict(list)
        for pair in alignment.split():
            pair = pair.split('-')
            sent_align[int(pair[0])] += [int(pair[1])]
        return sent_align


    def _get_headword_index_ch_indices(self, instance, ch_sent, sent_align):
        ch_indices = []
        headword_en_index = [] # 為了標記出在句子的哪個地方
        headword_ch_index = [] # 為了標記出在句子的哪個地方
        for pat_word in instance: # pat_word: ('V', word, index)
            if pat_word[1].lower() in 'must/may/will/would/can/could/should/n\'t/n’t/'.split('/') or pat_word[1][0] in ['’', '\'']: # exclude "V" and abbreviations
                continue
            if pat_word[0] in self.cobuild_prep: # for 'prep' pattern label -> re-align by stanfordparser
                continue
            headword_ch_index.sort()
            verb_index = headword_ch_index[0] if headword_ch_index else 0
            if pat_word[2] in sent_align:
                # filter out translation not in phrase table
                remain = PHRASE_TABLE.filter_ch_alignment(pat_word[1], ch_sent, sent_align[pat_word[2]], pat_word[0], verb_index, nlp)
            else: # no alignment of the pattern instance token (eg. 'me' of 'Ask me')
                remain = PHRASE_TABLE.find_no_align_trans(ch_sent, pat_word[1], pat_word[0], verb_index, nlp)
            ch_indices += remain
            if pat_word[0] == 'V':
                headword_en_index.append(pat_word[2])
                headword_ch_index += [j for j, pos in remain if j not in headword_ch_index]
        ch_indices = sorted(ch_indices, key=lambda k: k[0])
        return headword_en_index, headword_ch_index, ch_indices

    def write_to_file(self, file_1='SGP_result_1.json', file_2='SGP_trans.json'):
        with open(file_1, 'w') as outf1, open(file_2, 'w') as outf2:
            json.dump(self.verb_dict, outf1, ensure_ascii=False)
            json.dump(self.verb_trans, outf2, ensure_ascii=False)



SGP_ = SGP()