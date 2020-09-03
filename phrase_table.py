import json

class phrase_table(object):

    def __init__(self, file="data/alignment_table_all_final.json"):
        with open(file) as infile:
            self.trans_dict = json.load(infile)


    # if there's no alignment of the word -> find out Chinese tokens that appear in the phrase table
    def find_no_align_trans(self, ch_sent, en_word, en_pat_token, verb_index, nlp):
        if en_word not in self.trans_dict:
            return []
        candidates = []
        en_word = nlp(en_word.lower())[0].lemma_ if nlp(en_word.lower())[0].lemma_ != '-PRON-' else en_word.lower()
        for j, w in enumerate(ch_sent.split()):
            if w in self.trans_dict[en_word].keys():
                candidates += [(j, en_pat_token, abs(verb_index-j))]
        candidates = sorted(candidates, key=lambda k: k[2]) # sort by distance between 'V' and the word
    #     print(en_word, [ch_sent.split()[z] for z,x,c in candidates], ch_sent)
        return [ (candidates[0][0], candidates[0][1]) ] if candidates else [] # [(ch_index, en_pat_token)]


    # filter out aligned Chinese token that are not in the phrase table
    def filter_ch_alignment(self, en_word, ch_sent, ch_indices, en_pat_token, verb_index, nlp):
        remain = []
        en_word = nlp(en_word.lower())[0].lemma_ if nlp(en_word.lower())[0].lemma_ != '-PRON-' else en_word.lower()
        for index in ch_indices:
            try:
                ch_word = ch_sent.split()[index]
            except:
                continue
    #             print(ch_sent, index)
            if en_word not in self.trans_dict: # not in phrase table -> assume the alignment is correct
                remain += [(index, en_pat_token)]
            elif ch_word in self.trans_dict[en_word].keys():
                remain += [(index, en_pat_token)]

        # if the en_word has no aligned Chinese tokens
        if not remain:
            return self.find_no_align_trans(ch_sent, en_word, en_pat_token, verb_index, nlp)
        return remain


PHRASE_TABLE = phrase_table()