
class GDEX_score(object):
    
    def __init__(self):
        self.load_data()

    def load_data(self):
        with open('data/common_words.txt') as file:
            self.common_words = set([w.strip() for w in file])

    # rank examples by their GDEX score
    def rank_examples(self, exmples, en_len_avg, ch_len_avg, number=10):
        score_ex_id = [] # [(score, example_index)]
        sorted_examples = []
        for i, exmple in enumerate(exmples):
            score = self.example_score(exmple[2], exmple[3], en_len_avg, ch_len_avg)
            score_ex_id += [(score, i)]
        score_ex_id = sorted(score_ex_id, key=lambda k: -k[0])
        for score, i in score_ex_id[:number]:
            sorted_examples += [ exmples[i] ]
        return sorted_examples


    # calculate the GDEX score of each sentence pair
    # S(e) = 50 - P1 - # uncommon words - | r * ch_len - en_len |
    # concept1(P1): avg -7 <= len(sent) <= avg +7
    # concept2: uncommon words got panalty
    # concept3: len(en_sent) * r ~ len(ch_sent); r = avg(ch_len) / avg(en_len)
    def example_score(self, en_sent, ch_sent, en_len_avg, ch_len_avg):
        en_len = len(en_sent.split())
        ch_len = len(ch_sent.split())
        ratio = ch_len_avg/en_len_avg
        p = 0
        if abs(en_len-en_len_avg) > 7:
            p += abs(en_len - en_len_avg) - 7
        if abs(ch_len-ch_len_avg) > 7:
            p += abs(ch_len - ch_len_avg) - 7
        p += int(abs(en_len*ratio - ch_len)) # concept3
        for w in en_sent.split(): # concetp2
            if w not in self.common_words:
                p += 1
        return 50 - p

GDEX = GDEX_score()