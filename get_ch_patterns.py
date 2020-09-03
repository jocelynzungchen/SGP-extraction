from typing import List, Tuple
from collections import defaultdict
import json
import stanfordnlp
import warnings
warnings.simplefilter("ignore", UserWarning)

stanford_parser = stanfordnlp.Pipeline(lang='zh', tokenize_pretokenized=True)

with open('data/annotation_prep.txt') as file_1, open("data/ch_pat_count.json") as file_2:
    reserved_words = set([ w.strip() for w in file_1 ])
    ch_pat_count = json.load(file_2)

# "vh" could be "adj" or "v"
# gen_vh_possibilities('V vh n vh n'.split(), 0, []) 
# -> ['V v n v n', 'V v n adj n', 'V adj n v n', 'V adj n adj n']
def gen_vh_possibilities(text: List[str], index: int, poss: List[str]) -> List[str]:
    
    def gen_cand(list_):
        new_list = []
        for string in list_:
            new_list += [string+' v']
            new_list += [string+' adj']
        return new_list

    if index >= len(text):
        return poss
    if index == 0:
        poss = [text[0]]
    else:
        poss = gen_cand(poss) if text[index] == 'vh' else [string + ' ' + text[index] for string in poss]
    return gen_vh_possibilities(text, index+1, poss)


# Simplify the Chinese pattern
# follow the below rules:
# rule1: for 'vh' -> generate two possibilities 'v' & 'adj'
# rule2: if no adj/adv in English pattern: 'adv adj n' (or) 'adj n' (or) 'adv n' -> 'n' (or) 'adv v' -> 'v'
# rule3: 'n conj n' -> 'n'
# rule4: for English pattern 'V that' & 'V wh': check 'V clause' condition
# rule5: if headword align to more than one word -> filter out the word with distane > 3
# TODO: 何時保留 DE? if English pattern contains more than one 'n'?
def sim_ch_pattern(ch_pat: List[str], en_pat: str) -> str:
    remain = []
    output = []
    # step1: transform "vh" to "adj" or "v"
    pats = gen_vh_possibilities(ch_pat, 0, []) # rule1
    # step2: for each pattern candidate: check rules
    for pat in pats:
        if 'adj' not in en_pat: # rule2
            pat = pat.replace('adj n', 'n')
        if 'adv' not in en_pat: # rule2
            pat = pat.replace('adv n', 'n').replace('adv v', 'v')
        remain += [pat.replace('n conj n', 'n')] # rule3
    # step3: for each remain patterns: check if it is in annotated data or meets 'clause condition'
    for p in remain:
        p = transform_ch_pat(p)
        if en_pat in ['V that', 'V wh'] and (p.startswith('V n v') or p.startswith('V v n v')): # rule4
            return 'V clause'
        if p in ch_pat_count: # 只保留出現在 annotated data 中的 pattern
            output += [(p, ch_pat_count[p])]
    # step4: sort the final patterns by the count in annotated data
    output = sorted(output, key=lambda k: -k[1])
    return output[0][0] if output else ''

# amount -> Nf開頭(量詞), Neu, Neq開頭; adj -> A, VH; adv -> Da, Df開頭, D;
# T(語助詞) 省略; DE /*的, 之, 得, 地*/; SHI /*是*/; FW /*外文標記*/
def sim_ch_pos(pos):
    if pos == 'A':
        pos = 'adj'
    elif pos in ['D', 'Da'] or pos.startswith('Df'):
        pos = 'adv'
    elif pos.startswith('N') or pos.startswith('F'):
        pos = 'n'
    elif pos.startswith('VH'):
        pos = 'vh'
    elif pos.startswith('V') or pos == 'SHI':
        pos = 'v'
    elif pos == 'Caa':
        pos = 'conj'
    elif pos == 'DE':
        pos = 'de'
    else:
        pos = pos[0].lower()
    return pos


# Transforming original Chinese pattern to the form of 'p n V' (for simply matching with annotated data)
# eg. '在 n 後面 V' -> '在^後面 n V'
def transform_ch_pat(pat):
    new_pat = []
    for i, p in enumerate(pat.split()):
        if p in reserved_words and len(pat.split()) > i+2 and pat.split()[i+1] in ['n', 'pl-n'] and pat.split()[i+2] in reserved_words:
            new_pat += [p+'^'+pat.split()[i+2]]
            new_pat += [pat.split()[i+1]]
            break
        else:
            new_pat += [p]
    for j in range(i+3, len(pat.split())): # add the rest of the pattern tokens
        new_pat += [pat.split()[j]]
    return ' '.join(new_pat)


# Re-align English prep by Stanford parser
# check if there are preps outside instance tokens that share dependency with instance tokens
def re_align_en_prep(ch_line: str, ch_instance: List[int]) -> List[int]:
    re_aligned_index = set()
    doc = stanford_parser(' '.join(ch_line.split()))
    index_word = defaultdict() # index_word[word_1] = word_2; word_1 point to word_2
    for line in doc._conll_file.conll_as_string().split('\n'):
        if line:
            line = line.split()
            index_word[line[0]] = line[1]
    for line in doc._conll_file.conll_as_string().split('\n'):
        line = line.split()
        if not line or line[6] == '0': # point to root
            continue
        if int(line[0])-1 in ch_instance and index_word[line[6]] in reserved_words: # instance word point to reserved_words
            re_aligned_index.add(int(line[6])-1)
        elif line[1] in reserved_words and int(line[6])-1 in ch_instance: # reserved_words point to instance word
            re_aligned_index.add(int(line[0])-1)
    return list(re_aligned_index)


# Simplify and exclude Chinese patterns that are not in the annotated data
def get_ch_pat_instance(ch_indices: List[Tuple[int, str]], pos_list: List[str], ch_sent: str, headword_ch_index: List[int]):
    ch_pat = []
    en_pat = []
    # step1: generate initial pattern instance
    # 照理說應該不需要 check i < len(ch_sent.split())? 因為 產生 ch_indices時就檢查過了？
    instance_indices = [i for i, en_tag in ch_indices if i < len(ch_sent.split()) and ch_sent.split()[i] != '了']
    # step2: generate re-align preps (需要先知道哪些是 pattern instance 所以要先做 step1)
    prep_aligned_ch = re_align_en_prep(ch_sent, instance_indices)
    ch_indices += [(i, 'PREP') for i in prep_aligned_ch]
    ch_indices = sorted(ch_indices, key=lambda k: k[0])
    # step3: generate re-align pattern
    for i, en_tag in ch_indices:
        word = ch_sent.split()[i]
        pos = sim_ch_pos(pos_list[i])
        en_pat += [en_tag]
        if word in {'了', '著'} or pos in {'t', 'de'}: # DE: /*的, 之, 得, 地*/
            pass
        elif word in reserved_words: # for Chinese prep -> save word
            ch_pat += [word]
        else:
            if ch_pat and ch_pat[-1] == pos: # continous pos -> only save one
                pass
            else:
                ch_pat += [ pos.upper() if i in headword_ch_index else pos ] # headword uppercase
    ch_pat = sim_ch_pattern(ch_pat, en_pat)
    # step3: generate pattern instance after add re-align Chinese preps
    instance_indices = list(set(instance_indices)|set(prep_aligned_ch))
    instance_indices.sort()
    instance = [ '['+ch_sent.split()[i]+']' if i in headword_ch_index else ch_sent.split()[i] for i in instance_indices]
    return ch_pat, ' '.join(instance)