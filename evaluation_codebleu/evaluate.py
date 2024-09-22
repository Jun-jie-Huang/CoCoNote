import argparse
import bleu
import weighted_ngram_match
import syntax_match
import dataflow_match
import json
import os
from rouge import Rouge
import sys
sys.setrecursionlimit(8735 * 2080 + 10)


def read_json(name):
    with open(name, 'r') as f:
        json_file = json.load(f)
    return json_file


def write_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f)


def read_txt_last(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines[-1]


lang = 'python'
# params = '0.1,0.4,0.1,0.4'
params = '0.25,0.25,0.25,0.25'
# params = '0.1,0.1,0.4,0.4'
alpha, beta, gamma, theta = [float(x) for x in params.split(',')]


import tokenize
from io import StringIO
def tokenize_code(code_string):
    try:
        token_stream = tokenize.generate_tokens(StringIO(code_string).readline)
        tokens = [tokval for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream]
    except:
        tokens = code_string.split(' ')
    tokens = [tok for tok in tokens if tok]
    return tokens


def get_codebleu(all_generations, all_references):

    tokenized_hyps = [x.split(' ') for x in all_generations]
    tokenized_refs = [[x.split(' ')] for x in all_references]
    tokenized_hyps, tokenized_refs = [], []
    for x in all_generations:
        try:
            tok = tokenize_code(x)
        except:
            tok = x.split(' ')
        tokenized_hyps.append(tok)
    for x in all_references:
        try:
            tok = [tokenize_code(x)]
        except:
            tok = [x.split(' ')]
        tokenized_refs.append(tok)

    # calculate ngram match
    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open('./keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]
    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 \
                for token in reference_tokens}
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                    for reference_tokens in reference] for reference in tokenized_refs]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match([[i] for i in all_references], all_generations, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match([[i] for i in all_references], all_generations, lang)

    code_bleu_score = alpha * ngram_match_score \
                      + beta * weighted_ngram_match_score \
                      + gamma * syntax_match_score \
                      + theta * dataflow_match_score

    return {"NgramM": round(ngram_match_score*100, 2),
            "WNgramM": round(weighted_ngram_match_score*100, 2),
            "SynM": round(syntax_match_score*100, 2),
            "DFM": round(dataflow_match_score*100, 2),
            "CodeBLEU": round(code_bleu_score*100, 2),}


def get_metrics(all_generations, all_references):
    all_generations_token = [" ".join(tokenize_code(i.strip())) for i in all_generations]
    all_references_token = [" ".join(tokenize_code(i.strip())) for i in all_references]
    all_scores = {}

    print("Calculating Exact Match...")
    em_list = []
    for i, j in zip(all_generations_token, all_references_token):
        if i == j:
            em_list.append(1)
        else:
            em_list.append(0)
    all_scores['Number'] = len(all_generations)
    all_scores['EM(a)'] = round(sum(em_list) / len(em_list) * 100, 2)

    print("Calculating Rouge...")
    rouge = Rouge()
    scores = rouge.get_scores(all_generations_token, all_references_token, avg=True)
    rouge_scores_list = rouge.get_scores(all_generations_token, all_references_token)
    all_scores['R1-R'] = round(scores['rouge-1']['r']*100, 2)
    all_scores['R1-P'] = round(scores['rouge-1']['p']*100, 2)
    all_scores['R1-F'] = round(scores['rouge-1']['f']*100, 2)
    all_scores['R2-R'] = round(scores['rouge-2']['r']*100, 2)
    all_scores['R2-P'] = round(scores['rouge-2']['p']*100, 2)
    all_scores['R2-F'] = round(scores['rouge-2']['f']*100, 2)
    all_scores['RL-R'] = round(scores['rouge-l']['r']*100, 2)
    all_scores['RL-P'] = round(scores['rouge-l']['p']*100, 2)
    all_scores['RL-F'] = round(scores['rouge-l']['f']*100, 2)

    print("Calculating CodeBLEU...")
    codebleu_scores_list = [get_codebleu([gen], [ref]) for gen, ref in zip(all_generations_token, all_references_token)]
    combine_score = [{"EM": i,
                      "R1-R": j['rouge-1']['r'],
                      "R1-P": j['rouge-1']['p'],
                      "R1-F": j['rouge-1']['f'],
                      "R2-R": j['rouge-2']['r'],
                      "R2-P": j['rouge-2']['p'],
                      "R2-F": j['rouge-2']['f'],
                      "RL-R": j['rouge-l']['r'],
                      "RL-P": j['rouge-l']['p'],
                      "RL-F": j['rouge-l']['f'],
                      "NgramM": k['NgramM'],
                      "WNgramM": k['WNgramM'],
                      "SynM": k['SynM'],
                      "DFM": k['DFM'],
                      "CodeBLEU": k['CodeBLEU'],
                      "reference_text": x,
                      "translation_text": y,
                      }for i, j, k, x, y in zip(em_list, rouge_scores_list, codebleu_scores_list, all_references, all_generations)]
    NgramM = [k['NgramM'] for k in codebleu_scores_list]
    WNgramM = [k['WNgramM'] for k in codebleu_scores_list]
    SynM = [k['SynM'] for k in codebleu_scores_list]
    DFM = [k['DFM'] for k in codebleu_scores_list]
    CodeBLEU = [k['CodeBLEU'] for k in codebleu_scores_list]
    all_scores.update({
        'NgramM': round(sum(NgramM)/len(NgramM), 2),
        'WNgramM': round(sum(WNgramM)/len(WNgramM), 2),
        'SynM': round(sum(SynM)/len(SynM), 2),
        'DFM': round(sum(DFM)/len(DFM), 2),
        'CodeBLEU': round(sum(CodeBLEU)/len(CodeBLEU), 2),
    })
    return all_scores, combine_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_path", default='../evaluation_example/test_1654_gpt35.json', type=str, required=False, help="Path to save the splited generation results.")
    args = parser.parse_args()
    generation_dir = os.path.dirname(args.generation_path)
    result = read_json(args.generation_path)

    all_generations, all_references = [], []
    for item in result:
        if len(item['target_code']) == 0 or len(item['generation']) == 0:
            continue
        all_generations.append(item['generation'])
        all_references.append(item['target_code'])

    print("$$$$$$$$$$$$$$$$$$")
    print("prediction items: {}, target items: {}".format(len(all_generations), len(all_references)))
    print("All data ori: {}".format(len(all_references)))
    results, combine_score = get_metrics(all_references, all_generations)
    print("Results All dev/test original string")
    print("\t".join(results.keys()))
    print("\t".join([str(i) for i in results.values()]))
    write_json([results, combine_score], os.path.join(generation_dir, 'surface_form_score_ori.json'))
