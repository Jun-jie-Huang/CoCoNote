import copy
import os
import re
import random
import sys
sys.setrecursionlimit(2000)
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import Counter
from pipeline_utils import csv_reader, write_json, read_json, write_jsonl, read_jsonl
from pipeline_utils import tokenize_cell_and_get_token_num, judge_language


def get_code_string_list(cell):
    source = cell.get("source", [])
    source = [] if source is None else source
    if type(source) == str:
        source = source.split('\n')
        source = [s+'\n' if i<len(source)-1 else s for i, s in enumerate(source)]
    return source


def clean_nb_code_to_script(cell):
    code = get_code_string_list(cell)
    code = [s for s in code if len(s)>0]
    code = ['#'+s if s[0]=='%' else s for s in code]
    return ''.join(code)



from io import StringIO
import tokenize
import re
def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)



def cleaning_string(string):
    if string:
        string = re.sub(r'\n[ \n\t]*\n', r'\n', string)  # remove extra \n\n
        string = re.sub("\"", "\'", string)
        return string
    else:
        return ''


from tokenize import tokenize, untokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER
from tokenize import generate_tokens
from io import StringIO

def judge_line_number(source):
    return len([i for i in ('\n'.join(source)).split('\n') if len(i)>0])


def remove_pipeline(nbid):
    dirs = os.path.join(from_path, nbid)
    files = [p for p in os.listdir(dirs) if p.split('.')[0]!=nbid]
    target_vars = [p.split('-')[0].split(',')[1:] for p in files]
    files = [os.path.join(dirs, p) for p in files]

    new_stats = []
    for file in files:
        # TODO change eng to english
        stat = {'file': os.path.basename(file), 'nbid': nbid, 'eng': True, 'len_cc': -1, 'len_md': -1,
                'for_eval': False, 'num_tok_code': -1, 'num_tok_md_word': -1}
        try:
            nb = read_json(file)
            if 'cells' in nb.keys():
                cell_key = 'cells'
            elif 'worksheets' in nb.keys():
                cell_key = 'worksheets'
            else:
                continue
        except Exception as e:
            continue

        # clean the cell strings
        code_cells = nb[cell_key]
        new_cells = []
        for cell in code_cells:
            if "source" in cell:
                new_cell = cell.copy()
                source = cell['source']
                if source is None:
                    source = []
                if type(source) == str:
                    source = source.split('\n')
                    source = [s + '\n' if i < len(source) - 1 else s for i, s in enumerate(source)]
                if len(source) > 0:
                    source = [s for s in source if len(s) > 0]
                    source = ['#' + s if s[0] == '%' else s for s in source]
                new_cell['source'] = source
                new_cells.append(new_cell)
        nb[cell_key] = new_cells
        stat['len_cc'] = len([cell for cell in nb[cell_key] if cell['cell_type']=='code'])
        stat['len_md'] = len(nb[cell_key]) - len([cell for cell in nb[cell_key] if cell['cell_type']=='code'])

        # remove non-english tokens or
        langs = judge_language('\n'.join(['\n'.join(cell['source']) for cell in new_cells]))
        if langs != ['en']:
            stat['eng'] = False

        # judge tokens number
        # code_all = '\n'.join(['\n'.join(cell['source']) for cell in new_cells if cell['cell_type']=='code'])
        # stat['num_tok_code'] = tokenize_cell_and_get_token_num(code_all)
        tokens_md_word = [len(' '.join(cell['source']).split(' ')) for cell in new_cells if cell['cell_type']!='code']
        stat['num_tok_md_word'] = sum(tokens_md_word)

        # # judge line number
        # num_line_code = [judge_line_number(cell['source']) for cell in new_cells if cell['cell_type'] == 'code']
        # num_line_md = [judge_line_number(cell['source']) for cell in new_cells if cell['cell_type'] != 'code']
        # stat['num_line_code'], num_line_md = num_line_code, num_line_md

        # judge whether this case could be used for evaluation
        if stat['eng']:
            if stat['len_cc'] <= 8 and stat['len_md'] <= 8:
                if stat['num_tok_code'] < 1000 and stat['num_tok_md_word'] < 1000:
                    stat['for_eval'] = True
        new_stats.append(stat)

    return new_stats



debug=False
# TODO: change the path
# from_path = sys.argv[1]
from_path = '../CoCoMine-saved_results'
# to_path = sys.argv[2]
to_path = '../CoCoMine-saved_results'
if not os.path.exists(to_path):
    os.makedirs(to_path)

nbids = os.listdir(from_path)
nbids = [p for p in nbids if p[:2]=='NB']
nbids = [p for p in nbids if os.path.isdir(os.path.join(from_path, p))]
print("Processing for {} raw notebooks".format(len(nbids)))

remove_stat = []
for nbid in nbids:
    stat = remove_pipeline(nbid)
    remove_stat.extend(stat)

# n_threads = 8
# with Pool(n_threads) as p:
#     func_ = partial(remove_pipeline,)
#     results = list(tqdm(p.imap(func_, nbids, chunksize=16), total=len(nbids), desc="remove_pipeline", ))
#     remove_stat = [j for i in results for j in i]

write_json(remove_stat, os.path.join(to_path, 'remove_stats.json'))
item_non_english = [item for item in remove_stat if not item['eng']]
print("NBs with non-english token: {}/{}, {:.3f}".format(len(item_non_english), len(remove_stat), len(item_non_english)/len(remove_stat)))

