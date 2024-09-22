import json
import csv
import ast
import pandas as pd

dict_file = "stages.csv"
# dc = pd.read_csv(dict_file, index_col=0, squeeze=True, header=None).squeeze("columns")
dc = pd.read_csv(dict_file, index_col=0, header=None).to_dict()


def csv_reader(path):
    with open(path, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp)
        data = [i[0].split('||||') for i in reader]
    return data


def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=1)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def write_jsonl(data, path):
    # data是一个包含多个dict的list
    with open(path, 'w') as fp:
        for inst in data:
            fp.write(json.dumps(inst) + '\n')


def read_jsonl(path):
    # data是一个包含多个dict的list
    data = []
    with open(path, 'r') as fp:
        for line in fp:
            line = line.strip()
            data.append(json.loads(line))
    return data


import pickle
def write_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


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
    # code = ['#'+s if s[0] in {'%', '!', '?'} else s for s in code] # remove magic function
    code = ['#'+s if judge_notebook_magic_code(s) else s for s in code]
    return ''.join(code)


def clean_unparse_code(string):
    if string[0] == '\n':
        string = string[1:]
    if string[-1] != '\n':
        string = string+'\n'
    return string


def judge_notebook_magic_code(code_line):
    return True if code_line[0] in {'%', '!', '?'} else False


import re
def judge_path_url(ori_string):
    # return True is ori_string is a link or a path
    # path
    if re.search(r'[/]+',  ori_string):
        return True
    # http
    if re.search(r'^(http://){0,1}[A-Za-z0-9][A-Za-z0-9\-\.]+[A-Za-z0-9]\.[A-Za-z]{2,}[\43-\176]*$',  ori_string):
        return True
    # ip address
    if re.search(r'^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$',  ori_string):
        return True
    # html
    if re.search(r'<([A-Za-z][A-Za-z0-9]*)\b[^>]*>(.*?)</\1>',  ori_string):
        return True
    return False


class KeyWord:
    keywords_s_dataframe = [
        'DataFrame', 'Series', 'read_csv', 'read_excel', 'read_table', 'read_sql_query',
        'read_hdf', 'read_html', 'read_json', 'read_pickle', 'read_sql', 'get_dummies',
        '.merge', '.concat', '.stack', '.unstack', '.cut', '.qcut', '.melt', '.corr',
        '.crosstab', 'pivot_table', 'query'
    ]
    keywords_s_np_array = ['np.arange', 'np.array', 'np.sum', ]
    keywords_s_model = [
        # sklearn
        'RandomForestRegressor', 'RandomForestClassifier', 'LinearRegression', 'LogisticRegression',
        'MultinomialNB', 'GaussianNB',
        'KNeighborsRegressor', 'KNeighborsClassifier', 'DecisionTreeRegressor', 'DecisionTreeClassifier',
        'PCA', 'TSNE', 'SVC', 'CountVectorizer', 'StandardScaler', 'StratifiedKFold', 'AdaBoostClassifier',
        'GradientBoostingClassifier', 'GradientBoostingRegressor', 'SGDClassifier', 'XGBClassifier', 'LGBMClassifier',
        'BaggingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'MLPRegressor',
    ]
    keywords_s_plot = ['plt.subplots', 'plt.figure', '.subplot', 'plt.axes', 'plt.gca', 'plt.gcf', 'plt.GridSpec()',  ]
    keywords_interesting_start = keywords_s_dataframe

    # from matplotlib, 'plt'
    f_plot_plt = ['.show(', '.imshow(', '.matshow(', '.plot(', 'probplot(',
                  '.barh(', '.bar(', '.errorbar(', '.pie(', '.hist(', '.scatter(',
                  '.semilogy(', '.pcolormesh(', '.hist2d(', '.contour',
    ]
    # from seaborn, 'sns' or 'sb'
    f_plot_sns = ['.distplot(', '.heatmap(', '.countplot(', '.barplot(', '.boxplot(', '.jointplot(', '.clustermap(',
                  '.pairplot(', '.scatterplot(', '.violinplot(', '.stripplot(', '.swarmplot(', '.regplot(', '.rugplot(',
                  '.kdeplot(', '.lmplot(',  '.catplot(', '.relplot(', '.factorplot(', '.lineplot(', '.tsplot(',
                  '.FacetGrid(', '.pointplot(', '.palplot(', '.corrplot(', '.residplot(', '.boxenplot(', '.lvplot(',
    ]
    # from scipy.stats, 'stats'
    f_plot_stat = ['.plot.', '.hist.', '.scatter.',
                   # 'ttest_ind', 'linregress', 'tocsr', 'cdf', 'csr_matrix', 'rvs', 'boxcox'
    ]
    keywords_f_plot = f_plot_plt+f_plot_sns+f_plot_stat

    keywords_f_model = [
        '.fit(', '.fit_transform(', '.fit_generator(',
        '.predict(', '.evaluate(', '.predict_generator(', '.evaluate_generator(',
    ]
    keywords_f_stats = [
        'pearsonr', 'confusion_matrix', 'accuracy_score', 'classification_report', 'cross_val_score', 'roc_curve',
        'roc_auc_score', 'f1_score', 'r2_score', 'mean_squared_error', 'mean_absolute_error', 'cosine_similarity',
        'softmax_cross_entropy_with_logits', 'recall_score', 'precision_score',
    ]
    keywords_f_stats_small = [
        '.std(', '.mean('
    ]
    keywords_f_analyze_data = [
        '.pivot_table(', '.pivot(', '.groupby(', '.summary(',
    ]
    keywords_f_database = [
        '.to_sql(',
    ]
    keywords_final_goal = keywords_f_model+keywords_f_plot+keywords_f_stats


class Utils:
    def get_val(self, node):
        if isinstance(node, ast.Num):
            return str(node.n)
        elif isinstance(node, ast.Str):
            return str(node.s)
        elif isinstance(node, ast.Name):
            return str(node.id)
        elif isinstance(node, ast.NameConstant):
            return str(node.value)
        elif isinstance(node, ast.Call):
            return "CALL"
        elif isinstance(node, ast.Subscript):
            return str(Utils().get_val(node.value))  # + handle subcript Slice(Index, Slice or ExtSlice)
        elif isinstance(node, ast.Attribute):
            FuncLister.trailler = ""
            AttrLister().visit(node)
            return FuncLister.trailler
        elif isinstance(node, ast.List):
            return str(Utils().get_elts(node))
        elif isinstance(node, ast.Tuple):
            return str(Utils().get_elts(node))
        else:
            return "UNKNOWN"

    def get_elts(self, node):
        a = []
        for e in node.elts:
            a.append(Utils().get_val(e))
        return str(a)

    def get_bin_op(self, node):
        if isinstance(node, ast.Add):
            return " + "
        elif isinstance(node, ast.Sub):
            return " - "
        elif isinstance(node, ast.Mult):
            return " * "
        elif isinstance(node, ast.Div):
            return " / "
        elif isinstance(node, ast.FloorDiv):
            return " // "
        elif isinstance(node, ast.Mod):
            return " % "
        elif isinstance(node, ast.Pow):
            return " ** "
        elif isinstance(node, ast.LShift):
            return " << "
        elif isinstance(node, ast.RShift):
            return " >> "
        elif isinstance(node, ast.BitAnd):
            return " B_AND "
        elif isinstance(node, ast.BitOr):
            return " B_OR "
        elif isinstance(node, ast.BitXor):
            return " B_XOR "

    def get_unary_val(self, node):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return -node.operand.value
        elif isinstance(node.op, ast.UAdd) and isinstance(node.operand, ast.Constant):
            return +node.operand.value

    def write(path, content):
        with open(path, 'a') as file:
            file.write(content)

    def flush(path):
        open(path, 'w').close()

    def get_stage(api, fs):
        name = api.split(" [")[0]
        parts = name.split(".")
        root = parts[-1]

        if root in fs:
            s = "9" + root
            return s

        for a in dc:
            if name.endswith(a):
                st = str(dc.get(a))
                if (st == 'None'):
                    print(a)
                return st
        return "0"


import tokenize
from io import StringIO
def tokenize_cell_and_get_token_num(code_string, split_comment=False):
    # code_string = filter_str(code_string)
    ### !!!!! Remember this version contains tokens in the commented comments
    try:
        token_stream = generate_tokens(StringIO(code_string).readline)
        tokens = [tokval for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream]
    except:
        tokens = code_string.split(' ')

    tokens = [tok for tok in tokens if tok]
    if split_comment:
        tokens = [_tok for tok in tokens for _tok in tok.split(' ')]
    return tokens, len(tokens)


import tokenize
# def remove_comments_and_docstrings(source, lang):
def seperate_comment_code(source):
    """
    Returns 'source' and 'comments and docstrings'.
    """
    comments = []
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for token_type, token_string, (start_line, start_col), (end_line, end_col), ltext in tokenize.generate_tokens(io_obj.readline):
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # add comments:
        if token_type == tokenize.COMMENT:
            comments.append(token_string)
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
                        continue
            comments.append(token_string)
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    temp_code = [x for x in out.split('\n') if x.strip()!=""]
    comments = [x for x in comments if x.strip()!=""]
    return '\n'.join(temp_code), comments


import re
import string
# import pycld2 as cld2
remove_nota = u'[’·°–!"#$%&\'\n\t()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def filter_str(sentence):
    sentence = re.sub(remove_nota, ' ', sentence)
    sentence = sentence.translate(remove_punctuation_map)
    return sentence.strip()

# judga language: english, chinese, french, korean, japanese, arabic
def judge_language(s):
    # s = unicode(s)   # python2 should convert string with unicode encoding, but python3 don't need
    s_ori = s
    s = filter_str(s)
    result = []
    s = re.sub('[0-9]', '', s).strip()
    # unicode english
    re_words = re.compile(u"[a-zA-Z]")
    res = re.findall(re_words, s)  # get all matched strings
    res2 = re.sub('[a-zA-Z]', '', s).strip()
    if len(res) > 0:
        result.append('en')
    if len(res2) <= 0:
        return ['en']

    # unicode chinese
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    res = re.findall(re_words, s)  # get all matched strings
    res2 = re.sub(u"[\u4e00-\u9fa5]+", '', s).strip()
    if len(res) > 0:
        result.append('zh')
    if len(res2) <= 0:
        return ['zh']

    # unicode korean
    re_words = re.compile(u"[\uac00-\ud7ff]+")
    res = re.findall(re_words, s)  # get all matched strings
    res2 = re.sub(u"[\uac00-\ud7ff]+", '', s).strip()
    if len(res) > 0:
        result.append('ko')
    if len(res2) <= 0:
        return ['ko']

    # unicode japanese katakana and unicode japanese hiragana
    re_words = re.compile(u"[\u30a0-\u30ff\u3040-\u309f]+")
    res = re.findall(re_words, s)  # get all matched strings
    res2 = re.sub(u"[\u30a0-\u30ff\u3040-\u309f]+", '', s).strip()
    if len(res) > 0:
        result.append('ja')
    if len(res2) <= 0:
        return ['ja']

    # unicode arabic
    re_words = re.compile(u"[\u0600-\u06ff]+")
    res = re.findall(re_words, s)  # get all matched strings
    res2 = re.sub(u"[\u0600-\u06ff]+", '', s).strip()
    if len(res) > 0:
        result.append('ar')
    if len(res2) <= 0:
        return ['ar']
    # try:
    #     isReliable, textBytesFound, details, vectors = cld2.detect(s_ori, returnVectors=True)
    #     for i in vectors:
    #         if i[3] not in result:
    #             result.append(i[3])
    # except:
    #     result = result

    return result


import langid
def judge_language_langid(s):
    # langid.set_languages(['en', 'it', 'fr', 'zh', 'de', 'ja', 'ko', 'ru', 'th', 'hi', 'he', 'cs', 'ar', ])
    langid.set_languages(['en', 'zh', 'ja', 'ko', 'ru', 'th', 'hi', 'he', 'cs', 'ar', ])
    s = filter_str(s)
    result = langid.classify(s)
    if result[0]!='en':
        a=1
    return [result[0]]



