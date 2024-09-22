import os
import string
import re
import sys
sys.setrecursionlimit(2000)
import ast
import pandas as pd
import astunparse
import nbformat as nbf
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import Counter
import sys
from pipeline_utils import read_json, Utils, read_pickle, write_pickle
from pipeline_utils import clean_nb_code_to_script, seperate_comment_code
# from identify_my_pipeline_ok import get_code_cell_to_markdown_idx


"""
pip install nbformat
pip install lxml
pip install html5lib
"""


def depth_ast(root):
    return 1 + max((depth_ast(child)
                       for child in ast.iter_child_nodes(root)),
                   default = 0)


remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
# remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def filter_str(sentence):
    sentence = re.sub(remove_nota, '', sentence)
    # sentence = sentence.translate(remove_punctuation_map)
    return sentence.strip()


def cleaning_string(string):
    if string:
        string = re.sub(r'\n[ \n\t]*\n', r'\n', string)  # remove extra \n\n
        string = re.sub("\"", "\'", string)
        return string
    else:
        return ''


def get_output_items(cell):
    my_outputs = []
    flag = {'dfs2': False, 'other_instance': False, 'exception': ''}
    if cell.get('outputs', []):
        for item in cell['outputs']:
            if item.get('data', {}) and 'text/html' in item['data'] and 'text/plain' in item['data']:
                try:
                    dfs = pd.read_html(item['data']['text/html'])
                    if len(dfs) == 1 and isinstance(dfs[0], pd.DataFrame):
                        my_outputs.append(dfs[0])
                    elif len(dfs) > 1:
                        flag['dfs2'] = True
                    else:
                        flag['other_instance'] = True
                except Exception as e:
                    flag['exception'] = e
                    # print("get output item: {}, {}".format(e, item['data'].keys()))
                    continue
    return my_outputs, flag


def get_input_output_information(file, from_path):
    nbid = file.split(',')[0]
    target_vars = file.split('-')[0].split(',')[1:]
    target_file_id = file.split('-')[1].split('.')[0]
    possible_pipe = {'idx': -1, 'nbid': nbid, 'file': file, 'target_vars': target_vars,
                     'outputs': [], 'outputs_ccidx': [],
                     'other_instance': False, 'dfs2': False, 'except': '', 'less1': False,
                     }
    # if int(target_file_id) > 1:
    #     return possible_pipe

    try:
        # nb = read_json(os.path.join(from_path, nbid, file))
        nb = nbf.read(os.path.join(from_path, nbid, file), nbf.NO_CONVERT)
        if 'cells' in nb.keys():
            cell_key = 'cells'
        elif 'worksheets' in nb.keys():
            cell_key = 'worksheets'
        else:
            return possible_pipe
    except Exception as e:
        possible_pipe['except'] = e
        return possible_pipe

    code_cells = [cell for idx, cell in enumerate(nb[cell_key]) if cell.get('cell_type', '') == 'code']
    outputs = [get_output_items(cell) for cell in code_cells[:-1]] # don't consider final goal cell
    outputs_flag = [item[1] for item in outputs]
    outputs = [item[0] for item in outputs]
    have_output_idx = [idx for idx, item in enumerate(outputs) if item!=[]]
    # only keep pipelines with multiple dataframe output
    if len(have_output_idx) <= 1:
        possible_pipe['less1'] = True
        return possible_pipe
    possible_pipe['outputs'] = [item for item in outputs if item != []]
    possible_pipe['outputs_ccidx'] = have_output_idx


    possible_pipe['other_instance'] = False
    possible_pipe['dfs2'] = False
    possible_pipe['except'] = []
    if any([item['other_instance'] for item in outputs_flag]):
        possible_pipe['other_instance'] = True
    if any([item['dfs2'] for item in outputs_flag]):
        possible_pipe['dfs2'] = True
    if any([item['exception']!='' for item in outputs_flag]):
        possible_pipe['except'] = [item['exception'] for item in outputs_flag if item['exception']!='']
    # code_cells = [clean_nb_code_to_script(cell) for idx, cell in enumerate(nb[cell_key]) if cell.get('cell_type', '') == 'code']
    return possible_pipe


def get_possible_pipe(nbid, from_path):
    possible_pipes = []
    files = [p for p in os.listdir(os.path.join(from_path, nbid)) if p.endswith('.ipynb')]
    files = [p for p in files if len(p.split('.')[0])!=len(nbid)]  # remove original notebook
    files = [p for p in files if int(p.split('-')[1].split('.')[0])<2]  # only keep one pipeline for variables with multiple pipelines
    for file in files:
        _possible_pipe = get_input_output_information(file, from_path)
        # if _possible_pipe['outputs'] != []:
        #     possible_pipes.append(_possible_pipe)
        possible_pipes.append(_possible_pipe)
    return possible_pipes


class IOVariableLister(ast.NodeVisitor):

    def __init__(self, possible_vars):
        super(IOVariableLister, self).__init__()
        self.possible_vars = possible_vars
        self.methods = []
        self.attrs = []
        self.values = []
        self.codes = []
        self.num_code_df_row = 0
        self.num_statements = 0

    def visit_FunctionDef(self, node):
        pass
    def visit_ClassDef(self, node):
        pass
    def visit_Assign(self, node):
        pass
    def visit_Import(self, node):
        pass
    def visit_ImportFrom(self, node):
        pass
    def visit_For(self, node):
        pass
    def visit_Expr(self, node):
        # value_node = ValueLister()
        # value_node.visit(node.value)
        # value_names = sorted(set(value_node.value_names), key=value_node.value_names.index)
        if isinstance(node.value, ast.Name):
            self.methods.append(['NAME'])
            self.attrs.append(['NAME'])
            self.values.append([node.value.id])
            self.num_code_df_row = 111
        else:
            method_node = ExprMethodLister(self.possible_vars)
            method_node.visit(node.value)
            self.methods.append(method_node.methods)
            self.attrs.append(method_node.attrs)
            self.values.append(method_node.values)
            self.num_code_df_row = method_node.row if method_node.row > 0 else self.num_code_df_row
        self.codes.append(astunparse.unparse(node))


class ExprMethodLister(ast.NodeVisitor):
    def __init__(self, possible_vars):
        self.possible_vars = possible_vars
        self.method_this = ""
        self.methods = []  # APIs
        self.value_this = ""
        self.attr_this = ""
        self.values = []
        self.attrs = []
        self.row = 0

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Attribute):
            self.method_this = "."+node.attr+"()" if self.method_this == "" else "."+node.attr+"()."+self.method_this
        elif isinstance(node.value, ast.Name):
            self.method_this = node.value.id+"."+node.attr+"()" if self.method_this == "" else node.value.id+"."+node.attr+"()"+self.method_this
            # if node.value.id in self.possible_vars: ## TODO un-comment this line
            self.value_this = node.value.id
            self.attr_this = node.attr
            if self.attr_this in POSSIBLE_DF_PRINT_API:
                try:
                    father = node.parent
                    self.row = 5
                    if len(father.args) == 1:
                        if isinstance(father.args[0], ast.Constant):
                            self.row = father.args[0].value
                    if len(father.keywords) == 1:
                        if isinstance(father.keywords[0], ast.keyword):
                            if father.keywords[0].arg=='n' and isinstance(father.keywords[0].value, ast.Constant):
                                self.row = father.keywords[0].value.value
                except:
                    pass

        elif isinstance(node.value, ast.Call):
            self.method_this = "."+node.attr+"()" if self.method_this == "" else "."+node.attr+"()"+self.method_this
        else:
            self.method_this = "."+node.attr+"()"
        self.generic_visit(node)

    def visit_Name(self, node):
        # self.method_this = node.id if self.method_this == "" else node.id+"."+self.method_this
        if self.method_this == "":
            self.method_this = node.id
        # self.generic_visit(node)
        if len(self.method_this) > 0:
            self.methods.append(self.method_this)
            self.values.append(self.value_this)
            self.attrs.append(self.attr_this)
            self.method_this = ""
            self.value_this = ""
            self.attr_this = ""

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.visit_Name(node.func)
        elif isinstance(node.func, ast.Attribute):
            self.visit_Attribute(node.func)
        else:
            self.generic_visit(node.func)
        for arg_node in node.args:
            if isinstance(arg_node, ast.Name):
                self.visit_Name(arg_node)
            else:
                self.generic_visit(arg_node)

    def visit_Subscript(self, node):
        self.generic_visit(node.value)
    def visit_IfExp(self, node):
        self.generic_visit(node.body)


def judge_api(api_in, api_out):
    # flag = False
    # if api_in == api_out:
    #     flag = True
    # elif [api_in, api_out] == ['head', 'NAME'] or [api_in, api_out] == ['NAME', 'head']:
    #     flag = True
    flag = True
    if [api_in, api_out] == ['head', 'tail'] or [api_in, api_out] == ['tail', 'head']:
        flag = False
    return flag


def remove_abbr_row(df):
    # judge whether contains abbreviation mark ... in two dfs
    abbr_count = (df == '...').sum().sum()
    if abbr_count == df.shape[1]:
        temp_id = df.shape[0]//2
        df = df.drop(axis=0, index=[temp_id])
    elif abbr_count == df.shape[0]:
        temp_id = df.shape[1]//2
        df = df.drop(df.columns[temp_id], axis=1)
        # df = df.drop(axis=1, columns=['...'])
    elif abbr_count == df.shape[0]+df.shape[1]-1:
        temp_id = df.shape[0]//2
        df = df.drop(axis=0, index=[temp_id])
        temp_id = df.shape[1]//2
        df = df.drop(df.columns[temp_id], axis=1)
    elif abbr_count != 0:
        df = df
    return df


def normalize_input_output_dataframe(df_in, df_out, api_in, api_out):
    shape_in, shape_out = df_in.shape, df_out.shape
    if api_in == api_out:
        df_in = remove_abbr_row(df_in)
        df_out = remove_abbr_row(df_out)
        # if shape_in[0] == shape_out[0]:
        #     df_in = remove_abbr_row(df_in)
        #     df_out = remove_abbr_row(df_out)
        # else:
        #     df_in = remove_abbr_row(df_in)
        #     df_out = remove_abbr_row(df_out)
        if shape_in[0] != shape_out[0]:
            row_min = min(df_in.shape[0], df_out.shape[0])
            if api_in == 'head':
                df_in, df_out = df_in[:row_min], df_out[:row_min]
            elif api_in == 'tail':
                df_in, df_out = df_in[-row_min:], df_out[-row_min:]
            else:
                df_in, df_out = df_in[:row_min], df_out[:row_min]
    else:
        df_in = remove_abbr_row(df_in)
        df_out = remove_abbr_row(df_out)
        # judge whether index consistent
        row_min = min(df_in.shape[0], df_out.shape[0])
        if 'head' in [api_in, api_out]:
            df_in, df_out = df_in[:row_min], df_out[:row_min]
        elif api_in == 'tail':
            df_in, df_out = df_in[-row_min:], df_out[-row_min:]
        else:
            df_in, df_out = df_in[:row_min], df_out[:row_min]
    return df_in, df_out


POSSIBLE_DF_PRINT_API = ['head', 'tail', 'NAME']
IMPOSSIBLE_EXPR_API = {'chdir', 'isnull', 'drop', 'info', 'reset_index', 'replace', 'drop_duplicates', 'rename', 'where', 'Period', 'describe', 'isna', 'columns', 'transpose', 'sample', 'sort_values'}
IMPOSSIBLE_EXPR_NAME = ['print', 'plt', 'np']
def get_qualified_task(task_item, nb_dir):
    # # Step 2: Normalize data.
    # # (check same variable, check dataframe dimensions, >=3 execution output examples, identify code complexity)
    # # obtain context (previous cells, and middle md/cmt NL, and code/import)
    # # This step is to obtain qualified tasks that can be used in TESTING!!!
    nb_path = os.path.join(nb_dir, task_item['nbid'], task_item['file'])

    try:
        # nb = read_json(nb_path)
        nb = nbf.read(nb_path, nbf.NO_CONVERT)
        if 'cells' in nb.keys():
            cell_key = 'cells'
        elif 'worksheets' in nb.keys():
            cell_key = 'worksheets'
        else:
            return {}
    except Exception as e:
        return {}

    code_cells = [cell for idx, cell in enumerate(nb[cell_key]) if cell.get('cell_type', '') == 'code']
    ccidx2pcidx = dict(zip(range(len(code_cells)), [idx for idx, cell in enumerate(nb[cell_key]) if cell.get('cell_type', '') == 'code']))
    code_string_all = [clean_nb_code_to_script(cell) for ccidx, cell in enumerate(code_cells)]
    code_length_all = []
    for idx, cell in enumerate(code_string_all):
        try:
            tree = ast.parse(cell)
            code_length_all.append(len(tree.body))
        except Exception as e:
            code_length_all.append(0)

    # idx_global_markdown, idx_code2markdown = get_code_cell_to_markdown_idx(nb[cell_key], )
    # cells_string = [[cidx, clean_nb_code_to_script(cell)] for cidx, cell in enumerate(nb[cell_key]) if cell.get('cell_type', '')=='code']

    cells_string = [clean_nb_code_to_script(cell) for ccidx, cell in enumerate(code_cells) if ccidx in task_item['outputs_ccidx']]
    code_cell_info = []
    for idx, (ccidx, cell) in enumerate(zip(task_item['outputs_ccidx'], cells_string)):
        info = {'idx': ccidx, 'nbid': task_item['nbid'], 'file': task_item['file'],
                'vars': [], 'var_ok': False, 'apis': [], 'api_ok': False, 'target_vars': task_item['target_vars'],
                'real_var': "", 'real_api': "", 'n_stmt': 0, 'stmt_pos': 0,
                'out_ok': False, 'df_dim': (0, 0), 'c_df_row': 0, 'out': pd.DataFrame(), 'test':False}

        # judge which variable yield df output, considering the feature of JupyNB that expresion yields dataframe
        try:
            tree = ast.parse(cell)
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node
            # astpretty.pprint(tree, show_offsets=False)
            visitor_s1 = IOVariableLister(possible_vars=task_item['target_vars'])
            visitor_s1.visit(tree)
        except Exception as e:
            if debug:
                print("parsing fail: {}, {}, in ast {}/{}".format(e, nb_path, idx, len(cells_string)))
            # return {}
            continue

        expr_idxs = [_idx for _idx, stmt in enumerate(tree.body) if isinstance(stmt, ast.Expr)]
        idx_expr2stmt = {expr_i:stmt_i for expr_i, stmt_i in enumerate(expr_idxs)}
        info['n_stmt'] = len(tree.body)
        info['vars'] = visitor_s1.values
        info['apis'] = visitor_s1.attrs
        info['stmts'] = visitor_s1.codes
        info['c_df_row'] = visitor_s1.num_code_df_row
        for _expr_idx, (var_per_expr, api_per_expr) in enumerate(zip(info['vars'], info['apis'])):
            if len(var_per_expr) == 1:
                if var_per_expr[0] in task_item['target_vars']:
                    info['var_ok'] = True
                    info['real_var'] = var_per_expr[0]
            # if len(api_per_expr) == 1:
            #     if api_per_expr[0] in POSSIBLE_DF_PRINT_API:
            #         info['api_ok'] = True
            #         info['real_api'] = api_per_expr[0]
            for _api in api_per_expr:
                if _api in POSSIBLE_DF_PRINT_API:
                    info['api_ok'] = True
                    info['real_api'] = _api
            # if info['var_ok'] and info['api_ok']:
            #     info['stmt_pos'] = idx_expr2stmt[_expr_idx]

        # judge input output
        output = task_item['outputs'][idx]
        if len(output) == 1:
            info['out_ok'] = True
            info['out'] = output[0]
            info['df_dim'] = output[0].shape
            # if info['c_df_row'] == 111:
            # if info['c_df_row'] == 111 and info['df_dim'][0] <= 60:
            #     info['c_df_row'] = info['df_dim'][0]
            if info['c_df_row'] != info['df_dim'][0]:
                info['test'] = True
        code_cell_info.append(info)


    dont_judge =[]
    tasks = []
    qualified_code_cells = [info for info in code_cell_info if info['out_ok'] and info['api_ok'] and info['var_ok']]
    if len(qualified_code_cells) <= 1:
        return {'info': code_cell_info, 'tasks': tasks, 'dont_judge': dont_judge}
    # # continuous cells
    # paired_input_output_info = [[qualified_code_cells[i], qualified_code_cells[i+1]] for i in range(len(qualified_code_cells)-1)]
    # all previous-later cells
    paired_input_output_info = [[qualified_code_cells[i], qualified_code_cells[i+gap]]
                                for gap in range(1, len(qualified_code_cells)) for i in range(len(qualified_code_cells)-gap)]

    for input_info, output_info in paired_input_output_info:
        io_task = {'nbid': task_item['nbid'], 'file': task_item['file'], 'lang': [],
                   'context': [], 'target_code': "", 'target_n_stmt': 0,
                   'in_n_stmt': [], 'in_stmt_pos': input_info['stmt_pos'],
                   'in_df_dim': input_info['df_dim'], 'in_df': input_info['out'], 'in_api': input_info['real_api'],
                   'out_n_stmt': [], 'out_stmt_pos': output_info['stmt_pos'],
                   'out_df_dim': output_info['df_dim'], 'out_df': output_info['out'], 'out_api': output_info['real_api'],
                   }

        if judge_api(io_task['in_api'], io_task['out_api']):
            io_task['in_df'], io_task['out_df'] = normalize_input_output_dataframe(io_task['in_df'], io_task['out_df'], io_task['in_api'], io_task['out_api'])
            io_task['in_df_dim'], io_task['out_df_dim'] = io_task['in_df'].shape, io_task['out_df'].shape

            # tgt
            target_code_cell_idx = [idx for idx, cell in enumerate(code_string_all) if input_info['idx']<idx<=output_info['idx']]
            target_codes = [cell['source'] for idx, cell in enumerate(code_cells) if idx in target_code_cell_idx]
            try:
                target_code, target_comments = seperate_comment_code('\n'.join(target_codes))
            except:
                continue
            io_task['target_code'] = target_code
            # num_target_statements = [info['n_stmt'] for info in qualified_code_cells if input_info['idx']<info['idx']<=output_info['idx']]
            num_target_statements = [code_length_all[ccidx] for ccidx in target_code_cell_idx]
            io_task['target_n_stmt'] = sum(num_target_statements)
            io_task['out_n_stmt'] = num_target_statements

            # context
            previous_cells = []
            # TODO add import and assign code
            if add_import_context:
                previous_cells += [['import', ''.join([_import_ctx[3] for _import_ctx in fn2ctx[task_item['file']][1]])]]
                previous_cells += [['data_depen', ''.join([_code_ctx[1] for _code_ctx in fn2ctx[task_item['file']][0]])]]
                io_task['lang'] = fn2ctx[task_item['file']][2]
            # previous code and markdown
            previous_cells += [[cell.get('cell_type', ''), cell['source']] for idx, cell in enumerate(nb[cell_key])
                               if idx<=ccidx2pcidx[input_info['idx']]]
            # plus middle md cells
            previous_cells += [['markdown_mid', cell['source']] for idx, cell in enumerate(nb[cell_key])
                               if cell.get('cell_type', '') == 'markdown' and ccidx2pcidx[input_info['idx']] < idx <= ccidx2pcidx[output_info['idx']]]
            # comments in tgt code cells
            previous_cells.append(['comment_in_tgt', '\n'.join(target_comments)])
            io_task['context'] = previous_cells
            io_task['in_n_stmt'] = [info['n_stmt'] for info in qualified_code_cells if info['idx']<=input_info['idx']]


            # whether to add this example to task:
            DONT_ADD = False
            # Situation 1: input dataframe is the same as output dataframe
            if io_task['in_df'].shape == io_task['out_df'].shape:
                # if assert_frame_equal(io_task['in_df'], io_task['out_df']):
                if io_task['in_df'].equals(io_task['out_df']):
                    print("data frame same: {} == {}, in:{}, out:{}".format(io_task['in_df'].shape, io_task['out_df'].shape,
                                                                            io_task['in_df'], io_task['out_df']))
                    DONT_ADD = True
            if io_task['target_n_stmt']>11 or io_task['target_n_stmt']==0:
                DONT_ADD = True
            if not DONT_ADD:
                tasks.append(io_task)
            dont_judge.append(1)

    return {'info': code_cell_info, 'tasks': tasks, 'dont_judge': dont_judge}


if __name__ == '__main__':
    from_path = sys.argv[1]
    # from_path = './saved_wrangling_cells'

    to_path = sys.argv[2]
    # to_path = '../dataset-CSIO'
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    flag_judge_possible_pipelines = True
    flag_normalize_data = True
    debug=False
    add_import_context=True

    # Step 1: judge possible pipelines
    # (pipeline with >=2 df input-output, code context, code target, qualified execution outputs)
    if flag_judge_possible_pipelines:
        if not os.path.exists(to_path):
            os.makedirs(to_path)
        dirs = os.listdir(from_path)
        dirs = [i for i in dirs if i[:2]=='NB']
        print(len(dirs))

        if len(dirs) > 1000:
            # n_threads = 8
            n_threads = 2
            with Pool(n_threads) as p:
                func_ = partial(get_possible_pipe, from_path=from_path)
                possible_tasks = list(tqdm(p.imap(func_, dirs, chunksize=16), total=len(dirs), desc="create CS with IO"))
                possible_tasks = [j for i in possible_tasks for j in i]
        else:
            possible_tasks = []
            for nbid in dirs:
                print(nbid)
                possible_pipes = get_possible_pipe(nbid, from_path=from_path)
                possible_tasks.extend(possible_pipes)

        write_pickle(possible_tasks, os.path.join(to_path, 'synthesis_input_output.pkl'))
        print("number of tasks: {}".format(len(possible_tasks)))
        print("number of qualified tasks: {}".format(len([i for i in possible_tasks if len(i['outputs'])>0])))

    # Step 2: Normalize data.
    # (check same variable, check dataframe dimensions, >=3 execution output examples, identify code complexity)
    if flag_normalize_data:
        possible_tasks = pd.read_pickle(f'{to_path}/synthesis_input_output.pkl')
        outputs_tasks = [item for item in possible_tasks if len(item['outputs'])>0]
        outputs_tasks = sorted(outputs_tasks, key=lambda x:x['file'])
        print(len(possible_tasks), len(outputs_tasks))

        if add_import_context:
            statistics = read_json(os.path.join(from_path, 'statistics.json'))
            global fn2ctx
            fn2ctx = {}
            for nbid, item in statistics.items():
                dedup_paths = item['dedup_path']
                file_idx = Counter()
                for idx, item in enumerate(dedup_paths):
                    _name_vars = [_i[0] for _i in item['target_names']]
                    _name_vars = ','.join(sorted(set(_name_vars), key=_name_vars.index))[:130]
                    _name_vars = re.sub(r'[^\x00-\x7F]+', '', _name_vars)
                    file_idx[_name_vars] += 1
                    new_file = nbid + ',' + _name_vars + '-{}.ipynb'.format(file_idx[_name_vars])
                    if new_file[:3] != "NB_":
                        new_file = 'NB_' + new_file
                    fn2ctx[new_file] = [item['code_var_context'], item['import_context'], item['lang']]
            print("fn2ctx: {}".format(len(fn2ctx)))

        if len(outputs_tasks) > 1000:
            n_threads = 4
            with Pool(n_threads) as p:
                func_ = partial(get_qualified_task, nb_dir=from_path)
                results = list(tqdm(p.imap(func_, outputs_tasks, chunksize=16), total=len(outputs_tasks), desc="get qualified CSIO task split-{}".format(i)))
                all_tasks_info = [item['info'] for item in results]
                all_qualified_tasks = [item['tasks'] for item in results]
                dont_judge_tasks = [item['dont_judge'] for item in results]
        else:
            all_tasks_info = []
            all_qualified_tasks = []
            dont_judge_tasks = []
            for item in outputs_tasks:
                qualified_tasks = get_qualified_task(item, from_path)
                all_tasks_info.append(qualified_tasks['info'])
                all_qualified_tasks.append(qualified_tasks['tasks'])
                dont_judge_tasks.append(qualified_tasks['dont_judge'])

        tasks_temp = [pipe for nb in all_qualified_tasks for pipe in nb]
        dont_judge_tasks_temp = [pipe for nb in dont_judge_tasks for pipe in nb]
        temp = [pipe for nb in all_tasks_info for pipe in nb]
        # temp_ok = [pipe for nb in all_tasks_info for pipe in nb if pipe['var_ok'] and pipe['api_ok']]
        temp_ok_three = [pipe for nb in all_tasks_info for pipe in nb if pipe['var_ok'] and pipe['api_ok'] and pipe['out_ok']]
        # temp_ok_test = [pipe for pipe in temp_ok_three if pipe['test']]
        not_ok_vars = [[pipe['target_vars'], pipe['vars'], pipe['real_var'], pipe['stmts']] for pipe in temp if not pipe['var_ok']]
        not_ok_apis = [[pipe['apis'], pipe['real_api'], pipe['stmts']] for pipe in temp if not pipe['api_ok']]
        not_ok_outs = [[pipe['out']] for pipe in temp if not pipe['out_ok']]
        print("all qualified tasks: {} from {} pipelines ".format(len(tasks_temp), len(outputs_tasks)))
        print("all_tasks_info: {}, task with 'api_ok', 'var_ok', and 'out_ok': {}".format(len(temp), len(temp_ok_three)))
        print("not ok items: not_ok_vars: {}, not_ok_apis: {}, not_ok_outs: {}, ".format(len(not_ok_vars), len(not_ok_apis), len(not_ok_outs)))
        if not os.path.exists(os.path.join(from_path, 'temp')):
            os.makedirs(os.path.join(from_path, 'temp'))
        write_pickle(all_tasks_info, os.path.join(from_path, 'temp', 'task_data_dependency_info.pkl'))
        write_pickle(not_ok_vars, os.path.join(from_path, 'temp', 'not_ok_vars.pkl'))
        write_pickle(not_ok_apis, os.path.join(from_path, 'temp', 'not_ok_apis.pkl'))
        write_pickle(not_ok_outs, os.path.join(from_path, 'temp', 'not_ok_outs.pkl'))

        write_pickle(tasks_temp, os.path.join(to_path, 'task_data_dependency_qualified.pkl'))
        # write_pickle(all_qualified_tasks, os.path.join(to_path, 'task_data_dependency_qualified.pkl'))
        # write_pickle(dont_judge_tasks_temp, os.path.join(to_path, 'task_temp_dont_judge.pkl'))










