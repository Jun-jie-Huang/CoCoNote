import copy
import os
import re
import sys
sys.setrecursionlimit(1000000)
import ast
import astunparse
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import Counter
from pipeline_utils import write_json, read_json
from pipeline_utils import KeyWord
from pipeline_utils import tokenize_cell_and_get_token_num, clean_nb_code_to_script, clean_unparse_code
from pipeline_utils import judge_language_langid, judge_language


def get_source_string(cell):
    source = cell.get("source", "")
    source = "".join(source) if type(source) == list else source
    source = "" if source is None else source
    if type(source) != str:
        source = ""
    return source


def resplit_code_cells(cells):
    return cells


def judge_blank_cell(cell):
    if len(get_source_string(cell)) == 0:
        return True
    else:
        return False


def get_code_cell_to_markdown_idx(cells, ):
    global_markdown = []
    code2markdown = {}
    type_idx = []
    for idx, cell in enumerate(cells):
        source = get_source_string(cell)
        if len(source) > 0:
            if cell.get('cell_type', '')=='code':
                type_idx.append([idx, 'c'])
            else:
                type_idx.append([idx, 'm'])
    md_idxs = []
    first = False
    for idx, ty in type_idx:
        if not first:
            if ty == 'm':
                global_markdown.append(idx)
            else:
                first = True
                code2markdown[idx] = []
        else:
            if ty == 'c':
                code2markdown[idx] = md_idxs
                md_idxs = []
            else:
                md_idxs.append(idx)
    return global_markdown, code2markdown


class S1DefinitionAssignLister(ast.NodeVisitor):

    def __init__(self):
        super(S1DefinitionAssignLister, self).__init__()
        self.value_names = []
        self.target_names = []  # definition
        self.target_define_line = []  # definition code
        self.variable_start_apis = []
        self.usage_names = []  # usage variable name
        self.usage_apis = []  # use variable name with what api
        self.import_pkg = []
        self.import_module = []
        self.import_asname = []
        self.import_code = []
        self.block_target = self.import_asname
        self.isFunc = 0
        self.num_statements = 0
        self.token_num = 0

    def visit_FunctionDef(self, node):
        self_function_history.add(node.name)
        self.isFunc = 1
        pass
    def visit_ClassDef(self, node):
        self_function_history.add(node.name)
        for body in node.body:
            if isinstance(body, ast.FunctionDef):
                self.visit_FunctionDef(body)
        self.isFunc = 1
        pass

    def visit_Assign(self, node):
        add_new_target = False
        for target in node.targets[:1]:
            target_node = TargetLister(block_target=self.block_target)
            target_node.visit(target)
            if len(target_node.target_names) > 0:
                add_new_target = True
                self.target_names.append(target_node.target_names)
                code = astunparse.unparse(node)
                self.target_define_line.append(clean_unparse_code(code))

        value_node = ValueLister()
        value_node.visit(node.value)
        value_names = sorted(set(value_node.value_names), key=value_node.value_names.index)
        method_node = MethodLister()
        method_node.visit(node.value)
        if add_new_target and self.isFunc == 0:
            self.value_names.append(value_names)
            self.variable_start_apis.append(method_node.methods)
            self.usage_names.append(value_names)
            self.usage_apis.append(method_node.methods)

    def visit_Expr(self, node):
        value_node = ValueLister()
        value_node.visit(node.value)
        value_names = sorted(set(value_node.value_names), key=value_node.value_names.index)
        method_node = MethodLister()
        method_node.visit(node.value)
        if self.isFunc == 0:
            self.usage_names.append(value_names)
            self.usage_apis.append(method_node.methods)

    def visit_Import(self, node):
        for name_node in node.names:
            self.import_pkg.append(name_node.name.split('.')[0] if name_node.name else '')
            self.import_module.append(name_node.name if name_node.name else '')
            self.import_asname.append(name_node.asname if name_node.asname else name_node.name)
            self.import_code.append(clean_unparse_code(astunparse.unparse(node)))

    def visit_ImportFrom(self, node):
        for name_node in node.names:
            self.import_pkg.append(node.module.split('.')[0] if node.module else '')
            self.import_module.append(node.module if node.module else '')
            self.import_asname.append(name_node.asname if name_node.asname else name_node.name)
            self.import_code.append(clean_unparse_code(astunparse.unparse(node)))


class ValueLister(ast.NodeVisitor):
    def __init__(self):
        self.value_names = []

    def visit_Name(self, node):
        if node.id in variable_history:
            self.value_names.append(node.id)
        else:
            self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node.func)
        for arg_node in node.args:
            if isinstance(arg_node, ast.Name):
                self.visit_Name(arg_node)
            else:
                self.generic_visit(arg_node)
        for key_node in node.keywords:
            if isinstance(key_node.value, ast.Name):
                self.visit_Name(key_node.value)
            else:
                self.generic_visit(key_node.value)

    # def visit_Constant(self, node):
    # def visit_Subscript(self, node):
    #     self.generic_visit(node.value)

    def visit_Tuple(self, node):
        for e in node.elts:
            if isinstance(e, ast.Name):
                self.visit_Name(e)
            else:
                self.generic_visit(e)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in variable_history:
            self.value_names.append(node.value.id)
        else:
            self.generic_visit(node)



class TargetLister(ast.NodeVisitor):
    def __init__(self, block_target=None):
        if block_target is None:
            block_target = []
        self.target_names = []
        self.block_target = block_target

    def visit_Name(self, node):
        # if node.id not in self.block_target:
        self.target_names.append(node.id)
        variable_history.add(node.id)

    def visit_Tuple(self, node):
        for e in node.elts:
            if isinstance(e, ast.Name):
                self.visit_Name(e)
            else:
                self.generic_visit(e)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Attribute):
            self.generic_visit(node.value)
        elif isinstance(node.value, ast.Name):
            self.generic_visit(node)
        else:
            # print("visit target subscript: ", type(node.value))
            self.generic_visit(node.value)

    def visit_Attribute(self, node):
        self.generic_visit(node)


class MethodLister(ast.NodeVisitor):
    def __init__(self):
        self.method_this = ""
        self.methods = []

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Attribute):
            # self.method_this = node.attr+"()" if self.method_this == "" else node.attr+"()."+self.method_this
            self.method_this = "."+node.attr+"()" if self.method_this == "" else "."+node.attr+"()."+self.method_this
        elif isinstance(node.value, ast.Name):
            # self.method_this = node.value.id+"."+node.attr+"()" if self.method_this == "" else node.value.id+"."+node.attr+"()."+self.method_this
            self.method_this = node.value.id+"."+node.attr+"()" if self.method_this == "" else node.value.id+"."+node.attr+"()"+self.method_this
        elif isinstance(node.value, ast.Call):
            # self.method_this = node.attr+"()" if self.method_this == "" else node.attr+"()."+self.method_this
            self.method_this = "."+node.attr+"()" if self.method_this == "" else "."+node.attr+"()"+self.method_this
        else:
            self.method_this = "."+node.attr+"()"
        self.generic_visit(node)

    def visit_Name(self, node):
        # self.method_this = node.id if self.method_this == "" else node.id+"."+self.method_this
        if self.method_this == "" and node.id not in variable_history:
            self.method_this = node.id
        # self.generic_visit(node)
        if len(self.method_this) > 0:
            self.methods.append(self.method_this)
            self.method_this = ""

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
        # elif isinstance(node.value, ast.IfExp) and isinstance(node.value.body, ast.Call):
        #     method_node.visit(node.value.body.func)
        # elif isinstance(node.value, ast.Subscript):
        #     method_node.visit(node.value.value)


def tranverse_path_from_start(start_item, cells):
    if len(start_item['target_names'])==0 or start_item['target_names'][0]=='':
        return []
    ccidx2cellsidx = {item['ccidx']:cellsidx for cellsidx, item in enumerate(cells)}

    start_var = start_item['target_names'][0]
    path_idx_var_not_derive = [start_item['ccidx']]
    path_reassign = [False]*ccidx2cellsidx[start_item['ccidx']] + [True]
    # path_idx_var_not_derive: only count cells which use start_var
    for cell in cells[ccidx2cellsidx[start_item['ccidx']]+1:]:
        add_reassign = False
        for idx_line, usage_names_per_line in enumerate(cell['usage_names']):
            if start_var in usage_names_per_line:
                path_idx_var_not_derive.append(cell['ccidx'])
        for idx_line, value_names_per_line in enumerate(cell['value_names']):
            if start_var in value_names_per_line:
                # add_reassign = True
                if start_var in cell['target_names'][idx_line]:
                    add_reassign = True
        path_reassign.append(add_reassign)

    path_idx_var_not_derive = sorted(set(path_idx_var_not_derive), key=path_idx_var_not_derive.index)
    return path_idx_var_not_derive, path_reassign


def tranverse_path_find_end(start_item, pipeline, reassign, cells, final_goals):
    if len(pipeline)==0:
        return []
    cellsidx2ccidx = {cellsidx:item['ccidx'] for cellsidx, item in enumerate(cells)}
    reassign_idx = [cellsidx2ccidx[_i] for _i, f in enumerate(reassign) if f]
    ccidx2numstatments = {item['ccidx']:item['n_stmt'] for item in cells}
    ccidx2tokennum = {item['ccidx']:item['n_tok'] for item in cells}

    # # raw final goal cc idx
    final_goal_ccidx = [item['ccidx'] for item in final_goals]
    # (Higher precision but Less number) final goal cc idx only for start_item's target_names
    # final_goal_ccidx = [item['ccidx'] for item in final_goals if len(set(item['usage_names']).intersection(set(start_item['target_names'])))>0 ]
    final_goal_ccidx = sorted(set(final_goal_ccidx), key=final_goal_ccidx.index)
    final_goal_this_pipe = [-1] + [i for i in pipeline if i in final_goal_ccidx]
    final_pipeline = []
    final_num_statements = []
    final_token_num = []
    # if len(final_goal_this_pipe) > 1:
    #     for pipe in range(len(final_goal_this_pipe)-1):
    #         # ## old version: [3,5,6,7,8,9,10,12] -> [[3,5,6], [7,8,9,10], [12]]
    #         # final_pipeline.append([ccidx for ccidx in pipeline if final_goal_this_pipe[pipe]<ccidx and ccidx<=final_goal_this_pipe[pipe+1]])
    #         ## new version: [3,5,6,7,8,9,10,12] -> [[3,5,6], [3,5,6,7,8,9,10], [3,5,6,7,8,9,10,12]]
    #         final_pipeline.append([ccidx for ccidx in pipeline if ccidx<=final_goal_this_pipe[pipe+1]])
    if len(final_goal_this_pipe) > 1:
        for pipe in range(len(final_goal_this_pipe)-1):
            ## new version: reassign [3,5,8,10,12], goal [5, 9, 13]-> [[3,5], [3,5,8,9], [3,5,8,10,12,13]]
            ## and we require former cells to rather be re-assigned or used for final goal
            this_pipeline = [i for i in reassign_idx if i<final_goal_this_pipe[pipe+1]] + [final_goal_this_pipe[pipe+1]]
            final_pipeline.append(this_pipeline)

            # get number of statements for each pipeline:
            final_num_statements.append([ccidx2numstatments[ccidx] for ccidx in this_pipeline])
            final_token_num.append([ccidx2tokennum[ccidx] for ccidx in this_pipeline])
    final_stat = {'final_num_statements': final_num_statements, 'final_token_num': final_token_num,}
    return final_pipeline, final_stat


def get_target_to_definition(cells):
    target_to_definition = []
    for cell in cells:
        try:
            t2d = {target_name: cell['target_define_line'][idx] for idx, target_names in enumerate(cell['target_names']) for target_name in target_names}
        except:
            t2d = {}
        target_to_definition.append(t2d)
    return target_to_definition


def get_context_code(pipeline, cells, target_to_definition):
    ccidx2cellsidx = {item['ccidx']:cellsidx for cellsidx, item in enumerate(cells)}
    target_names = [_usage for ccidx in pipeline for _usage in cells[ccidx2cellsidx[ccidx]]['target_names'] if len(_usage)>0]
    target_names = set([j for i in target_names for j in i])
    usage_names = [_usage for ccidx in pipeline for _usage in cells[ccidx2cellsidx[ccidx]]['usage_names'] if len(_usage)>0]
    # remove names defined in this pipeline. we dont add this type to context
    usage_names = [j for i in usage_names for j in i if j not in target_names]
    usage_names = sorted(set(usage_names), key=usage_names.index)  # deduplicate

    # Method 1: only check the definition before the pipeline beginning cell, closest definition will replace old one
    # Method 2(better): if we have vars defined in other middle cells not in this pipeline, use this one (a more context setting than Method 1
    posible_cells = [i for i in range(pipeline[-1]) if i not in pipeline and i in ccidx2cellsidx]
    posible_definitions = {_tgt: _def for ccidx in posible_cells for _tgt, _def in target_to_definition[ccidx2cellsidx[ccidx]].items()}

    try:
        contexts = [[usage_name, posible_definitions[usage_name]] for usage_name in usage_names]
    except:
        contexts = []
    return contexts


def get_context_import(pipeline, cells, target_to_definition):
    ccidx2cellsidx = {item['ccidx']:cellsidx for cellsidx, item in enumerate(cells)}
    temp_import_context = [import_lines for ccidx in pipeline for import_lines in cells[ccidx2cellsidx[ccidx]]['ip']]
    import_context = []
    [import_context.append(item) for item in temp_import_context if not item in import_context]
    return import_context


def judge_start_variable(methods, values_names, target_name, df_vars=None):
    if df_vars is None:
        df_vars = set()
    flag_start_var = False
    flag11, flag12, flag13, flag3 = False, False, False, False

    # 1. 1st. (1 or 2 or 3) whether in keywords dataframe
    if any([kw in mth for mth in methods for kw in KeyWord.keywords_interesting_start]):
        flag11 = True

    # 1. 2nd. (1 or 2 or 3) whether it's a (direct slice var) from (existing dfs). If do not contain calling a method and use assign, usually is derived from a var.
    if len(methods)==0 and len(set(values_names) & df_vars)>0:
        flag12 = True

    # 1. 3rd. (1 or 2 or 3) whether it's a direct slice var from (existing dfs). If call a built-in method from the var
    if len(methods)>0 and len(set(values_names) & df_vars)>0 and values_names[0] in methods[0]:
        flag13 = True

    # 3rd. (and 3) assigned definition appears the first time
    if target_name not in df_vars:
        flag3 = True

    if (flag11 or flag12 or flag13) and flag3:
        flag_start_var = True
    return flag_start_var


def judge_final_goal(methods):
    flag_final_goal = False
    flag1 = False

    # 1. 1st. whether in keywords final goal
    if any([kw in mth for mth in methods for kw in KeyWord.keywords_final_goal]):
        flag1 = True

    if flag1:
        flag_final_goal = True
    return flag_final_goal


def deduplicate_pipelines(pipeline_items):
    if len(pipeline_items) == 0:
        return []
    existing_paths = []
    new_pipelines = [pipeline_items[0].copy()]
    new_pipelines[0].update({'func': [new_pipelines[0]['func']],
                             'target_names': [new_pipelines[0]['target_names']],
                             'value_names': [new_pipelines[0]['value_names']], })
    for idx, item in enumerate(pipeline_items[1:]):
        if item['path_cell_idx'] in existing_paths:
            idx_same_pipe = existing_paths.index(item['path_cell_idx'])
            new_pipelines[idx_same_pipe]['func'].append(item['func'])
            new_pipelines[idx_same_pipe]['target_names'].append(item['target_names'])
            new_pipelines[idx_same_pipe]['value_names'].append(item['value_names'])
        else:
            new_item = item.copy()
            new_item.update({'func': [new_item['func']],
                             'target_names': [new_item['target_names']],
                             'value_names': [new_item['value_names']]})
            new_pipelines.append(new_item)
            existing_paths.append(item['path_cell_idx'])
    return new_pipelines


def remove_coverage_pipelines(pipeline_items):
    if len(pipeline_items) == 0:
        return []
    pipeline_items = sorted(pipeline_items, key=lambda x: len(x['path_cell_idx']), reverse=True)
    existing_paths = [item['path_cell_idx'] for item in pipeline_items]
    if debug:
        print("pipeline before remove coverage pipelines", len(pipeline_items))
        print(existing_paths)

    new_pipelines = []
    for idx, item in enumerate(pipeline_items):
        path = set(item['path_cell_idx'])
        if all([len(set(p).intersection(path))!=len(p) for p in existing_paths[idx+1:]]):
            new_pipelines.append(item)
    existing_paths = [item['path_cell_idx'] for item in new_pipelines]
    if debug:
        print("pipeline after remove coverage pipelines", len(new_pipelines))
        print(existing_paths)
    return new_pipelines


def add_markdown_cell_to_pipelines(pipeline_items, code2markdown_all, cells_idx_code2all, nb_langs):
    # * reusable
    if len(pipeline_items) == 0:
        return []

    new_pipelines = []
    for item in pipeline_items:
        new_item = item.copy()
        new_cell_idx = [code2markdown_all[cells_idx_code2all[idx]]+[cells_idx_code2all[idx]] for idx in item['path_cell_idx']]
        new_cell_idx = [j for i in new_cell_idx for j in i]
        new_item['path_cell_idx_w_md'] = new_cell_idx
        new_item['n_md'] = len(new_cell_idx) - len(item['path_cell_idx'])
        # add language
        new_item['lang'] = [nb_langs[i] for i in new_cell_idx]
        new_pipelines.append(new_item)
    return new_pipelines


def extract_from_notebooks(file, save_dspath_nb=False):
    stat = {'notebook_id': file.split('.')[0].split('_')[1], 'langs': [], 'dedup_path': [], 'node_info': [], 'df_cell': []}
    try:
        nb = read_json(os.path.join(from_path, file))
        if 'cells' in nb.keys():
            cell_key = 'cells'
        elif 'worksheets' in nb.keys():
            cell_key = 'worksheets'
        else:
            return stat
    except Exception as e:
        # print(file, e)
        return stat

    nb[cell_key] = [cell for cell in nb[cell_key] if not judge_blank_cell(cell)]
    nb_langs = [judge_language_langid(get_source_string(cell)) if cell.get('cell_type', '')!='code'  else judge_language(get_source_string(cell)) for cell in nb[cell_key]]
    # nb_langs = [judge_language_langid(get_source_string(cell)) for cell in nb[cell_key]  if cell.get('cell_type', '')!='code']
    # nb_langs = [get_source_string(cell) for cell in nb[cell_key]  if not cell.get('cell_type', '')!='code']
    # nb_langs = [judge_language(i) for i in nb_langs]

    idx_global_markdown, idx_code2markdown = get_code_cell_to_markdown_idx(nb[cell_key], )
    cells_string = [[cidx, clean_nb_code_to_script(cell)] for cidx, cell in enumerate(nb[cell_key]) if cell.get('cell_type', '')=='code']

    asts = []
    failed_asts = []
    for idx, (cidx, cell) in enumerate(cells_string):
        try:
            tree = ast.parse(cell)
            asts.append([idx, tree])
            # astpretty.pprint(tree, show_offsets=False)
        except Exception as e:
            if debug:
                print("failed_ast: {}, in ast {}/{}".format(file, idx, len(cells_string)))
            failed_asts.append(idx)
            continue
    # [cells_string.pop(i) for i in reversed(failed_asts)]
    # cells_idx_code2all = {i[0]: j[0] for i, j in zip(asts, cells_string)}
    cells_idx_code2all = {i[0]: j[0] for i, j in zip(asts, [_item for _item in cells_string if _item[0] not in failed_asts])}

    global variable_history
    variable_history = set()
    global self_function_history
    self_function_history = set()
    start_variable_list = []
    final_goal_api_list = []
    cells_node_info = []
    import_lib_list = []  # ['pandas', 'pd', 'import pandas as pd'], [lib, asname, code]
    df_variables, initial_variables = set(), set()
    # 1. find paired variables, 2. find start variables and 3. find final goal variables
    for idx, tree in asts:
        if idx == 9:
            a=1
        visitor_s1 = S1DefinitionAssignLister()
        # visitor_s1.s_list, visitor_s1.f_name, visitor_s1.arg_arr, visitor_s1.symb_arr, visitor_s1.f_dict, visitor_s1.symb_dict = [], [], [], [], {}, {}
        try:
            visitor_s1.visit(tree)
            visitor_s1.num_statements = len(tree.body)
            visitor_s1.token_num = tokenize_cell_and_get_token_num(cells_string[idx][1], split_comment=True)[1]
            import_lib_list.extend(list(map(list, zip(visitor_s1.import_pkg, visitor_s1.import_module, visitor_s1.import_asname, visitor_s1.import_code))))
        except RecursionError:
            if debug:
                print("RecursionError: {}, in ast {}/{}".format(file, idx, len(asts)))
            cells_node_info.append({'ccidx': idx, 'target_names': [], 'target_define_line': [], 'value_names': [],
                                    'variable_start_apis': [], 'usage_names': [], 'usage_apis': [],
                                    'n_stmt': 0, 'n_tok':0})
            continue
        if debug:
            print(idx)
            # print('import_asname\t', visitor_s1.import_asname)
            # print('block_target\t', visitor_s1.block_target)
            print('targe_names\t', visitor_s1.target_names)
            print('target_define_line\t', visitor_s1.target_define_line)
            print('value_names\t', visitor_s1.value_names)
            print('usage_names\t', visitor_s1.usage_names)
            print('varsta_apis\t', visitor_s1.variable_start_apis)
            print('usages_apis\t', visitor_s1.usage_apis)
            print('num_statements\t', visitor_s1.num_statements)
            print('token_num\t', visitor_s1.token_num)
        #     # print(astunparse.unparse(tree))

        # get Cells with a start variable:
        for j, funcs in enumerate(visitor_s1.variable_start_apis):
            flag_start_var = judge_start_variable(methods=funcs,
                                                  values_names=visitor_s1.value_names[j],
                                                  target_name=visitor_s1.target_names[j][0],
                                                  df_vars=df_variables)
            if flag_start_var:
                start_variable_list.append({'ccidx': idx,
                                            'func': funcs,
                                            'target_names': visitor_s1.target_names[j],
                                            'value_names': visitor_s1.value_names[j],
                                            'usage_names': visitor_s1.usage_names,
                                            'usage_apis': visitor_s1.usage_apis,
                                            '_idx': j,
                                            'path_cell_idx': []
                                            })
                df_variables.update(visitor_s1.target_names[j])
                initial_variables.add(visitor_s1.target_names[j][0])

        # get Cells with a usage function
        for j, funcs in enumerate(visitor_s1.usage_apis):
            flag_final_goal = judge_final_goal(methods=funcs)
            if flag_final_goal:
                final_goal_api_list.append({'ccidx': idx,
                                            'func': funcs,
                                            'usage_names': visitor_s1.usage_names[j],
                                            'usage_apis': visitor_s1.usage_apis[j],
                                            '_idx': j,
                                            'path_cell_idx': []
                                            })

        # get import information for each cell
        import_lib_dict = {item[2]: item for item in import_lib_list}
        # _apis_clean = [re.split(r'[.()]', _stmt) for _expr in visitor_s1.usage_apis for _stmt in _expr]
        _apis_clean = [_word for _expr in visitor_s1.usage_apis for _stmt in _expr
                                for _word in re.split(r'[.()]', _stmt) if len(_word)>0]
        temp_import_info = [import_lib_dict[_api] for _api in _apis_clean if _api in import_lib_dict]
        import_info = []
        [import_info.append(item) for item in temp_import_info if not item in import_info]

        # write save node info
        cells_node_info.append({'ccidx': idx,
                                'target_names': visitor_s1.target_names,
                                'target_define_line': visitor_s1.target_define_line,
                                'value_names': visitor_s1.value_names,
                                'variable_start_apis': visitor_s1.variable_start_apis,
                                'usage_names': visitor_s1.usage_names,
                                'usage_apis': visitor_s1.usage_apis,
                                'n_stmt': visitor_s1.num_statements,
                                'n_tok': visitor_s1.token_num,
                                'ip': import_info,
                                })
    if debug:
        print('start_variable_list: ', len(start_variable_list))
        print('final_goal_api_list: ', len(final_goal_api_list))
        print('cells_node_info: ', len(cells_node_info))
        # [print(item) for item in start_variable_list]
        # [print(item) for item in final_goal_api_list]
        # print()

    # tranverse from start variable
    extracted_path_list = []
    all_dataflow_cells = []
    target2definition = get_target_to_definition(cells=cells_node_info)
    for item in start_variable_list:
        if item['target_names'] == ['df_t0']:
            print()
        path_var_not_derive, path_reassign = tranverse_path_from_start(start_item=item, cells=cells_node_info)
        path_with_final_goal, path_stat = tranverse_path_find_end(start_item=item, pipeline=path_var_not_derive, reassign=path_reassign, cells=cells_node_info, final_goals=final_goal_api_list)
        code_var_context = [get_context_code(pipeline=pipe, cells=cells_node_info, target_to_definition=target2definition) for pipe in path_with_final_goal]
        import_context = [get_context_import(pipeline=pipe, cells=cells_node_info, target_to_definition=target2definition) for pipe in path_with_final_goal]
        for idx, pipe in enumerate(path_with_final_goal):
            new_item = item.copy()
            new_item['path_cell_idx'] = pipe
            new_item['code_var_context'] = code_var_context[idx]
            new_item['import_context'] = import_context[idx]
            new_item['n_code'] = len(pipe)
            new_item['n_md'] = 0
            new_item['n_stmt'] = path_stat['final_num_statements'][idx]
            new_item['n_tok'] = path_stat['final_token_num'][idx]
            extracted_path_list.append(new_item)
            if debug:
                print('f ', new_item)
        if debug:
            print('v ', path_var_not_derive)
        # all_dataflow_path.append(path_var_not_derive)
        data_flow_cells = [cells_idx_code2all[idx] for idx in path_var_not_derive]
        if data_flow_cells not in all_dataflow_cells:
            all_dataflow_cells.append(data_flow_cells)

        # item['path_cell_idx'] = path_var_not_derive
        # if len(path_var_not_derive)>=0:
        #     extracted_path_list.append(item)
        #     if debug:
        #         print(item)
        #         print('f ', path_var_not_derive)
        #         print('f ', path_with_final_goal)

    # remove duplicate pipelines
    deduped_path_list = deduplicate_pipelines(extracted_path_list)
    if remove_coverage_pipe:
        deduped_path_list = remove_coverage_pipelines(deduped_path_list)
    deduped_path_list = add_markdown_cell_to_pipelines(deduped_path_list, idx_code2markdown, cells_idx_code2all, nb_langs)

    if debug:
        [print('d ', item) for item in deduped_path_list]

    if debug:
        print("len of all code cells:", len(asts))
        print("len of start_variable_list:", len(start_variable_list))
        print("len of final_goal_api_list:", len(final_goal_api_list))
        print("len of extracted_path_list:", len(extracted_path_list))
        print("len of deduped_path_list:", len(deduped_path_list))
        # print("idxs of asts", [i[0] for i in asts])
        # print("idxs of code cells in NBs", [i[0] for i in cells_string])
        # print(start_variable_list)
        print([item['target_names'] for item in start_variable_list])
        print([item['usage_apis'] for item in final_goal_api_list])
        # print("idx map from asts idx to code cell idx: ", {i[0]:j[0] for i, j in zip(asts, cells_string)})

    # convert path_cell_idx from idx in only code cell to idx in whole notebook
    [item.update({'path_cell_ccidx': [i for i in item['path_cell_idx']]}) for item in deduped_path_list]
    [item.update({'path_cell_idx': [cells_idx_code2all[i] for i in item['path_cell_idx']]}) for item in deduped_path_list]
    if save_dspath_nb:
        save_dir = os.path.join(to_path, file.split('.')[0])
        if len(deduped_path_list) > 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            write_json(nb, os.path.join(save_dir, file))
            file_idx = Counter()
            for item in deduped_path_list:
                # with markdown  # TODO create cells in different settings
                if add_markdown_idx:
                    new_cells = [nb[cell_key][idx] for idx in item['path_cell_idx_w_md']]
                else:
                    new_cells = [nb[cell_key][idx] for idx in item['path_cell_idx']]

                new_nb = copy.deepcopy(nb)
                new_nb[cell_key] = new_cells
                _name_vars = [_i[0] for _i in item['target_names']]
                _name_vars = ','.join(sorted(set(_name_vars), key=_name_vars.index))[:130]
                _name_vars = re.sub(r'[^\x00-\x7F]+', '', _name_vars)
                file_idx[_name_vars] += 1
                new_file = file.split('.')[0] + ',' + _name_vars + '-{}.ipynb'.format(file_idx[_name_vars])
                try:
                    write_json(new_nb, os.path.join(os.path.join(to_path, file.split('.')[0]), new_file))
                except:
                    # print(os.path.join(os.path.join(to_path, file.split('.')[0]), new_file))
                    print("write pipeline to new file json error: {}".format(file))
                    continue
    if debug:
        print("extracted path list: ", len(extracted_path_list))
        print("extracted path list: ", len(deduped_path_list))
        print("dataflow path list: ", len(all_dataflow_cells))
    stat['df_cell'] = all_dataflow_cells
    # stat['extra_path'] = extracted_path_list
    stat['dedup_path'] = deduped_path_list
    stat['node_info'] = cells_node_info
    stat['langs'] = nb_langs
    return stat


debug=False
add_markdown_idx=True
remove_coverage_pipe=False
# from_path = '../raw_notebooks'
# to_path = '../CoCoMine-saved_results'
from_path = sys.argv[1]
to_path = sys.argv[2]

if not os.path.exists(to_path):
    os.makedirs(to_path)
files = os.listdir(from_path)
files = [p for p in files if p.endswith('.ipynb')]
# files = files[:1000]
print(len(files))

if len(files) > 1000:
    n_threads = 8
    with Pool(n_threads) as p:
        # func_ = partial(extract_from_notebooks, save_dspath_nb=False)
        func_ = partial(extract_from_notebooks, save_dspath_nb=True)
        stats = list(tqdm(p.imap(func_, files, chunksize=16), total=len(files), desc="extract pipeline from {}".format(from_path)))
        stats = {"NB_{}".format(item['notebook_id']): item for item in stats}
else:
    stats = {}
    for file in files:
        print(file)
        stat = extract_from_notebooks(file, save_dspath_nb=True)
        stats[stat['notebook_id']] = stat

write_json(stats, os.path.join(to_path, 'statistics.json'))
n_pipe = sum([len(item['dedup_path']) for item in stats.values()])
no_path_one = [item for item in stats.values() if len(item['dedup_path']) == 0]
no_cell_one = [item for item in stats.values() if len(item['node_info']) == 0]
print("\nno path: {}/{}, no cell: {}/{}".format(len(no_path_one), len(stats), len(no_cell_one), len(stats)))
print("total pipeline number: {} in {} NBs, average: {}".format(n_pipe, len(files), n_pipe / len(files)))
print("(OnlyWithPath) total pipeline: {} in {} NBs, average: {}".format(n_pipe, len(files) - len(no_path_one),
                                                                            n_pipe / (len(files) - len(no_path_one))))


