import os
import re
import copy
import platform
import shutil
from utils import read_json, write_json


def merge_dataset_generation(dataset, generations, gt_code=False):
    """

    Args:
        dataset: raw dataset, "test_1654.pkl"
        generations: generation code in json format,
            which is a dict with keys of "nbid", "file", "row_id", "target_code", "generation"
        gt_code: boolean, whether to use ground truth code as generation code

    Returns:

    """
    # add identifier: "nbid_rowid"
    dataset_files = [{'idx': idx,
                      'nbid_rowid': f"{item['nbid']}R{item['row_id']}",
                      'nbid': item['nbid'],
                      'file': item['file'],
                      'row_id': item['row_id'],
                      'target_code': item['target_str'],
                      'input': item['in_df'],
                      'output': item['out_df'],
                      } for idx, item in enumerate(dataset)]
    _ = [item.update({"nbid_rowid": f"{item['nbid']}R{item['row_id']}"}) for item in generations]
    dataset_files.sort(key=lambda k: (k.get('nbid_rowid', 0)))
    generations.sort(key=lambda k: (k.get('nbid_rowid', 0)))

    empty_string = []
    for idx, (dataset_item, generation_item) in enumerate(zip(dataset_files, generations)):
        if gt_code:
            dataset_item['generation'] = dataset_item['target_code']
        else:
            dataset_item['generation'] = generation_item['generation']
            # dataset_item['generation'] = cleaning_string(''.join(generation_item['target_code'].split(' ')))
        if dataset_item['generation'] == "":
            empty_string.append(idx)
        # print("idx: {}, code: {}".format(idx, dataset_item['generation']))
    print("number of unfoundable generation: {}/{}".format(len([item for item in dataset_files if item['generation'] == ""]), len(dataset_files)))
    print("number of empty string notebooks: {}".format(len(empty_string)))
    dataset_files.sort(key=lambda k: (k.get('idx', 0)))

    return dataset_files


def compare_code_string(code1, code2):
    code1, code2 = ''.join(code1.split(' ')), ''.join(code2.split(' '))
    code1, code2 = code1.replace('\"', '\''), code2.replace('\"', '\'')
    return code1 == code2


def cleaning_string(string):
    # remove extra \n
    # change double \" to single \'
    if string:
        string = re.sub(r'\n[ \n\t]*\n', r'\n', string)
        string = re.sub("\"", "\'", string)
        return string
    else:
        return ''


# def read_generation_gpt(path_ground_truth, path_generation, path_save_merged=""):
#     with open(path_ground_truth, 'r', encoding='utf-8') as fh:
#         gold_file = fh.read().split('##########\n')
#     with open(path_generation, 'r', encoding='utf-8') as fh:
#         generation_file = fh.read().split('##########\n')
#     if gold_file[-1] == '' and generation_file[-1] == '':
#         gold_file = gold_file[:-1]
#         generation_file = generation_file[:-1]
#     print("reference_text: {}".format(len(gold_file)))
#     print("translation_text: {}".format(len(generation_file)))
#
#     generations = []
#     for target, generation in zip(gold_file, generation_file):
#         item = {}
#         item["input"] = ""
#         item["target"] = decode_string(target)
#         item["generation"] = decode_string(generation)
#         generations.append(item)
#     print("generations: {}".format(len(generations)))
#     if not path_save_merged:
#         path_save_merged = os.path.join(os.path.dirname(path_generation), 'split_generation_results.json')
#     write_json(generations, path_save_merged)
#     return generations
# post_replace = {
#     "Ġ": " ",
#     "Ċ": "\n",
#     "ĉ": "\t",
#     'madeupword0001': '\'jupyter_string\''
# }
# def remove_madeupword(code):
#     item_list = re.split(r'(madeupword\d{4})', code)
#     item_effective = []
#     for s in item_list:
#         if 'madeupword' in s:
#             continue
#         else:
#             item_effective.append(s)
#     return ''.join(item_effective)
# def decode_string(hyp):
#     if len(hyp) == 0:
#         return ''
#     if hyp[-1] == '\n':
#         hyp = hyp[:-1]
#     for s, t in post_replace.items():
#         hyp = hyp.replace(s, t)
#     hyp = remove_madeupword(hyp)
#     return hyp


# indent_template1 = re.compile(r' {4}(.+)')
# import_template1 = re.compile(r'from(.+)import(.+)as(.+)$')
# import_template2 = re.compile(r'import(.+)as(.+)$')
# import_template3 = re.compile(r'from(.+)import(.+)$')
# import_template4 = re.compile(r'import(.+)$')
# for_template1 = re.compile(r'(.*)for(.+)in(.+):(.*)$')
# for_template2 = re.compile(r'(.*)[\[\{](.*)for(.+)in(.+)[\]\}](.*)$')
# lambda_template1 = re.compile(r'(.+)lambda(.+):(.+)$')
# def merge_generation_code(generation):
#     # return ''.join(generation.split(' '))
#     codes = generation.split('\n')
#     new_codes = []
#     if len(codes) > 0:
#         codes[:-1] = [item + '\n' for item in codes[:-1]]
#         for code in codes:
#             temp = ''.join(code.split(' '))
#             if indent_template1.match(code):
#                 temp = '    ' + temp
#             # print("temp 1: {}".format(temp))
#             if import_template1.match(temp):
#                 match = import_template1.match(temp)
#                 temp = 'from '+match.group(1)+' import '+match.group(2)+' as '+match.group(3)
#             elif import_template2.match(temp):
#                 match = import_template2.match(temp)
#                 temp = 'import '+match.group(1)+' as '+match.group(2)
#             elif import_template3.match(temp):
#                 match = import_template3.match(temp)
#                 temp = 'from '+match.group(1)+' import '+match.group(2)
#             elif import_template4.match(temp):
#                 match = import_template4.match(temp)
#                 temp = 'import '+match.group(1)
#             elif for_template1.match(temp):
#                 match = for_template1.match(temp)
#                 _indi = 0 if len(match.group(1)) == 0 else 1
#                 temp = match.group(1)+' '*_indi+'for '+match.group(2)+' in '+match.group(3)+':'+match.group(4)
#             elif for_template2.match(temp):
#                 match = for_template2.match(temp)
#                 temp = match.group(1)+temp[len(match.group(1))]+match.group(2)+' for '+match.group(3)+' in '+\
#                        match.group(4)+temp[6+len(match.group(1))+len(match.group(2))+len(match.group(3))+len(match.group(4))]+match.group(5)
#             elif lambda_template1.match(temp):
#                 match = lambda_template1.match(temp)
#                 temp = match.group(1)+' lambda '+match.group(2)+':'+match.group(3)
#             # print("temp 2: {}".format(temp))
#             new_codes.append(temp)
#         # return ''.join(new_codes)
#     return '\n'.join(new_codes)


def replace_base_path(cells, old_path, new_path):
    new_cells = []
    for cell in cells:
        new_cell = copy.deepcopy(cell)
        string = '[SPLIT]'.join(cell['source'])
        string = re.sub("DATASET_NB_ROOT", new_path, string)
        string = string.replace(old_path, new_path)
        new_cell['source'] = string.split('[SPLIT]')
        new_cells.append(new_cell)
    return new_cells

def replace_file_dir_path(cells, old_path, new_path):
    new_cells = []
    for cell in cells:
        new_cell = copy.deepcopy(cell)
        string = '[SPLIT]'.join(cell['source'])
        string = string.replace(old_path, new_path)
        new_cell['source'] = string.split('[SPLIT]')
        new_cells.append(new_cell)
    return new_cells


def write_one_notebook(args, path_in, path_out, nbid, row_id, replace_code):
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    print("nbid:{}, rowid:{}, path_in: {}".format(nbid, row_id, path_in))
    files = os.listdir(path_in)
    if not os.path.exists(os.path.join(path_out, 'SAVE')):
        os.makedirs(os.path.join(path_out, 'SAVE'))
    for file in files:
        if file == '{}.ipynb'.format(nbid):
            notebook = read_json(os.path.join(path_in, file))
            new_cell = copy.deepcopy(notebook['cells'][int(row_id)])
            # ori_code = ''.join(new_cell['source'])
            new_cell['source'] = replace_code.split('\n')
            new_cell['source'][:-1] = [item + '\n' for item in new_cell['source'][:-1]]
            notebook['cells'][int(row_id)] = new_cell
            # replace the path in the cells to new path
            notebook['cells'] = replace_base_path(notebook['cells'], args.path_notebooks, os.path.dirname(path_out))
            notebook['cells'] = replace_file_dir_path(notebook['cells'], nbid, nbid+'R{}'.format(row_id))
            notebook['cells'] = notebook['cells'][:row_id+1]
            write_json(notebook, os.path.join(path_out, nbid+'R{}.ipynb'.format(row_id)))
        elif 'ipynb_checkpoints' in file:
            continue
        else:
            _src = os.path.join(path_in, file)
            _dst = path_out
            shutil.copy(_src, _dst)


def write_notebook(dataset_file, args, save=True):
    path_in = os.path.join(args.path_notebooks, dataset_file['nbid'])
    directory = dataset_file['nbid'] + 'R{}'.format(dataset_file['row_id'])

    # write notebook with generated code
    filling_code = dataset_file['generation']
    path_out = os.path.join(args.path_save_notebooks, "generation", directory)
    if save:
        write_one_notebook(args, path_in, path_out, dataset_file['nbid'], dataset_file['row_id'], filling_code)
    index = [[path_out, f"{directory}.ipynb", directory]]

    # write notebook with original code
    filling_code = dataset_file['target_code']
    path_out = os.path.join(args.path_save_notebooks, "ground_truth", directory)
    if save:
        write_one_notebook(args, path_in, path_out, dataset_file['nbid'], dataset_file['row_id'], filling_code)
    index.append([path_out, f"{directory}.ipynb", directory])

    return index
