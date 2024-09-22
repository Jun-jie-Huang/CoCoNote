import json
import os
from collections import Counter
from pipeline_utils import KeyWord


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data
def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=1)


# TODO: change the path
# basic_path = sys.argv[1]
basic_path = '../CoCoMine-saved_results'
if not os.path.exists(os.path.join(basic_path, "statistics")):
    os.makedirs(os.path.join(basic_path, "statistics"))
path = f'{basic_path}/statistics.json'
if not os.path.exists(path):
    print("No statistics.json file, exit.")
    exit()

stats = []
no_path_items = []
no_cell_items = []
all_pipeline_num = []
all_dataflow_cells_num = []
pipeline_length_code = []
pipeline_length_md = []
pipeline_length_all = []
num_statements = []
num_tokens = []
variable_start_apis_counter = Counter()
usage_apis_counter = Counter()
keywords_start_counter = Counter()
keywords_start_counter_soft = Counter()
keywords_final_counter = Counter()
keywords_final_counter_soft = Counter()
pipe_type_counter_plot = Counter()
pipe_type_counter_model = Counter()
pipe_type_counter_stats = Counter()


stat = read_json(path)
print(f"number of notebooks: {len(stat)}")

# find path/pipeline number
no_path_one = [item for item in stat.values() if len(item['dedup_path']) == 0]
path_num = [len(item['dedup_path']) for item in stat.values()]
no_cell_one = [item for item in stat.values() if len(item['node_info']) == 0]
print("no path: {}/{}, no cell: {}/{}, N pipelines: {}".format(len(no_path_one), len(stat), len(no_cell_one), len(stat), sum(path_num)))
no_path_items.append((len(no_path_one), len(stat)))
no_cell_items.append((len(no_cell_one), len(stat)))
all_pipeline_num.append(path_num)
all_dataflow_cells_num.extend([item.get('df_cell', []) for item in stat.values()])

# find code/md/all cell numbers
length_all= [[len(p.get('path_cell_idx_w_md', [])) for p in item['dedup_path']] for item in stat.values()]
length_c = [[len(p.get('path_cell_idx', [])) for p in item['dedup_path']] for item in stat.values()]
length_w_md = [[len(p.get('path_cell_idx_w_md', [])) - len(p.get('path_cell_idx', [])) for p in item['dedup_path']] for item in stat.values()]
pipeline_length_code.append(length_c)
pipeline_length_md.append(length_w_md)
pipeline_length_all.append(length_all)

# count number of code statements
statements = [[p.get('n_stmt', []) for p in item['dedup_path']] for item in stat.values()]
num_statements.append(statements)
# count number of code tokens
tokens = [[p.get('n_tok', []) for p in item['dedup_path']] for item in stat.values()]
num_tokens.append(tokens)

# find variable start apis for data frame
for item in stat.values():
    for node in item['node_info']:
        names = node.get('variable_start_apis', [[]])
        for key in [j for i in names for j in i]:
            variable_start_apis_counter[key] += 1

# find apis for tasks
for item in stat.values():
    for node in item['node_info']:
        names = node.get('usage_apis', [[]])
        for key in [j for i in names for j in i]:
            usage_apis_counter[key] += 1

# keywords start api
for item in stat.values():
    for node in item['node_info']:
        names = node.get('variable_start_apis', [[]])
        for key in [j for i in names for j in i]:
            if key in KeyWord.keywords_interesting_start:
                keywords_start_counter[key] += 1
            for mth in KeyWord.keywords_interesting_start:
                if key in mth:
                    keywords_start_counter_soft[mth] += 1

# keywords final goal
for item in stat.values():
    for node in item['node_info']:
        names = node.get('usage_apis', [[]])
        for key in [j for i in names for j in i]:
            if key in KeyWord.keywords_final_goal:
                keywords_final_counter[key] += 1
            for mth in KeyWord.keywords_final_goal:
                if key in mth:
                    keywords_final_counter_soft[mth] += 1
                    # final_goal_per_pipe.append(mth)

# pipeline type with only one last final goal keyword
for item in stat.values():
    pipe_last_cell_ccidx = [p['path_cell_idx'][-1] for p in item['dedup_path']]
    last_cell_nodes = [node for node in item['node_info'] if node['ccidx'] in pipe_last_cell_ccidx]
    for last_cell_node in last_cell_nodes:
        names = last_cell_node.get('usage_apis', [[]])
        names = [j for i in names for j in i]
        final_goal_per_pipe = []
        for key in names:
            for mth in KeyWord.keywords_final_goal:
                if key in mth:
                    final_goal_per_pipe.append(mth)
        if len(final_goal_per_pipe) > 0:
            key = final_goal_per_pipe[-1]
            if key in KeyWord.keywords_f_model:
                pipe_type_counter_model[key] += 1
            elif key in KeyWord.keywords_f_stats:
                pipe_type_counter_stats[key] += 1
            elif key in KeyWord.keywords_f_plot:
                pipe_type_counter_plot[key] += 1

n_actual_nbs = sum([j for i,j in no_cell_items]) - sum([i for i,j in no_cell_items])
n_have_path = sum([j for i,j in no_cell_items]) - sum([i for i,j in no_cell_items]) - sum([i for i,j in no_path_items])
n_all_pipelines = sum([sum(i) for i in all_pipeline_num])
print("ALL: No. NB no path: {}/{}, No. NB no cell: {}/{},".format(sum([i for i,j in no_path_items]),
                                                                  sum([j for i,j in no_path_items]),
                                                                  sum([i for i,j in no_cell_items]),
                                                                  sum([j for i,j in no_cell_items]),))
print("ALL: NO. of pipe: {}, NO. of NB have pipe: {}, NO. of actual NBs: {}".format(n_all_pipelines, n_have_path, n_actual_nbs))
print("ALL: avg. pipes per NB: {:.3f}".format(n_all_pipelines/n_have_path, ))

path = '{}/statistics/pipeline_length_code.json'.format(basic_path)
write_json(pipeline_length_code, path)
path = '{}/statistics/pipeline_length_md.json'.format(basic_path)
write_json(pipeline_length_md, path)
path = '{}/statistics/pipeline_length_all.json'.format(basic_path)
write_json(pipeline_length_all, path)

n_cell_list_code = [k for i in pipeline_length_code for j in i for k in j if len(j)>0]
n_cell_list_md = [k for i in pipeline_length_md for j in i for k in j if len(j)>0]
n_cell_list_all = [k for i in pipeline_length_all for j in i for k in j if len(j)>0]
print()
print("ALL: code cell: max:{}, mean:{:.3f} ".format(max(n_cell_list_code), sum(n_cell_list_code)/len(n_cell_list_code)))
print("ALL: markdown cell: max:{}, mean:{:.3f} ".format(max(n_cell_list_md), sum(n_cell_list_md)/len(n_cell_list_md)))
print("ALL: all cell: max:{}, mean:{:.3f} ".format(max(n_cell_list_all), sum(n_cell_list_all)/len(n_cell_list_all)))

path = '{}/statistics/num_tokens.json'.format(basic_path)
write_json(num_tokens, path)
path = '{}/statistics/num_statements.json'.format(basic_path)
write_json(num_statements, path)

n_tokens_code_per_pipe = [sum(pipe) for result in num_tokens for item in result for pipe in item]
n_tokens_code_per_cell = [cell for result in num_tokens for item in result for pipe in item for cell in pipe]
print()
print("ALL Code Tokens number: per pipeline: max: {}, mean: {}/{}={:.3f}".format(max(n_tokens_code_per_pipe), sum(n_tokens_code_per_pipe), len(n_tokens_code_per_pipe), sum(n_tokens_code_per_pipe)/len(n_tokens_code_per_pipe)))
print("ALL Code Tokens number: per code cell: max: {}, mean: {}/{}={:.3f}".format(max(n_tokens_code_per_cell), sum(n_tokens_code_per_cell), len(n_tokens_code_per_cell), sum(n_tokens_code_per_cell)/len(n_tokens_code_per_cell)))
n_stmt_code_per_pipe = [sum(pipe) for result in num_statements for item in result for pipe in item]
n_stmt_code_per_cell = [cell for result in num_statements for item in result for pipe in item for cell in pipe]
print("ALL Statements number: per pipeline: max: {}, mean: {}/{}={:.3f}".format(max(n_stmt_code_per_pipe), sum(n_stmt_code_per_pipe), len(n_stmt_code_per_pipe), sum(n_stmt_code_per_pipe)/len(n_stmt_code_per_pipe)))
print("ALL Statements number: per code cell: max: {}, mean: {}/{}={:.3f}".format(max(n_stmt_code_per_cell), sum(n_stmt_code_per_cell), len(n_stmt_code_per_cell), sum(n_stmt_code_per_cell)/len(n_stmt_code_per_cell)))

path = '{}/statistics/variable_start_apis.json'.format(basic_path)
write_json(variable_start_apis_counter, path)

path = '{}/statistics/usage_apis.json'.format(basic_path)
write_json(usage_apis_counter, path)

path = '{}/statistics/keywords_interesting_start.json'.format(basic_path)
write_json(keywords_start_counter, path)
path = '{}/statistics/keywords_interesting_start_soft.json'.format(basic_path)
write_json(keywords_start_counter_soft, path)

path = '{}/statistics/keywords_final_goal.json'.format(basic_path)
write_json(keywords_final_counter, path)
path = '{}/statistics/keywords_final_goal_soft.json'.format(basic_path)
write_json(keywords_final_counter_soft, path)


path = '{}/statistics/pipeline_final_goal_plot.json'.format(basic_path)
write_json(pipe_type_counter_plot, path)
path = '{}/statistics/pipeline_final_goal_model.json'.format(basic_path)
write_json(pipe_type_counter_model, path)
path = '{}/statistics/pipeline_final_goal_stats.json'.format(basic_path)
write_json(pipe_type_counter_stats, path)


path = '{}/statistics/all_dataflow_cells_num.json'.format(basic_path)
write_json(all_dataflow_cells_num, path)
n_dfs = sum([len(i) for i in all_dataflow_cells_num])
n_nbs = len([i for i in all_dataflow_cells_num if len(i)>0])
print()
print("ALL: dataflow path num: {}, nbs with dataflow: {}/{}, nbs without dfs: {}".format(n_dfs, n_nbs, len(all_dataflow_cells_num), len(all_dataflow_cells_num)-n_nbs))
print("ALL: avg. dataflow path in nbs: {}".format(n_dfs/n_nbs))







