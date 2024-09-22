import os
import time
import argparse
import pandas as pd
import nbformat as nbf

from utils import read_json, write_json, write_pickle, read_pickle
# from prepare_nb import decode_string, read_generation_gpt, read_generations
from prepare_nb import cleaning_string, write_notebook, merge_dataset_generation
from postprocess_nb import obtain_cell_output, compare_answers_notebookcoder_new, error_count
from metric import compute_macro_f1

"""
python evaluate.py \ 
    --do_create_notebook \
    --do_run \
    --do_evaluate \
    --generation_file ../evaluation_example/test_1654_gpt35.json \
    --path_save_notebooks ../saved_results/evaluate_gpt35

"""
parser = argparse.ArgumentParser()
parser.add_argument("--do_create_notebook", action='store_true', help="Whether to run write generation.")
parser.add_argument("--do_run", action='store_true', help="Whether to run notebooks.")
parser.add_argument("--do_evaluate", action='store_true', help="Whether to run evaluation.")
parser.add_argument("--path_generation", default="../evaluation_example/test_1654_gpt35.json", type=str, help="The path to generation output .")
parser.add_argument("--path_save_notebooks", default="../saved_results/evaluate_gpt35", type=str, help="The path to save notebooks.")

## Other parameters
parser.add_argument("--path_dataset", default="../dataset/test_1654.pkl", type=str, help="The path to raw testset.")
parser.add_argument("--path_notebooks", default="../notebooks", type=str, help="The path to original notebooks .")
parser.add_argument("--prefix", default="icl1", type=str, help="The path to generation output .", choices=["icl0", "icl1", "icl1_noCol", "icl1_noIn", "icl1_noIn_noOut", "icl1_noOut", "icl1_noNL", "icl1_noVal"])
args = parser.parse_args()

args.path_generation = os.path.join(args.path_generation, f"test_1654.json_coconote")
args.path_save_notebooks = os.path.join(args.path_save_notebooks, f"evaluate_{args.prefix}")

path_index_gt = os.path.join(args.path_save_notebooks, "test_gt_index.json")
path_index_gen = os.path.join(args.path_save_notebooks, "test_gen_index.json")
if not os.path.exists(args.path_save_notebooks):
    os.makedirs(args.path_save_notebooks)
    os.makedirs(os.path.join(args.path_save_notebooks, "ground_truth"))
    os.makedirs(os.path.join(args.path_save_notebooks, "generation"))

# args.codegpt = False
# if args.codegpt:
#     path_generation_result = os.path.join(os.path.dirname(args.path_generation), 'split_generation_results.json')
#     generations = read_generation_gpt(args.path_ground_truth, args.path_generation, path_save_merged=path_generation_result)
#     # generations = read_json(path_merged_generation_result)
#     generation_map = {cleaning_string(''.join(generations[idx]['target'].split(' '))): idx for idx in range(len(generations))}

raw_dataset = pd.read_pickle(args.path_dataset)
generations = read_json(args.path_generation)
print("length of raw_dataset: {}".format(len(raw_dataset)))
print("length of generations: {}".format(len(generations)))
dataset_gt = merge_dataset_generation(raw_dataset, generations, gt_code=False)

# process data, create new notebooks and save index files
if args.do_create_notebook:
    input_data = dataset_gt
    indexes = []
    for idx, dataset_file in enumerate(input_data):
        index = write_notebook(dataset_file, args, save=True)
        # index = write_notebook(dataset_file, args, save=False)
        indexes.append(index)
    print("number of notebooks to run : {}".format(len(indexes)))
    write_json([i[0] for i in indexes], path_index_gen)
    write_json([i[1] for i in indexes], path_index_gt)

# running evaluation
# TODO change logic by running the execution in the directory that save the notebooks
if args.do_run:
    print("##### Start running notebooks (generations)")
    t1 = time.time()
    cmd = f'python run_one_nb.py --base_dir {os.path.join(args.path_save_notebooks, "generation")} --idx_path {path_index_gen}'
    cmd += f' 2>&1 |tee {args.path_save_notebooks}/ALL_run_nbs_gen.log'
    # cmd = f'python run_one_nb.py --base_dir {args.path_save_notebooks} --idx_path {os.path.basename(args.path_index)} 2>&1 |tee {args.path_save_notebooks}/replace_base.log'
    print("cmd: {}".format(cmd))
    os.system(cmd)
    print("time: {}s".format(time.time()-t1))

    print("##### Start running notebooks (ground truth)")
    t1 = time.time()
    cmd = f'python run_one_nb.py --base_dir {os.path.join(args.path_save_notebooks, "ground_truth")} --idx_path {path_index_gt}'
    cmd += f' 2>&1 |tee {args.path_save_notebooks}/ALL_run_nbs_gt.log'
    # cmd = f'python run_one_nb.py --base_dir {args.path_save_notebooks} --idx_path {os.path.basename(args.path_index)} 2>&1 |tee {args.path_save_notebooks}/replace_base.log'
    print("cmd: {}".format(cmd))
    os.system(cmd)
    print("time: {}s".format(time.time()-t1))

# test
# TODO change saved file from pickle to json
if args.do_evaluate:
    indexes = read_json(path_index_gen)

    all_output = []
    all_results = []
    print("length of indexes: {}".format(len(indexes)))
    for idx, index in enumerate(indexes):
        nbid = index[2]
        row_id = int(nbid.split('R')[1])
        notebook_path_gt = os.path.join(args.path_save_notebooks, "ground_truth", nbid, f'run_{nbid}.ipynb')
        notebook = nbf.read(notebook_path_gt, nbf.NO_CONVERT)
        output_gt = obtain_cell_output(notebook, row_id)
        # notebook_path_gen = os.path.join(args.path_save_notebooks, "ground_truth", nbid, f'run_{nbid}.ipynb')
        notebook_path_gen = os.path.join(args.path_save_notebooks, "generation", nbid, f'run_{nbid}.ipynb')
        notebook = nbf.read(notebook_path_gen, nbf.NO_CONVERT)
        output_gen = obtain_cell_output(notebook, row_id)
        all_output.append({'idx': idx, 'dir': index[0], 'nbid': nbid, 'out_gen': output_gen, 'out_gt': output_gt})

        result = compare_answers_notebookcoder_new(output_gt, output_gen)
        all_results.append(result)
    # results_errors = [result for result in all_results if result['error']]
    outputs_errors = [output for result, output in zip(all_results, all_output) if result['error']]
    results_not_skip = [result for result in all_results if not result['skip']]
    error_counter = error_count(outputs_errors)
    print("### Execution Accuracy: \nAcc  \tErr\tErrRate")
    print(f"{round(100*sum([i['correct'] for i in results_not_skip])/len(results_not_skip), 2)}\t{len(outputs_errors)}\t{round(100*len(outputs_errors)/len(results_not_skip),2)}")
    print(f"### Number of errors: {len(outputs_errors)}/{len(results_not_skip)}, {round(100*len(outputs_errors)/len(results_not_skip), 2)}% ")
    print(error_counter)
    write_pickle(all_output, os.path.join(args.path_save_notebooks, 'temp_output.pkl'))
    write_pickle(all_results, os.path.join(args.path_save_notebooks, 'eval_results.pkl'))

# all_output = read_pickle(os.path.join(args.path_save_notebooks, 'temp_output.pkl'))
# all_results = []
# for idx, item in enumerate(all_output):
#     output_gt, output_gen = item['out_gt'], item['out_gen']
#     result = compare_answers_notebookcoder_new(output_gt, output_gen)
#     all_results.append(result)
#
# # results_errors = [result for result in all_results if result['error']]
# outputs_errors = [output for result, output in zip(all_results, all_output) if result['error']]
# results_not_skip = [result for result in all_results if not result['skip']]
# error_counter = error_count(outputs_errors)
# print("### Execution Accuracy: \nAcc\tErr\tErrRate")
# print(f"{round(100*sum([i['correct'] for i in results_not_skip])/len(results_not_skip), 2)}\t{len(outputs_errors)}\t{round(100*len(outputs_errors)/len(results_not_skip),2)}")
# print(f"### Number of errors: {len(outputs_errors)}/{len(results_not_skip)}, {round(100*len(outputs_errors)/len(results_not_skip), 2)}% ")
# print(error_counter)
# # write_pickle(all_output, os.path.join(args.path_save_notebooks, 'temp_output.pkl'))
# write_pickle(all_results, os.path.join(args.path_save_notebooks, 'eval_results.pkl'))
