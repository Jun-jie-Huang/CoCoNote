import os
import re
import sys
import traceback
sys.path.append('./papermill/')
import papermill as pm
import argparse
import json
import pandas as pd
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


# process_id = int(sys.argv[4])
def process_args(args):
    args.error_log_filename = os.path.join(args.base_dir, f'./error_{args.process_id}.log')
    args.progress_log = open(os.path.join(args.base_dir, f'./progress_{args.process_id}.log'), 'a+')
    args.failed_log = open(os.path.join(args.base_dir, f'failed_files_{args.process_id}.log'), 'a+')
    args.missing_file_log = open(os.path.join(args.base_dir, f'missing_file_{args.process_id}.log'), 'a+')
    return args


# def try_run_one_notebook(repo_path, nb_name, nbid):
def try_run_one_notebook(kv):
    repo_path, nb_name, nbid = kv
    # os.system('touch /home/azureuser/temp/object_ids_{}.txt'.format(args.process_id))
    row_id = int(nbid.split('R')[1])
    exec_param = pm.ExecutionParam(args.error_log_filename, 1000, repo_path, True, args.progress_log,
                                   args.missing_file_log, args.process_id, nbid)
    fname = os.path.join(repo_path, nb_name)

    # fix kernel
    content = json.load(open(fname, 'r'))
    if content["metadata"]["kernelspec"]["name"] not in ['python2', 'python3']:
        orig_name = content["metadata"]["kernelspec"]["name"]
        if content["metadata"]['language_info']['version'] == 2:
            kernel = 'python2'
        else:
            kernel = 'python3'
        print("replace kernal from {} to {}".format(orig_name, kernel))
        nb_content = ''.join([line for line in open(fname, 'r')])
        nb_content = nb_content.replace('"name": "{}"'.format(orig_name), '"name": "{}"'.format(kernel))
        with open(fname, 'w') as fpw:
            fpw.write(nb_content)

    # nb_name = nb_name.split('/')[-1]
    # nb_name = os.path.basename(nb_name)
    print("running {}".format(fname))
    output_notebook = os.path.join(repo_path, 'run_{}'.format(nb_name))
    try:
        rt = pm.execute_notebook(fname, output_notebook, execution_param=exec_param)
    except Exception as e:
        args.failed_log.write('FAIL {}: error = {}\n'.format(fname, str(e)))
        traceback.print_exc(file=args.failed_log)
        print("Error: {}".format(e))
        traceback.print_exc(file=sys.stdout)
        args.failed_log.flush()
    args.progress_log.flush()
    # os.system('rm /home/azureuser/temp/object_ids_{}.txt'.format(args.process_id))
    return


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='../notebooks/', type=str)
parser.add_argument('--idx_path', default='index.pkl', type=str)
parser.add_argument('--process_id', default=0, type=int)
# parser.add_argument('--replace_base', action='store_true')
args = parser.parse_args()
args = process_args(args)

idx2dir = read_json(args.idx_path)
n_threads = 8
with Pool(n_threads) as p:
    func_ = partial(try_run_one_notebook, )
    all_results = list(tqdm(p.imap(func_, idx2dir, chunksize=16), total=len(idx2dir), desc="### Executing: ",))
    print("number of output data {}".format(len(all_results)))
# all_results = []
# for item in tqdm(idx2dir, desc="### Executing: "):
#     result = try_run_one_notebook(item)


