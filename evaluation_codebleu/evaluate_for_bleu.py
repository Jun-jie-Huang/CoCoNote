import json
import os
import re

def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp)
def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data
def read_jsonl(src_filename):
    data = []
    with open(src_filename) as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
    return data
def write_jsonl(data, output_filename):
    lines = ""
    for d in data:
        s = json.dumps(d)
        lines += s + "\n"
    with open(output_filename, "wt") as f:
        f.write(lines)
import csv
def csv_reader(path):
    with open(path, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp)
        data = [i for i in reader]
    return data
def csv_writer(path, header, data):
    with open(path, 'w', encoding='utf-8', newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)


def cleaning_string(string):
    if string:
        string = re.sub(r'\n[ \n\t]*\n', r'\n', string)  # remove extra \n\n
        # string = re.sub(' +', ' ', string)  # remove extra space
        # string = re.sub("\'", "\"", string)
        string = re.sub("\"", "\'", string)
        return string
    else:
        return ''

### create examples for evaluate.py (Dev set)
result_data = read_jsonl('./inference_results/codex_dev_samples.0.jsonl')
print(len(result_data))
for i in range(5):
    split_generation_results = []
    for idx, item in enumerate(result_data):
        target = item['metadata']
        prediction = item['samples'][i]
        _input = item['context']
        split_generation_results.append({"input": _input, "target": target, "generation": prediction, 'idx': idx})
    if not os.path.exists('./results/result_dev{}'.format(i)):
        os.makedirs('./results/result_dev{}'.format(i))
    print(len(split_generation_results))
    write_json(split_generation_results, './results/result_dev{}/split_generation_results.json'.format(i))


### create examples for evaluate.py (Test set)
result_data = read_jsonl('./inference_results/codex_test_samples.0.jsonl')
print(len(result_data))
for i in range(5):
    split_generation_results = []
    for idx, item in enumerate(result_data):
        target = item['metadata']
        prediction = item['samples'][i]
        _input = item['context']
        split_generation_results.append({"input": _input, "target": target, "generation": prediction, 'idx': idx})
    if not os.path.exists('./results/result_test{}'.format(i)):
        os.makedirs('./results/result_test{}'.format(i))
    print(len(split_generation_results))
    write_json(split_generation_results, './results/result_test{}/split_generation_results.json'.format(i))





# ### evaluate script:
# python ./CodeBLEU/evaluate.py --generation_dir ./results/result_dev0 2>&1 |tee ./logs/evaluate_dev0.log
# python ./CodeBLEU/evaluate.py --generation_dir ./results/result_dev1 2>&1 |tee ./logs/evaluate_dev1.log
# python ./CodeBLEU/evaluate.py --generation_dir ./results/result_dev2 2>&1 |tee ./logs/evaluate_dev2.log
# python ./CodeBLEU/evaluate.py --generation_dir ./results/result_dev3 2>&1 |tee ./logs/evaluate_dev3.log
# python ./CodeBLEU/evaluate.py --generation_dir ./results/result_dev4 2>&1 |tee ./logs/evaluate_dev4.log
# python ./CodeBLEU/evaluate.py --generation_dir ./results/result_test0 2>&1 |tee ./logs/evaluate_test0.log
# python ./CodeBLEU/evaluate.py --generation_dir ./results/result_test1 2>&1 |tee ./logs/evaluate_test1.log
# python ./CodeBLEU/evaluate.py --generation_dir ./results/result_test2 2>&1 |tee ./logs/evaluate_test2.log
# python ./CodeBLEU/evaluate.py --generation_dir ./results/result_test3 2>&1 |tee ./logs/evaluate_test3.log
# python ./CodeBLEU/evaluate.py --generation_dir ./results/result_test4 2>&1 |tee ./logs/evaluate_test4.log

# tail -2 ./logs/evaluate_dev0.log
# tail -2 ./logs/evaluate_dev1.log
# tail -2 ./logs/evaluate_dev2.log
# tail -2 ./logs/evaluate_dev3.log
# tail -2 ./logs/evaluate_dev4.log
# tail -2 ./logs/evaluate_test0.log
# tail -2 ./logs/evaluate_test1.log
# tail -2 ./logs/evaluate_test2.log
# tail -2 ./logs/evaluate_test3.log
# tail -2 ./logs/evaluate_test4.log
