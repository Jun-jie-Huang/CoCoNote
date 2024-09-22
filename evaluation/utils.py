import json
def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=1)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


import pickle
def write_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


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
