
import pickle
def write_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data



# old_data = read_pickle('../dataset/test_1654.pkl')
new_data1 = read_pickle('../dataset-CSIO/synthesis_input_output.pkl')
new_data2 = read_pickle('../dataset-CSIO/task_data_dependency_qualified.pkl')

print()
