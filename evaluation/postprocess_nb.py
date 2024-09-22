import numpy as np
import pandas as pd
from collections import Counter
from pandas.testing import assert_frame_equal


def get_code_string(cell):
    source = cell.get("source", [])
    source = [] if source is None else source
    if type(source) == str:
        source = source.split('\n')
        source = [s+'\n' if i<len(source)-1 else s for i, s in enumerate(source)]
    return ''.join(source)


def obtain_cell_output(notebook, row_id):
    cell1_source = get_code_string(notebook['cells'][1])
    if cell1_source == '%load_ext dumpvar_0_extension':
        row_id = row_id+2
    cell = notebook['cells'][row_id]
    output = {'exception': cell['metadata']['papermill']['exception'],
              'output_type': "",
              'text': "",
              'data': pd.DataFrame([]),
              'code': cell['source']}

    if not output['exception']:
        if len(cell.get('outputs', [])) > 0:
            output_types = [out['output_type'] for out in cell['outputs']]
            # print("{}, {}".format(len(output_types), output_types))
            # if 'stream' in output_types:
            #     output['output_type'] = 'stream'
            #     output['text'] = cell['outputs'][output_types.index('stream')].get('text', [])
            if 'execute_result' in output_types:
                output['output_type'] = 'execute_result'
                item = cell['outputs'][output_types.index('execute_result')]
                if 'text/html' in item['data'] and 'text/plain' in item['data']:
                    try:
                        output['output_type'] = 'dataframe'
                        # dfs = pd.read_html(item['data']['text/html'])
                        dfs = pd.read_html(item['data']['text/html'], flavor='html5lib')
                        # TODO: pd.read_html will yield ImportError: lxml not found, please install it,
                        if len(dfs) >= 1 and isinstance(dfs[0], pd.DataFrame):
                            output['data'] = dfs[0]
                    except Exception as e:
                        output['output_type'] = 'error'
                        output['exception'] = e

            elif 'error' in output_types:
                output['output_type'] = 'error'
                _index = 0
                if len(cell['outputs']) > 0:
                    for _i, _out in enumerate(cell['outputs']):
                        if 'ename' in _out and 'evalue' in _out:
                            _index = _i
                            break
                try:
                    output['text'] = cell['outputs'][_index]['ename'] + ': ' + cell['outputs'][_index]['evalue']
                except:
                    output['text'] = ''
            else:
                output['output_type'] = ''.join(output_types)
                output['text'] = ''
        else:
            output['output_type'] = 'None'
            output['text'] = ''
    else:
        output['output_type'] = 'error'
        _index = 0
        if len(cell['outputs']) > 0:
            for _i, _out in enumerate(cell['outputs']):
                if 'ename' in _out and 'evalue' in _out:
                    _index = _i
                    break
            try:
                output['text'] = cell['outputs'][_index]['ename'] + ': ' + cell['outputs'][_index]['evalue']
            except:
                output['text'] = cell['outputs'][_index]
        else:
            output['text'] = 'no error output'
    return output


def compare_answers_notebookcoder_new(output_gt, output_gen):
    result = {
              'skip': False,
              'error': False,
              'correct': False,
              'exception': "",
              'output_not_df': False,
              'answer': output_gt['data'],
              'prediction': output_gen['data'],
    }
    if output_gt['data'].shape[0] == 0:
        result['skip'] = True
        return result

    if output_gen['output_type'] == 'error':
        # if have error (exception), then not correct
        result['error'] = True
    elif output_gen['output_type'] == 'dataframe':
        pre = output_gen['data']
        ans = output_gt['data']

        # 保留两位小数，避免小数位数过多引起的float类型匹配错误
        pre, ans = pre.round(2), ans.round(2)

        # # 如果index是multiindex有两层，那就拉平成一层
        # if isinstance(pre.columns, pd.MultiIndex):
        #     pre.columns = [a if 'Unname' not in a else b for a, b in pre.columns]
        # if isinstance(ans.columns, pd.MultiIndex):
        #     ans.columns = [a if 'Unname' not in a else b for a, b in ans.columns]
        # if isinstance(pre.columns, pd.RangeIndex):
        #     pre.columns = list(range(pre.columns.start, pre.columns.stop, pre.columns.step))
        # if isinstance(ans.columns, pd.RangeIndex):
        #     ans.columns = list(range(ans.columns.start, ans.columns.stop, ans.columns.step))
        # if isinstance(pre.columns, pd.Int64Index):
        #     pre.columns = [str(i) for i in pre.columns.tolist()]
        # if isinstance(ans.columns, pd.Int64Index):
        #     ans.columns = [str(i) for i in ans.columns.tolist()]

        # answer和prediction有时候会多一列最前面的index列，如果有的话，去掉
        if len(pre.columns) > 0 and 'Unnamed: 0' == pre.columns[0]:
            pre = pre.iloc[:, 1:]
        if len(ans.columns) > 0 and 'Unnamed: 0' == ans.columns[0]:
            ans = ans.iloc[:, 1:]

        # answer和prediction有时候会多一列的...列，如果有的话，去掉
        if '...' in pre.columns:
            pre = pre.drop('...', axis=1)
        if '...' in ans.columns:
            ans = ans.drop('...', axis=1)

        # 有时候ans有很多行，而prediction是用head搞出来的，就只有5或者几行，就sample一下
        if pre.shape[0] != ans.shape[0]:
            pre_line = pre.shape[0]
            ans = ans[:pre_line]
            ans1 = ans[:pre_line]
            ans2 = ans[-pre_line:]

        # MultiIndex两层搞成一层
        truth = pre._stat_axis.values != ans._stat_axis.values
        if (isinstance(truth, np.ndarray) and any(truth)) or (isinstance(truth, bool) and bool):
            pre.rename(index={p: a for p, a in zip(pre._stat_axis.values, ans._stat_axis.values)}, inplace=True)

        if pre.shape[1] != ans.shape[1]:
            try:
                ans = ans[pre.columns]
            except:
                ans = ans

        if pre.shape == ans.shape:
            # ans中的datetime等类型数据，在pre中显示的是string，所以要做一个类型转换
            try:
                pre = pre.astype(ans.dtypes)
                idx = pre.select_dtypes(include='object').columns
                pre[idx] = pre[idx].replace(to_replace=r'\s+', value=' ', regex=True)
                idx = ans.select_dtypes(include='object').columns
                ans[idx] = ans[idx].replace(to_replace=r'\s+', value=' ', regex=True)
                ans = ans.astype(pre.dtypes)
            except Exception as e:
                result['exception'] = e
        try:
            assert_frame_equal(pre, ans)
            result['correct'] = True
        except Exception as e:
            result['exception'] = e
            result['correct'] = False
        result['answer'] = ans
        result['prediction'] = pre
    else:
        # print(this_index['output_type'])
        result['output_not_df'] = True
    return result


def compare_answers_notebookcoder(this_index):
    assert this_index['nbid'] == this_index['nbid']
    result = {'idx': this_index['nbid'],
              'answer_type': this_index['answer_type'],
              'error': False,
              'correct': False,
              'exception': "",
              'output_not_df': False,
              'answer': this_index['ans2'],
              'prediction': this_index['data'],
              'cmp_answer': this_index['ans2'],
              'cmp_prediction': this_index['data'], }
    answers, predictions = [], []
    if this_index['output_type'] == 'error':
        # if have error (exception), then not correct
        result['error'] = True
    elif this_index['output_type'] == 'dataframe':
        pre = this_index['data']
        ans = this_index['ans2']
        # 保留两位小数，避免小数位数过多引起的float类型匹配错误
        pre, ans = pre.round(2), ans.round(2)

        # 如果index是multiindex有两层，那就拉平成一层
        if isinstance(pre.columns, pd.MultiIndex):
            pre.columns = [a if 'Unname' not in a else b for a, b in pre.columns]
        if isinstance(ans.columns, pd.MultiIndex):
            ans.columns = [a if 'Unname' not in a else b for a, b in ans.columns]
        if isinstance(pre.columns, pd.RangeIndex):
            pre.columns = list(range(pre.columns.start, pre.columns.stop, pre.columns.step))
        if isinstance(ans.columns, pd.RangeIndex):
            ans.columns = list(range(ans.columns.start, ans.columns.stop, ans.columns.step))
        if isinstance(pre.columns, pd.Int64Index):
            pre.columns = [str(i) for i in pre.columns.tolist()]
        if isinstance(ans.columns, pd.Int64Index):
            ans.columns = [str(i) for i in ans.columns.tolist()]

        # answer和prediction有时候会多一列最前面的index列，如果有的话，去掉
        if len(pre.columns) > 0 and 'Unnamed: 0' == pre.columns[0]:
            pre = pre.iloc[:, 1:]
        if len(ans.columns) > 0 and 'Unnamed: 0' == ans.columns[0]:
            ans = ans.iloc[:, 1:]

        # answer和prediction有时候会多一列的...列，如果有的话，去掉
        if '...' in pre.columns:
            pre = pre.drop('...', axis=1)
        if '...' in ans.columns:
            ans = ans.drop('...', axis=1)

        # 有时候ans有很多行，而prediction是用head搞出来的，就只有5或者几行，就sample一下
        if pre.shape[0] != ans.shape[0]:
            pre_line = pre.shape[0]
            ans = ans[:pre_line]
            ans1 = ans[:pre_line]
            ans2 = ans[-pre_line:]

        # MultiIndex两层搞成一层
        truth = pre._stat_axis.values != ans._stat_axis.values
        if (isinstance(truth, np.ndarray) and any(truth)) or (isinstance(truth, bool) and bool):
            pre.rename(index={p: a for p, a in zip(pre._stat_axis.values, ans._stat_axis.values)}, inplace=True)

        if pre.shape[1] != ans.shape[1]:
            try:
                ans = ans[pre.columns]
            except:
                ans = ans

        if pre.shape == ans.shape:
            # ans中的datetime等类型数据，在pre中显示的是string，所以要做一个类型转换
            try:
                pre = pre.astype(ans.dtypes)
                idx = pre.select_dtypes(include='object').columns
                pre[idx] = pre[idx].replace(to_replace=r'\s+', value=' ', regex=True)
                idx = ans.select_dtypes(include='object').columns
                ans[idx] = ans[idx].replace(to_replace=r'\s+', value=' ', regex=True)
                ans = ans.astype(pre.dtypes)
            except Exception as e:
                result['exception'] = e
        try:
            assert_frame_equal(pre, ans)
            result['correct'] = True
        except Exception as e:
            result['exception'] = e
            result['correct'] = False
        result['cmp_answer'] = ans
        result['cmp_prediction'] = pre
    else:
        # print(this_index['output_type'])
        result['output_not_df'] = True
    return result


def error_statistic(indexes):
    error_counter = Counter()
    for this_index in indexes:
        if this_index['text']:
            error_name = this_index['text'].split(': ')[0]
            if error_name == 'NameError':
                error_counter['VarAPINotDefined'] += 1
            elif error_name == 'SyntaxError':
                error_counter['InvalidSyntax'] += 1
        else:
            error_counter['NoMessage'] += 1


def error_count(results):
    error_counter = Counter()
    for result in results:
        output_gen = result['out_gen']
        if output_gen['text'] and type(output_gen['text'])==str:
            error_name = output_gen['text'].split(': ')[0]
            error_counter[error_name] += 1
        else:
            error_counter['NoMessage'] += 1
    return error_counter

