from collections import Counter


def compute_f1(pred_toks, gold_toks):
    try:
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
    except:
        common = [item_pred for item_pred in pred_toks if item_pred in gold_toks]
        num_same = len(common)
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_macro_f1(preds, gold_list):
    # preds: a list of list
    # gold_list: a list
    return max([compute_f1(pred_list, gold_list) for pred_list in preds])



def strict_correct_match(prediction_text, result):
    # return true if correct, else False
    correct = False
    # if result['answer_type'] == 'number':
    answers, predictions = [], []
    _truth = [prediction_text==result['ans1'], prediction_text==result['ans2'], prediction_text==result['ans3']]
    if any(_truth):
        answers = [result['ans{}'.format(idx)] for idx in range(1, 4) if len(result['ans{}'.format(idx)])>0]
        predictions = [[prediction_text]]
        correct = True
    elif prediction_text[-1] == '\n':
        _line_truth = [prediction_text[:-1]==result['ans1'], prediction_text[:-1]==result['ans2'], prediction_text[:-1]==result['ans3']]
        if any(_line_truth):
            answers = [result['ans{}'.format(idx)] for idx in range(1, 4) if len(result['ans{}'.format(idx)])>0]
            predictions = [[prediction_text[:-1]]]
            correct = True
    return {'strict_correct': correct,
            'correct': correct,
            'partial_correct': correct,
            'f1': 1.0 if correct else 0.0,
            'answers': answers,
            'predictions': predictions}



# def match_one_number(prediction_text, answer_text):
#     if prediction_text[-1] == '\n':
#         prediction_text = prediction_text[:-1]
#     if answer_text[-1] == '\n':
#         answer_text = answer_text[:-1]
#     try:
#         answer_digit = float(answer_text)
#         answer_digit = round(answer_digit, 2)
#         prediction_digit = float(prediction_text)
#         prediction_digit = round(prediction_digit, 2)
#         if answer_digit == prediction_digit:
#             return True
#         else:
#             return False
#     except:
#         return False


# def extract_one_number(number_string):
#     if len(number_string) == 0:
#         return False
#     if number_string[-1] == '\n':
#         number_string = number_string[:-1]
#     # TODO 检测number_stirng里面有没有数字float
#     try:
#         number = float(number_string)
#         number = round(number, 2)
#     except:
#         number = False
#     return number
