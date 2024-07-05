from copy import deepcopy
import numpy as np


def update_matrix(sample):
    # add 'prediction' section to each sample
    image_ent = sample['image_ent']
    image_cnt = sample['image_cnt']
    similarity = sample['similarity']
    if 'processed' not in sample:
        # cal scores
        sample['processed'] = True
        entail = image_ent.max(-1).tolist()
        entail_mcnt = (image_ent - image_cnt).max(-1).tolist()
        sim = similarity.max(-1).tolist()
        if type(entail) != list: entail = [entail]
        if type(entail_mcnt) != list: entail_mcnt = [entail_mcnt]
        if type(sim) != list: sim = [sim]
        sample['entail'] = entail
        sample['sentiment'] = sim
        sample['entail_mcnt'] = entail_mcnt
        sample['image_ent'] = sample['image_ent'].tolist()
        sample['image_cnt'] = sample['image_cnt'].tolist()
        sample['similarity'] = sample['similarity'].tolist()
        if type(sample['image_ent']) != list: sample['image_ent'] = [sample['image_ent']]
        if type(sample['image_cnt']) != list: sample['image_cnt'] = [sample['image_cnt']]
    return sample


def join_sents(sent_list, selection, fill_empty_with='*EMPTY_SENT*'):
    joined = '\n'.join([sent_list[i] for i in selection])
    if joined == '':
        joined = fill_empty_with
    return joined


def filter_sent_by_threshold(scores, th):
    return [i for i, s in enumerate(scores) if s > th]


def filter_sent_by_percentage(scores, percentage=None, num=None):
    if num is None:
        num = int(len(scores) * percentage / 100)
        if num == 0:
            num = 1
    idx = np.argsort(scores)[-num:].tolist()
    idx.sort()
    return idx


def extend_idx(index_list: list, size, length):
    index = np.array(index_list)
    ext_pos = np.concatenate([index + (i + 1) for i in range(size)])
    ext_neg = np.concatenate([index - (i + 1) for i in range(size)])
    ext = np.concatenate([ext_pos, index, ext_neg]).clip(0, length - 1)
    ext = np.unique(ext)
    return ext.tolist()


def update_data(_benchmark, _prefix, name, arg):
    # filtered by threshold
    new_benchmark = []
    for index, sample in enumerate(_benchmark):
        # extract values
        sample = update_matrix(sample)
        # creat new sample
        new_sample = {'id': index, "name": name, 'prediction': {}}
        new_sample.update(sample)
        if arg.model_name not in new_sample['prediction']:
            new_sample['prediction'][f"{arg.model_name}"] = {}
        sample = new_sample
        # set the result of each method in prefix diction to default
        para = {key: {'resp': None, 'isConsistent': None, 'th': arg.th, 'th_cos': arg.th_cos} for key in _prefix}
        sample['prediction'][f"{arg.model_name}"].update(para)
        # form input
        entail = sample['entail']
        sim = sample['sentiment']
        entail_mcnt = sample['entail_mcnt']
        doc_sent = sample['doc_sent']
        sum_sent = sample['sum_sent']

        filter_index = filter_sent_by_threshold(entail, arg.th)
        filter_index_mcnt = filter_sent_by_threshold(entail_mcnt, arg.th)

        # and with window size 1
        win3_index = extend_idx(filter_index, 1, len(entail))
        win3_index_mcnt = extend_idx(filter_index_mcnt, 1, len(entail_mcnt))

        # update reformed inputs
        sample['fdoc_line'] = join_sents(doc_sent, filter_index)
        sample['win3_line'] = join_sents(doc_sent, win3_index)
        sample['fdoc_line_mcnt'] = join_sents(doc_sent, filter_index_mcnt)
        sample['win3_line_mcnt'] = join_sents(doc_sent, win3_index_mcnt)
        sample['sent_line'] = join_sents(doc_sent, range(len(sum_sent)))

        filter_index_sim = filter_sent_by_threshold(sim, arg.th_cos)
        win3_index_sim = extend_idx(filter_index_sim, 1, len(sim))
        sample['fdoc_line_sim'] = join_sents(doc_sent, filter_index_sim)
        sample['win3_line_sim'] = join_sents(doc_sent, win3_index_sim)


        # filter_index_mcnt_20 = filter_sent_by_percentage(entail_mcnt, percentage=20)
        # win3_index_mcnt_20 = extend_idx(filter_index_mcnt_20, 1, len(entail_mcnt))
        # filter_index_mcnt_same_as_win = filter_sent_by_percentage(entail_mcnt, num=len(win3_index_mcnt_20))
        #
        # sample['fdoc_line_mcnt_20'] = join_sents(doc_sent, filter_index_mcnt_20)
        # sample['win3_line_mcnt_20'] = join_sents(doc_sent, win3_index_mcnt_20)
        # sample['fdoc_line_mcnt_same_as_win'] = join_sents(doc_sent, filter_index_mcnt_same_as_win)

        new_benchmark.append(sample)
    return new_benchmark
