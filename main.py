import os
from tqdm import tqdm
import json
import argparse
from multiprocessing import Pool
from utils.common import check_dir, print_args, set_seed
from utils.data import read_file, load_data
from utils.crawl_from_GPT import get_response_from_message as get_response
import torch
from copy import deepcopy
from glob import glob
from sklearn.metrics import balanced_accuracy_score
from scripts.process import update_data

system = ""
image_key = ['image_ent', 'image_cnt']
prefix_path_direct = 'prefix/specific/direct/'
prefix_path_cot = 'prefix/specific/cot/'
prefix = {
    'direct': {'doc_key': 'document', 'sum_key': 'claim', 'template': prefix_path_direct},
    'cot': {'doc_key': 'document', 'sum_key': 'claim', 'template': prefix_path_cot},
    'SIFiD-entailment': {'doc_key': 'win3_line_mcnt', 'sum_key': 'claim', 'template': prefix_path_direct},
    'SIFiD-entailment-cot': {'doc_key': 'win3_line_mcnt', 'sum_key': 'claim', 'template': prefix_path_cot},
    'SIFiD-similarity': {'doc_key': 'win3_line_sim', 'sum_key': 'claim', 'template': prefix_path_direct},
    'SIFiD-similarity-cot': {'doc_key': 'win3_line_sim', 'sum_key': 'claim', 'template': prefix_path_cot},
}


def check_ans(resp):
    if 'Answer: Yes' in resp:
        return 1
    elif 'Answer: No' in resp:
        return 0
    flag_yes = False
    flag_no = False
    for yes in ['Yes', 'YES']:
        if yes in resp:
            flag_yes = True
            break
    for no in ['No', 'NO']:
        if no in resp:
            flag_no = True
            break
    if flag_yes and flag_no:
        pred = -1
    elif flag_yes:
        pred = 1
    elif flag_no:
        pred = 0
    else:
        pred = -1
    return pred


def judge_sample(args):
    sample, model_name, tmp_file, cot = args
    name = sample['dataset']
    for model in sample['prediction']:
        for method in sample['prediction'][model]:
            template = read_file(f"{prefix[method]['template']}{name}.txt")
            doc_key = prefix[method]['doc_key']
            sum_key = prefix[method]['sum_key']

            if sample['prediction'][model][method]['isConsistent'] is not None:
                continue
            if sample[doc_key] == '*EMPTY_SENT*':
                sample['prediction'][model][method]['resp'] = '*EMPTY_SENT*'
                sample['prediction'][model][method]['isConsistent'] = 0
            else:
                prompt = template.replace('{{  Article  }}', sample[doc_key]).replace('{{  Summary  }}',
                                                                                      sample[sum_key])
                message = [{'role': 'user', 'content': prompt}]
                resp = get_response(message, model_name, wait=10, timeout=60)
                while '\"error\"' in resp:
                    resp = get_response(message, model_name, wait=10, timeout=60)
                pred = check_ans(resp)
                sample['prediction'][model][method]['resp'] = resp
                sample['prediction'][model][method]['isConsistent'] = pred
    with open(tmp_file, 'a') as file:
        file.write(json.dumps(sample) + '\n')
    return sample


def judge(data, arg, name):
    cut = arg.cut
    tmp_file = arg.tmp_file.replace('[dataset]', f"{name}_{cut}")
    args_list = [(sample, arg.model_name, tmp_file, arg.cot) for index, sample in enumerate(data)]
    results = []
    with open(tmp_file, 'w') as file:
        pass
    with Pool(arg.num_workers) as pool:
        # Prepare the pool tasks
        tasks = [pool.apply_async(judge_sample, args=(_arg,)) for _arg in args_list]
        # Use tqdm to track progress
        for task in tqdm(tasks, total=len(tasks)):
            result = task.get()
            results.append(result)
    return results


def run(arg):
    # data version, used for naming
    dataset = arg.dataset
    # load data
    if '/' not in dataset:
        path = f'data/processed/{dataset}_{arg.nli_model}_{arg.cut}.torch'
    else:
        path = dataset

    if arg.from_result:
        path = glob(f'result/SummaC-0.1/{model}*')
        data = [load_data(p) for p in path]
        data = {data[name][0]['dataset']: data[name] for name in data}
    else:
        data = torch.load(path)

    # best_th = torch.load('data/best_th.torch')
    arg.specific = arg.specific.split(',') if arg.specific != '' else [name for name in data]
    print(f"Evaluate List: {arg.specific}")
    for name in data:
        if name not in arg.specific: continue
        benchmark = update_data(data[name], prefix, name, arg)
        print(f"Evaluating {name}...")
        result = judge(benchmark, arg, name)
        json.dump(result, open(arg.save_path.replace('[dataset]', f"{name}_{arg.cut}"), 'w'), indent=2)
        print(f"--------------{name}--------------")
        check_acc(result)


def check_acc(result):
    label = [sample['label'] for sample in result]
    pred = {model_name: {method: [] for method in result[0]['prediction'][model_name]}
            for model_name in result[0]['prediction']}
    miss = {model_name: {method: [] for method in result[0]['prediction'][model_name]}
            for model_name in result[0]['prediction']}
    acc = deepcopy(pred)
    for idx, sample in enumerate(result):
        all_pred = sample['prediction']
        for model_name in all_pred:
            for method in all_pred[model_name]:
                pred[model_name][method].append(all_pred[model_name][method]['isConsistent']==1)
                miss[model_name][method].append(all_pred[model_name][method]['isConsistent']==-1)
                acc[model_name][method].append(all_pred[model_name][method]['isConsistent'] == label[idx])
    for model_name in acc:
        for method in acc[model_name]:
            acc_score = round(sum(acc[model_name][method]) / len(acc[model_name][method]), 4) * 100
            bacc_score = round(balanced_accuracy_score(label, pred[model_name][method]), 4) * 100
            miss_num =  sum(miss[model_name][method])
            print(f"{model_name}-{method}:\t{bacc_score} \tmiss:{miss_num}")


if __name__ == "__main__":
    set_seed(123)
    parser = argparse.ArgumentParser(description='xxxxx')
    parser.add_argument('--dataset', type=str, default='SummaC')
    parser.add_argument('--nli_model', type=str, default='vitc')
    parser.add_argument('--method', type=str, default='sim')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--th', type=float, default=0.0)
    parser.add_argument('--th_cos', type=float, default=0.5)
    parser.add_argument('--model_name', type=str, default='gpt-4-1106-preview')
    # parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--save_path', type=str, default='tmp')
    parser.add_argument('--tmp_file', type=str, default='tmp')
    parser.add_argument('--cut', type=str, default='test')
    parser.add_argument('--specific', type=str, default='')
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--filter_doc', action='store_true')
    parser.add_argument('--from_result', action='store_true')
    arguments = parser.parse_args()
    # start
    arguments.specific = 'polytope'
    # arguments.from_result = True
    for model in [arguments.model_name]:
        arguments.model_name = model
        dataset = arguments.dataset
        if '/' in dataset:
            dataset = dataset.split('/')[-1].split('.')[0]
        arguments.dataname = dataset
        # if arguments.save_path == 'tmp':
        arguments.save_path = f'result_specific/{dataset}-{arguments.th}/{arguments.model_name}_[dataset].json'
        # if arguments.tmp_file == 'tmp':
        arguments.tmp_file = f'result_specific/cache/{dataset}-{arguments.th}/{arguments.model_name}_[dataset].json'
        check_dir(arguments.save_path)
        check_dir(arguments.tmp_file)
        print_args(arguments)
        run(arguments)
