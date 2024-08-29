import argparse

from recbole_gnn.quick_start import run_recbole_gnn

import yaml
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--exp_name', type=str, default=None, help='Name of the exp')
    parser.add_argument('--config_dict', type=str, help='config dict.')

    args, _ = parser.parse_known_args()
    print("IDK",(json.loads(args.config_dict)))
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    entire_results = run_recbole_gnn(model=args.model, dataset=args.dataset, config_dict=args.config_dict)
    
    metrics = list(entire_results['test_result'].items())
    
    results = {}
    for item in metrics:
        results[str(item[0])] = float(item[1])
    results['num_params'] = int(entire_results['num_model_params'])

    with open(f'../exps/{args.exp_name}/{args.exp_name}_results.yaml', 'w+') as f :
        yaml.dump(results,f,sort_keys=False) 
