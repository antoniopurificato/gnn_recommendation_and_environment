import os
import yaml
from codecarbon import EmissionsTracker
import argparse
from recbole_gnn.quick_start import run_recbole_gnn

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', '-m', type=str, nargs='+', default=['LightGCN'], help='name of models')
    parser.add_argument('--datasets', '-d', type=str, nargs='+', default=['ml-1m', 'dianping', 'amazon_beauty'], help='name of datasets')
    parser.add_argument('--epochs', '-e', type=int, nargs='+', default=[400], help='number of epochs')
    parser.add_argument('--embedding_sizes', '-emb', type=int, nargs='+', default=[32,64,128,256], help='embedding sizes')
    parser.add_argument('--topk', type=int, default=10, help='top k')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    return parser.parse_args()
    
if __name__ == '__main__':    
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    output_config = {"topk" : args.topk,
                     "gpu_id": args.gpu_id,
                     "eval_step": 100}
    file_name = os.getcwd() + '/../config.yaml'
    for model in args.models:
        for dataset in args.datasets:
            for number_of_epochs in args.epochs:
                for emb_size in args.embedding_sizes:
                    already_done = False

                    output_config["epochs"] = number_of_epochs
                    output_config["model"] = model
                    output_config["dataset"] = dataset
                    output_config["embedding_size"] = emb_size

                    experiment_name = f'{model}_{number_of_epochs}_{dataset}_emb_{emb_size}'
                    
                    if os.path.exists(os.getcwd() + f'/../exps/{experiment_name}/{experiment_name}_results.yaml'):
                        print(f"Experiment {experiment_name} already done.")
                        already_done = True
                    else:
                        if not os.path.exists(os.getcwd() + f'/../exps/{experiment_name}'):
                            os.mkdir(os.getcwd() + f'/../exps/{experiment_name}')
                    if not already_done:
                        output_config["exp_name"] = experiment_name
                        tracker = EmissionsTracker(project_name="GNN Recommendation",
                                output_dir = f"../exps/{experiment_name}" ,
                                experiment_id = experiment_name,
                                log_level = 'critical',
                                tracking_mode = 'process')
                        tracker.start()

                        entire_results = run_recbole_gnn(model=model, dataset=dataset, config_dict=output_config)
                        tracker.stop()
                        metrics = list(entire_results['test_result'].items())
                        
                        results = {}
                        for item in metrics:
                            results[str(item[0])] = float(item[1])
                        results['num_params'] = int(entire_results['num_model_params'])

                        with open(f'../exps/{experiment_name}/{experiment_name}_results.yaml', 'w+') as f :
                            yaml.dump(results,f,sort_keys=False) 