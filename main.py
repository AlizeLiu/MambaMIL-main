from __future__ import print_function

import argparse
import os

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from dataset.dataset_generic import Generic_MIL_Dataset

# pytorch imports
import torch
import pandas as pd
import numpy as np
import wandb
# torch.use_deterministic_algorithms(True)
def main(args):
    #TODO: add wandb logging
    os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project=args.task)
    wandb.config.update(args)
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(args.backbone, args.patch_size, from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        if args.preloading == 'yes':
            for d in datasets:
                d.pre_loading()
            
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    mean_auc_test = final_df['test_auc'].mean()
    std_auc_test = final_df['test_auc'].std()
    mean_auc_val = final_df['val_auc'].mean()
    std_auc_val = final_df['val_auc'].std()
    mean_acc_test = final_df['test_acc'].mean()
    std_acc_test = final_df['test_acc'].std()
    mean_acc_val = final_df['val_acc'].mean()
    std_acc_val = final_df['val_acc'].std()

    wandb.log({"mean_auc_test": mean_auc_test, "std_auc_test": std_auc_test, "mean_auc_val": mean_auc_val, "std_auc_val": std_auc_val})
    df_append = pd.DataFrame({
        'folds': ['mean', 'std'],
        'test_auc': [mean_auc_test, std_auc_test],
        'val_auc': [mean_auc_val, std_auc_val],
        'test_acc': [mean_acc_test, std_acc_test],
        'val_acc': [mean_acc_val, std_acc_val],
    })
    final_df = pd.concat([final_df, df_append])
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    final_df['folds'] = final_df['folds'].astype(str)
    table = wandb.Table(dataframe=final_df)
    wandb.log({"summary": table})
    wandb.log({"mean_auc_test": mean_auc_test, "mean_acc_test": mean_acc_test, "mean_auc_val": mean_auc_val, "mean_acc_val": mean_acc_val})

    # Save mean ROC across folds if eval artifacts enabled
    if getattr(args, 'save_eval_artifacts', False) and getattr(args, 'plot_roc', False):
        from utils.eval_utils import compute_binary_roc, plot_mean_roc, save_summary_metrics
        artifact_dir = getattr(args, 'eval_artifact_dir', None) or args.results_dir
        fold_fprs = []
        fold_tprs = []
        fold_aucs = []
        all_fold_metrics = []
        for fold_idx in folds:
            pred_csv = os.path.join(artifact_dir, 'eval_artifacts', f'fold_{fold_idx}', 'test_predictions.csv')
            metrics_json = os.path.join(artifact_dir, 'eval_artifacts', f'fold_{fold_idx}', 'test_metrics.json')
            if os.path.exists(pred_csv):
                pred_df = pd.read_csv(pred_csv)
                y_true = pred_df['label'].values
                y_prob = pred_df['prob_1'].values
                fpr, tpr, _, auc = compute_binary_roc(y_true, y_prob)
                fold_fprs.append(fpr)
                fold_tprs.append(tpr)
                fold_aucs.append(auc)
            if os.path.exists(metrics_json):
                import json
                with open(metrics_json) as f:
                    all_fold_metrics.append(json.load(f))
        if fold_fprs:
            mean_roc_path = os.path.join(artifact_dir, 'eval_artifacts', 'mean_roc.png')
            plot_mean_roc(fold_fprs, fold_tprs, fold_aucs, mean_roc_path)
            print(f'Mean ROC saved to {mean_roc_path}')
        if all_fold_metrics:
            summary_csv = save_summary_metrics(all_fold_metrics, artifact_dir, split_name='test')
            print(f'Summary metrics saved to {summary_csv}')


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='enable dropout (p=0.25)')
parser.add_argument('--model_type', type=str, default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str)
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--patch_size', type=str, default='')
parser.add_argument('--preloading', type=str, default='no')
parser.add_argument('--in_dim', type=int, default=1024)

## mambamil

parser.add_argument('--mambamil_rate',type=int, default=10, help='mambamil_rate')
parser.add_argument('--mambamil_layer',type=int, default=2, help='mambamil_layer')
parser.add_argument('--mambamil_type',type=str, default='SRMamba', choices= ['Mamba', 'BiMamba', 'SRMamba'], help='mambamil_type')

# 允许外部传入 csv 路径
parser.add_argument('--csv_path', type=str, default=None, help='path to the dataset csv file')

## IHG-Mamba architecture params

parser.add_argument('--hidden_dim', type=int, default=256,
                    help='hidden dimension for IHG-Mamba')
parser.add_argument('--max_seq_len', type=int, default=2500,
                    help='maximum number of patches per WSI; <=0 means no truncation')
parser.add_argument('--feature_subdir', type=str, default='pt_files',
                    help='subdirectory under data_root_dir for pt features (default: pt_files)')
parser.add_argument('--features_already_hilbert', dest='features_already_hilbert',
                    action='store_true', default=False,
                    help='input .pt features are already Hilbert ordered')
parser.add_argument('--features_not_hilbert', dest='features_already_hilbert',
                    action='store_false',
                    help='input .pt features are NOT Hilbert ordered (use with --use_hilbert_index)')
parser.add_argument('--use_hilbert_index', action='store_true', default=False,
                    help='load hilbert/*.pt index and reorder features inside DataLoader')
parser.add_argument('--hilbert_index_dir', type=str, default=None,
                    help='directory containing hilbert index files (default: None, uses data_dir/hilbert)')
parser.add_argument('--sampling_mode', type=str, default='random_points',
                    choices=['random_points', 'uniform_points', 'chunk'],
                    help='Patch sampling mode: random_points, uniform_points, or Hilbert contiguous chunk')
parser.add_argument('--chunk_size', type=int, default=50,
                    help='Number of contiguous Hilbert tokens per sampled chunk (default: 50)')
parser.add_argument('--eval_chunk_strategy', type=str, default='center',
                    choices=['center', 'random'],
                    help='Chunk sampling strategy in validation: center (deterministic) or random')
parser.add_argument('--order_mode', type=str, default='keep',
                    choices=['keep', 'random_perm'],
                    help='Feature order control: keep (preserve) or random_perm (fixed shuffle, negative control)')
parser.add_argument('--order_seed', type=int, default=1,
                    help='base seed for deterministic random_perm ordering (default: 1)')
parser.add_argument('--pool_size', type=int, default=50,
                    help='pooling window size for ATP-Pool')
parser.add_argument('--pool_mode', type=str, default='avg',
                    choices=['avg', 'diffusion', 'residual', 'bp'],
                    help='ATPPool mode: avg (plain), diffusion (PM diffusion), residual (boundary residual), bp (boundary-aware pooling)')
parser.add_argument('--diffusion_steps', type=int, default=0,
                    help='anisotropic diffusion steps in ATP-Pool (0=disable diffusion but keep avg_pool)')
parser.add_argument('--K_init', type=float, default=2.5,
                    help='initial boundary conductance threshold K in ATP-Pool')
parser.add_argument('--atp_dt', type=float, default=0.1,
                    help='diffusion step size in ATP-Pool')
parser.add_argument('--norm_type', type=str, default='mean', choices=['mean', 'sum'],
                    help='norm type for gradient in ATP-Pool: mean (scale-stable) or sum (L2)')
parser.add_argument('--tau_init', type=float, default=2.0,
                    help='initial temperature for boundary residual pooling')
parser.add_argument('--gamma_init', type=float, default=0.0,
                    help='initial raw gamma for boundary residual pooling; 0 means avg pooling at init')
parser.add_argument('--bp_alpha_init', type=float, default=1.0,
                    help='initial alpha for BP-Pool (cosine similarity weight)')
parser.add_argument('--bp_beta_init', type=float, default=1.0,
                    help='initial beta for BP-Pool (gradient penalty weight)')
parser.add_argument('--bp_lambda_init', type=float, default=1.0,
                    help='initial lambda for BP-Pool (softmax temperature)')
parser.add_argument('--local_layers', type=int, default=1,
                    help='number of local Mamba layers')
parser.add_argument('--global_layers', type=int, default=1,
                    help='number of global Mamba layers')
parser.add_argument('--local_segment_mode', type=str, default='none',
                    choices=['none', 'chunk'],
                    help='Segment-wise Local Mamba: none (flat) or chunk (independent per chunk)')
parser.add_argument('--local_segment_size', type=int, default=50,
                    help='Segment size for segment-wise Local Mamba (usually = chunk_size = pool_size)')
parser.add_argument('--disable_atp_pool', action='store_true', default=False,
                    help='completely remove ATP-Pool (NOT same as diffusion_steps=0)')
parser.add_argument('--disable_random_sampling', action='store_true', default=False,
                    help='disable random sampling and use deterministic sampling')

## Attention readout params
parser.add_argument('--attn_type', type=str, default='simple', choices=['simple', 'gated'],
                    help='attention readout type: simple (Linear->Tanh->Linear) or gated (V*U -> w)')
parser.add_argument('--attn_dim', type=int, default=128,
                    help='intermediate dimension for attention layers (default: 128)')

## Early stopping params

parser.add_argument('--es_patience', type=int, default=20,
                    help='early stopping patience')
parser.add_argument('--es_stop_epoch', type=int, default=50,
                    help='earliest epoch possible for stopping')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size (only batch_size=1 supported)')

## Eval artifact params
parser.add_argument('--save_eval_artifacts', action='store_true', default=False,
                    help='save per-fold eval artifacts (predictions CSV, ROC, confusion matrix, metrics JSON)')
parser.add_argument('--eval_artifact_dir', type=str, default=None,
                    help='directory for eval artifacts (default: results_dir/eval_artifacts)')
parser.add_argument('--plot_roc', action='store_true', default=False,
                    help='generate ROC curve plots')
parser.add_argument('--plot_confusion', action='store_true', default=False,
                    help='generate confusion matrix plots')


args = parser.parse_args()

# Compute derived parameters
args.use_atp_pool = not args.disable_atp_pool
args.use_random_sampling = not args.disable_random_sampling

# Current MIL classification pipeline only supports batch_size=1
assert args.batch_size == 1, f"Current MIL pipeline only supports batch_size=1, got {args.batch_size}"

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Deviece is:', device)

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

# IHG-Mamba full parameter logging
settings.update({
    'hidden_dim': args.hidden_dim,
    'max_seq_len': args.max_seq_len,
    'pool_size': args.pool_size,
    'local_layers': args.local_layers,
    'global_layers': args.global_layers,
    'diffusion_steps': args.diffusion_steps,
    'K_init': args.K_init,
    'atp_dt': args.atp_dt,
    'norm_type': args.norm_type,
    'pool_mode': args.pool_mode,
    'tau_init': args.tau_init,
    'gamma_init': args.gamma_init,
    'bp_alpha_init': args.bp_alpha_init,
    'bp_beta_init': args.bp_beta_init,
    'bp_lambda_init': args.bp_lambda_init,
    'use_atp_pool': args.use_atp_pool,
    'features_already_hilbert': args.features_already_hilbert,
    'use_hilbert_index': args.use_hilbert_index,
    'disable_random_sampling': args.disable_random_sampling,
    'es_patience': args.es_patience,
    'es_stop_epoch': args.es_stop_epoch,
    'mambamil_type': args.mambamil_type,
    'mambamil_rate': args.mambamil_rate,
    'mambamil_layer': args.mambamil_layer,
    'in_dim': args.in_dim,
    'preloading': args.preloading,
    'backbone': args.backbone,
    'patch_size': args.patch_size,
    'csv_path': args.csv_path,
    'data_root_dir': args.data_root_dir,
    'feature_subdir': args.feature_subdir,
    'sampling_mode': args.sampling_mode,
    'chunk_size': args.chunk_size,
    'eval_chunk_strategy': args.eval_chunk_strategy,
    'order_mode': args.order_mode,
    'local_segment_mode': args.local_segment_mode,
    'local_segment_size': args.local_segment_size,
    'batch_size': args.batch_size,
    'attn_type': args.attn_type,
    'attn_dim': args.attn_dim,
})


print('\nLoad Dataset')

if args.task == 'LUAD_LUSC':
    args.n_classes=2
    # 灵活读取 CSV
    csv_to_load = args.csv_path if args.csv_path else 'dataset_csv/LUAD_LUSC.csv'
    dataset = Generic_MIL_Dataset(csv_path = csv_to_load,
                            data_dir= args.data_root_dir, # <-- 将 None 改为 args.data_root_dir
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'BRACS':
    args.n_classes=7
    csv_to_load = args.csv_path if args.csv_path else 'dataset_csv/BRACS.csv'
    dataset = Generic_MIL_Dataset(csv_path = csv_to_load,
                            data_dir= args.data_root_dir, # <-- 将 None 改为 args.data_root_dir
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'PB':0, 'IC':1, 'DCIS':2, 'N':3, 'ADH': 4,
                                          'FEA':5, 'UDH': 6 },
                            patient_strat=False,
                            ignore=[])
elif args.task == 'toy_cls':
    args.n_classes = 2  # 假装这是一个二分类任务
    csv_to_load = args.csv_path if args.csv_path else 'dataset_csv/toy_survival.csv'

    # 🌟 绝招：引入 collections.defaultdict。遇到任何没见过的标签（如 25.6），都默认返回类别 0
    import collections

    dummy_dict = collections.defaultdict(lambda: 0)

    dataset = Generic_MIL_Dataset(csv_path=csv_to_load,
                                  data_dir=args.data_root_dir,
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict=dummy_dict,  # <--- 使用我们的万能伪装字典
                                  patient_strat=False,
                                  ignore=[],
                                  label_col='survival_months')

else:
    raise NotImplementedError
    
# if not os.path.isdir(args.results_dir):
    # os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment.txt', 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        


# set auto resume 
if args.k_start == -1:
    folds = args.k if args.k_end == -1 else args.k_end
    for i in range(folds):
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        if not os.path.exists(filename):
            args.k_start = i
            break
    print('Training from fold: {}'.format(args.k_start))

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")