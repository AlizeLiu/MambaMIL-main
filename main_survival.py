from __future__ import print_function

import argparse
import os
from timeit import default_timer as timer

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.survival_core_utils import train
from dataset.dataset_survival import Generic_MIL_Survival_Dataset

# pytorch imports
import torch
import pandas as pd
import numpy as np
import wandb

def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

#TODO：训练时修改为在线模式，测试时修改为离线模式，避免每次都要输入 API Key
    # 强制 Wandb 在离线模式下运行，不再索要 API Key！
    os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project=args.task)
    wandb.config.update(args)
    if args.k_start == -1:
        fold_start = 0
    else:
        fold_start = args.k_start
    if args.k_end == -1:
        fold_end = args.k
    else:
        fold_end = args.k_end

    latest_test_cindex = []
    latest_val_cindex = []
    
    folds = np.arange(fold_start, fold_end)
    
    for i in folds:
        start_time = timer()
        seed_torch(args.seed)
        results_pkl_path = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        if os.path.isfile(results_pkl_path):
            print("Skipping Split %d" % i)
            continue
        train_dataset, val_dataset, test_dataset = dataset.return_splits(args.backbone, args.patch_size, from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        if args.k_fold:
            print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        else: 
            print('training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
        if args.k_fold:
            datasets = (train_dataset, val_dataset)
        else:
            datasets = (train_dataset, val_dataset, test_dataset)
        if args.preloading == 'yes':
            for d in datasets:
                d.pre_loading()
                
        if args.task_type == 'survival':
            if args.k_fold:
                cindex_val = train(datasets, i, args)
                latest_val_cindex.append(cindex_val)
            else:
                results, cindex_test, cindex_val = train(datasets, i, args)
                latest_val_cindex.append(cindex_val)
                latest_test_cindex.append(cindex_test)
            
        # results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)

        # all_test_auc.append(test_auc)
        # all_val_auc.append(val_auc)
        # all_test_acc.append(test_acc)
        # all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        if not args.k_fold:
            save_pkl(filename, results)
    if args.k_fold:
        final_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})
    else: 
        final_df = pd.DataFrame({'folds': folds, 'test_cindex': latest_test_cindex, 
            'val_cindex': latest_val_cindex, })
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(fold_start, fold_end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    mean_val = final_df['val_cindex'].mean()
    std_val = final_df['val_cindex'].std()

    if not args.k_fold:
        mean_test = final_df['test_cindex'].mean()
        std_test = final_df['test_cindex'].std()

 
    if args.k_fold:
        df_append = pd.DataFrame({
            'folds': ['mean', 'std'],
            'val_cindex': [mean_val, std_val]
        })
    else:
        df_append = pd.DataFrame({
            'folds': ['mean', 'std'],
            'test_cindex': [mean_test, std_test],
            'val_cindex': [mean_val, std_val]
        })
    final_df = pd.concat([final_df, df_append])
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(fold_start, fold_end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    final_df['folds'] = final_df['folds'].astype(str)
    table = wandb.Table(dataframe=final_df)
    wandb.log({"summary": table})
    if args.k_fold:
        wandb.log({"mean_val_cindex": mean_val})
    else:
        wandb.log({"mean_test_cindex": mean_test, "mean_val_cindex": mean_val})

    
# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='Data directory to WSI features (extracted via CLAM)')
parser.add_argument('--feature_subdir', type=str, default='pt_files',
                    help='subdirectory under data_root_dir for pt features (default: pt_files)')
parser.add_argument('--sampling_mode', type=str, default='random_points',
                    choices=['random_points', 'uniform_points', 'chunk'],
                    help='Patch sampling mode: random_points, uniform_points, or Hilbert contiguous chunk')
parser.add_argument('--chunk_size', type=int, default=50,
                    help='Number of contiguous Hilbert tokens per sampled chunk (default: 50)')
parser.add_argument('--eval_chunk_strategy', type=str, default='center',
                    choices=['center', 'random'],
                    help='Chunk sampling strategy in validation: center (deterministic) or random')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=1,)
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
parser.add_argument('--gc', type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['mean_mil', 'max_mil', 'att_mil','trans_mil', 's4model','mamba_mil'], default='mamba_mil', 
                    help='type of model')
parser.add_argument('--mode', type = str, choices=['path', 'omic', 'pathomic', 'cluster'], default='path', help='which modalities to use')
parser.add_argument('--apply_sig', action='store_true', default=False, help='Use genomic features as signature embeddings')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--fusion', type=str, choices=['None', 'concat', 'bilinear'], default='None', help='Type of fusion. (Default: None).')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str)
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--patch_size', type=str, default='')
parser.add_argument('--preloading', type=str, default='no')
parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--k_fold', action='store_true', default=False, help='use k-fold cross validation')


## mambamil

parser.add_argument('--mambamil_rate',type=int, default=10, help='mambamil_rate')
parser.add_argument('--mambamil_layer',type=int, default=2, help='legacy param: actual layers controlled by local_layers/global_layers')
parser.add_argument('--mambamil_type',type=str, default='SRMamba', choices= ['Mamba', 'BiMamba', 'SRMamba'], help='mambamil_type')

parser.add_argument('--csv_path', type=str, default=None, help='path to the dataset csv file')

## IHG-Mamba architecture params

parser.add_argument('--hidden_dim', type=int, default=256,
                    help='hidden dimension for IHG-Mamba')

parser.add_argument('--max_seq_len', type=int, default=2500,
                    help='maximum number of patches per WSI; <=0 means no truncation')

parser.add_argument('--pool_size', type=int, default=100,
                    help='pooling window size for ATP-Pool')

parser.add_argument('--local_layers', type=int, default=1,
                    help='number of local Mamba layers')

parser.add_argument('--global_layers', type=int, default=1,
                    help='number of global Mamba layers')

parser.add_argument('--diffusion_steps', type=int, default=2,
                    help='anisotropic diffusion steps in ATP-Pool (0=disable diffusion but keep avg_pool)')

parser.add_argument('--K_init', type=float, default=0.5,
                    help='initial boundary conductance threshold K in ATP-Pool')

parser.add_argument('--atp_dt', type=float, default=0.1,
                    help='diffusion step size in ATP-Pool')

parser.add_argument('--norm_type', type=str, default='mean', choices=['mean', 'sum'],
                    help='norm type for gradient in ATP-Pool: mean (scale-stable) or sum (L2)')

parser.add_argument('--disable_atp_pool', action='store_true', default=False,
                    help='completely remove ATP-Pool (NOT same as diffusion_steps=0)')

parser.add_argument('--pool_mode', type=str, default='diffusion',
                    choices=['avg', 'diffusion', 'residual'],
                    help='ATPPool mode: avg (plain), diffusion (PM diffusion), residual (boundary residual)')

parser.add_argument('--tau_init', type=float, default=2.0,
                    help='initial temperature for boundary residual pooling')

parser.add_argument('--gamma_init', type=float, default=0.0,
                    help='initial raw gamma for boundary residual pooling; 0 means avg pooling at init')

parser.add_argument('--features_already_hilbert', dest='features_already_hilbert',
                    action='store_true', default=True,
                    help='input .pt features are already Hilbert ordered (default)')

parser.add_argument('--features_not_hilbert', dest='features_already_hilbert',
                    action='store_false',
                    help='input .pt features are NOT Hilbert ordered (use with --use_hilbert_index)')

parser.add_argument('--use_hilbert_index', action='store_true', default=False,
                    help='load hilbert/*.pt index and reorder features inside DataLoader')

parser.add_argument('--disable_random_sampling', action='store_true', default=False,
                    help='disable random sampling and use deterministic sampling')

parser.add_argument('--num_eval_views', type=int, default=1,
                    help='number of evaluation views for multiview risk averaging')

## Early stopping params

parser.add_argument('--es_warmup', type=int, default=0,
                    help='early stopping warmup epochs')

parser.add_argument('--es_patience', type=int, default=10,
                    help='early stopping patience')

parser.add_argument('--es_stop_epoch', type=int, default=10,
                    help='earliest epoch possible for stopping')


args = parser.parse_args()

# 计算派生参数
args.use_atp_pool = not args.disable_atp_pool
args.use_random_sampling = not args.disable_random_sampling

# 当前 MIL 生存分析 pipeline 仅支持 batch_size=1
assert args.batch_size == 1, f"Current survival MIL pipeline only supports batch_size=1, got {args.batch_size}"

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

# args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
print("Experiment Name:", args.exp_code)

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
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

# IHG-Mamba 完整参数记录
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
    'use_atp_pool': args.use_atp_pool,
    'features_already_hilbert': args.features_already_hilbert,
    'use_hilbert_index': args.use_hilbert_index,
    'disable_random_sampling': args.disable_random_sampling,
    'num_eval_views': args.num_eval_views,
    'es_warmup': args.es_warmup,
    'es_patience': args.es_patience,
    'es_stop_epoch': args.es_stop_epoch,
    'mambamil_type': args.mambamil_type,
    'mambamil_rate': args.mambamil_rate,
    'mambamil_layer': args.mambamil_layer,
    'gc': args.gc,
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
    'batch_size': args.batch_size,
})


print('\nLoad Dataset')

print('\nLoad Dataset')

# 只要没传任务名，我们就给个默认的 toy_survival，防止下面报错
if args.task is None:
    args.task = 'toy_survival'

if 'survival' in args.task:
    args.n_classes = 4

    # 🌟 IHG-Mamba: 灵活的数据源读取逻辑
    # 如果命令行传了 csv_path，就用命令行的；否则用原版的字符串硬切逻辑
    csv_to_load = args.csv_path if args.csv_path else 'dataset_csv/%s_processed.csv' % args.task.split('_')[1]

    dataset = Generic_MIL_Survival_Dataset(csv_path=csv_to_load,
                                           mode=args.mode,
                                           apply_sig=args.apply_sig,
                                           data_dir=args.data_root_dir,  # <-- 直接使用我们传入的根目录！
                                           shuffle=False,
                                           seed=args.seed,
                                           print_info=True,
                                           patient_strat=False,
                                           n_bins=4,
                                           label_col='survival_months',
                                           ignore=[])
else:
    raise NotImplementedError


if isinstance(dataset, Generic_MIL_Survival_Dataset):
	args.task_type = 'survival'
else:
	raise NotImplementedError
    
# if not os.path.exists(args.results_dir):
#     os.mkdir(args.results_dir)

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

if __name__ == "__main__":
    start_time = timer()
    results = main(args)
    end_time = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end_time - start_time))

