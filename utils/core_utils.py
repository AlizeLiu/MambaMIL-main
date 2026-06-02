import warnings
import numpy as np
import torch
from utils.utils import *
import os
from dataset.dataset_generic import save_splits
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from utils.eval_utils import save_fold_eval_artifacts, save_summary_metrics, compute_binary_roc, plot_mean_roc
import wandb

def find_func(model_name: str):
    model_name = model_name.lower()
    if model_name in ['mean_mil', 'max_mil', 'att_mil','trans_mil', 's4model','mamba_mil']:

        return train_loop, validate
    else:
        raise NotImplementedError
    

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):

    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    
    loss_fn = nn.CrossEntropyLoss()
    
    print('Done!')
    
    print('\nInit Model...', end=' ')
    
    if args.model_type == 'mean_mil':
        from models.Mean_Max_MIL import MeanMIL
        model = MeanMIL(args.in_dim, args.n_classes)
    elif args.model_type == 'max_mil':
        from models.Mean_Max_MIL import MaxMIL
        model = MaxMIL(args.in_dim, args.n_classes)
    elif args.model_type == 'att_mil':
        from models.ABMIL import DAttention
        model = DAttention(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 'trans_mil':
        from models.TransMIL import TransMIL
        model = TransMIL(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 's4model':
        from models.S4MIL import S4Model
        model = S4Model(in_dim = args.in_dim, n_classes = args.n_classes, act = 'gelu', dropout = args.drop_out)
    elif args.model_type == 'mamba_mil':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(
            in_dim=args.in_dim,
            n_classes=args.n_classes,
            dropout=args.drop_out,
            act='gelu',
            survival=False,
            layer=args.mambamil_layer,
            rate=args.mambamil_rate,
            type=args.mambamil_type,
            hidden_dim=args.hidden_dim,
            local_layers=args.local_layers,
            global_layers=args.global_layers,
            pool_size=args.pool_size,
            use_atp_pool=args.use_atp_pool,
            diffusion_steps=args.diffusion_steps,
            K_init=args.K_init,
            atp_dt=args.atp_dt,
            norm_type=args.norm_type,
            pool_mode=args.pool_mode,
            tau_init=args.tau_init,
            gamma_init=args.gamma_init,
            local_segment_mode=args.local_segment_mode,
            local_segment_size=args.local_segment_size,
            attn_type=getattr(args, 'attn_type', 'simple'),
            attn_dim=getattr(args, 'attn_dim', 128),
        )
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')

    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    # Set IHG-Mamba dataset parameters on all splits
    for split in [train_split, val_split, test_split]:
        split.max_seq_len = args.max_seq_len
        split.feature_subdir = args.feature_subdir
        split.features_already_hilbert = args.features_already_hilbert
        split.use_hilbert_index = args.use_hilbert_index
        split.hilbert_index_dir = args.hilbert_index_dir
        split.sampling_mode = args.sampling_mode
        split.chunk_size = args.chunk_size
        split.eval_chunk_strategy = args.eval_chunk_strategy
        split.order_mode = args.order_mode
        split.order_seed = args.order_seed
        split.use_random_sampling = args.use_random_sampling
    train_split.training_mode = True
    val_split.training_mode = False
    test_split.training_mode = False

    # Print sampling configuration
    print(f"[Sampling] mode={args.sampling_mode}, max_seq_len={args.max_seq_len}, "
          f"chunk_size={args.chunk_size}, eval_strategy={args.eval_chunk_strategy}, "
          f"order_mode={args.order_mode}")

    if args.sampling_mode == 'chunk' and args.chunk_size != args.pool_size:
        print(f"[WARNING] chunk_size({args.chunk_size}) != pool_size({args.pool_size}). "
              f"Pool windows may cross chunk boundaries.")

    if args.sampling_mode == 'chunk' and not args.features_already_hilbert and not args.use_hilbert_index:
        print("[WARNING] chunk sampling assumes Hilbert-ordered features. "
              "Current features may not be Hilbert sorted.")

    # Segment-wise Local Mamba configuration
    if args.local_segment_mode == 'chunk':
        print(f"[Local Segment] Enabled segment-wise Local Mamba: "
              f"segment_size={args.local_segment_size}, chunk_size={args.chunk_size}, pool_size={args.pool_size}")
        if args.sampling_mode != 'chunk':
            print(f"[WARNING] local_segment_mode=chunk is intended for sampling_mode=chunk. "
                  f"Current sampling_mode={args.sampling_mode}")
        if args.local_segment_size != args.chunk_size:
            print(f"[WARNING] local_segment_size({args.local_segment_size}) != chunk_size({args.chunk_size}). "
                  f"Segment boundaries may not match sampled chunks.")
        if args.pool_size > args.local_segment_size:
            raise ValueError(f"pool_size({args.pool_size}) > local_segment_size({args.local_segment_size}) is not allowed.")

    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.es_patience, stop_epoch=args.es_stop_epoch, verbose=True)

    else:
        early_stopping = None
    print('Done!')

    train_loop_func, validate_func = find_func(args.model_type)
    for epoch in range(args.max_epochs):

        train_loop_func(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate_func(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        # earlystop
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger, test_raw = summary(model, test_loader, args.n_classes, return_raw=True)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    # Save eval artifacts if requested
    if getattr(args, 'save_eval_artifacts', False):
        val_raw = summary(model, val_loader, args.n_classes, return_raw=True)[4]
        artifact_dir = getattr(args, 'eval_artifact_dir', None) or args.results_dir
        
        # Get slide_ids and case_ids from dataset
        test_slide_ids = test_raw.get('slide_ids', None)
        test_case_ids = test_raw.get('case_ids', None)
        val_slide_ids = val_raw.get('slide_ids', None)
        val_case_ids = val_raw.get('case_ids', None)
        
        test_metrics = save_fold_eval_artifacts(
            artifact_dir, cur, 'test',
            test_raw['y_true'], test_raw['y_pred'], test_raw['y_prob'],
            plot_roc_flag=getattr(args, 'plot_roc', False),
            plot_confusion_flag=getattr(args, 'plot_confusion', False),
            slide_ids=test_slide_ids, case_ids=test_case_ids, fold_num=cur,
        )
        val_metrics = save_fold_eval_artifacts(
            artifact_dir, cur, 'val',
            val_raw['y_true'], val_raw['y_pred'], val_raw['y_prob'],
            plot_roc_flag=getattr(args, 'plot_roc', False),
            plot_confusion_flag=getattr(args, 'plot_confusion', False),
            slide_ids=val_slide_ids, case_ids=val_case_ids, fold_num=cur,
        )
        print(f'  Eval artifacts saved to {artifact_dir}/eval_artifacts/fold_{cur}/')

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 



def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        try:
            auc = roc_auc_score(labels, prob[:, 1])
        except ValueError as e:
            warnings.warn(f"AUC undefined for fold {cur} val: {e}")
            auc = float('nan')
    else:
        try:
            auc = roc_auc_score(labels, prob, multi_class='ovr')
        except ValueError as e:
            warnings.warn(f"AUC undefined for fold {cur} val: {e}")
            auc = float('nan')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False



def summary(model, loader, n_classes, return_raw=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    all_Y_hat = []
    all_label = []
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds.extend(Y_hat.cpu().numpy())
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

        all_Y_hat.append(Y_hat.cpu().numpy())
        all_label.append(label.cpu().numpy())

    test_error /= len(loader)
    all_Y_hat = np.concatenate(all_Y_hat)
    all_label = np.concatenate(all_label)

    if n_classes == 2:
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError as e:
            warnings.warn(f"AUC undefined in summary: {e}")
            auc = float('nan')
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                try:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                except ValueError:
                    aucs.append(float('nan'))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    if return_raw:
        # Collect slide_ids and case_ids for prediction CSV
        slide_id_list = []
        case_id_list = []
        for batch_idx in range(len(loader)):
            sid = slide_ids.iloc[batch_idx] if slide_ids is not None else f'slide_{batch_idx}'
            slide_id_list.append(str(sid))
            # Try to get case_id from slide_data
            try:
                cid = loader.dataset.slide_data['case_id'].iloc[batch_idx]
                case_id_list.append(str(cid))
            except (KeyError, IndexError):
                case_id_list.append('')
        
        raw = {
            'y_true': all_labels,
            'y_pred': all_Y_hat,
            'y_prob': all_probs,
            'slide_ids': slide_id_list,
            'case_ids': case_id_list,
        }
        return patient_results, test_error, auc, acc_logger, raw

    return patient_results, test_error, auc, acc_logger