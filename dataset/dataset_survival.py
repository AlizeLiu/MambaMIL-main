from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


def hilbert_chunk_sample_indices(
    n: int,
    max_seq_len: int,
    chunk_size: int,
    training: bool = True,
    eval_strategy: str = 'center',
) -> torch.LongTensor:
    """Hilbert contiguous chunk sampling.

    从 Hilbert 排序后的完整序列 [0, n) 中抽取多个连续小块，
    每个块内部保持 Hilbert 局部连续性，总长度为 max_seq_len。

    Args:
        n: 完整 Hilbert 序列长度
        max_seq_len: 目标采样长度
        chunk_size: 每个连续块的 token 数
        training: True=随机选块起点, False=按 eval_strategy 选择
        eval_strategy: 'center' (确定性) 或 'random'

    Returns:
        indices: torch.LongTensor, 长度 ≤ max_seq_len, 按 Hilbert 顺序排列
    """
    if max_seq_len is None or max_seq_len <= 0 or n <= max_seq_len:
        return torch.arange(n, dtype=torch.long)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    num_full_chunks = max_seq_len // chunk_size
    remainder = max_seq_len % chunk_size

    chunk_sizes = [chunk_size] * num_full_chunks
    if remainder > 0:
        chunk_sizes.append(remainder)

    num_chunks = len(chunk_sizes)
    if num_chunks == 0:
        chunk_sizes = [max_seq_len]
        num_chunks = 1

    # 将 [0, n) 均分成 num_chunks 个大区间
    edges = torch.linspace(0, n, steps=num_chunks + 1).long()
    chunks = []

    for i, cur_chunk_size in enumerate(chunk_sizes):
        start = int(edges[i].item())
        end = int(edges[i + 1].item())
        seg_len = end - start

        if seg_len <= 0:
            continue

        cur_chunk_size = min(cur_chunk_size, seg_len)

        if seg_len <= cur_chunk_size:
            # 区间比 chunk 还小，直接取整个区间
            offset = start
        else:
            if training:
                offset = torch.randint(
                    low=start,
                    high=end - cur_chunk_size + 1,
                    size=(1,)
                ).item()
            else:
                if eval_strategy == 'center':
                    offset = start + (seg_len - cur_chunk_size) // 2
                elif eval_strategy == 'random':
                    offset = torch.randint(
                        low=start,
                        high=end - cur_chunk_size + 1,
                        size=(1,)
                    ).item()
                else:
                    raise ValueError(f"Unknown eval_strategy: {eval_strategy}")

        idx = torch.arange(offset, offset + cur_chunk_size, dtype=torch.long)
        chunks.append(idx)

    if len(chunks) == 0:
        return torch.arange(min(n, max_seq_len), dtype=torch.long)

    indices = torch.cat(chunks, dim=0)

    # 截断到 max_seq_len
    if indices.numel() > max_seq_len:
        indices = indices[:max_seq_len]

    return indices.long()

from utils.utils import generate_split, nth


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', mode = 'omic', apply_sig = False,
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.slide_data)


        slide_data = pd.read_csv(csv_path, low_memory=False)
        #slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        import pdb
        #pdb.set_trace()

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        # if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
        #     slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict

        # 统计多 slide patient
        multi_slide_count = sum(1 for v in patient_dict.values() if len(v) > 1)
        if multi_slide_count > 0:
            print(f"[WARNING] {multi_slide_count} patients have multiple slides. "
                  f"Slide boundaries may break Hilbert adjacency in Mamba/ATP.")

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.mode = mode
        self.cls_ids_prep()

        if print_info:
            self.summarize()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./dataset_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)

        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, backbone, patch_size, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, signatures=self.signatures, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
            split.set_backbone(backbone)
            split.set_patch_size(patch_size)
        else:
            split = None

        return split


    def return_splits(self, backbone, patch_size = '', from_id: bool=True, csv_path: str=None):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, mode = self.mode, metadata= self.apply_sig, data_dir=self.data_dir, num_classes=self.num_classes)
                train_split.set_backbone(backbone)
                train_split.set_patch_size(patch_size)
                print('hhhhhhhhhhhhhhhhhhhhhhhhh')
            else:
                train_split = None

            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, metadata = self.apply_sig, mode = self.mode, data_dir=self.data_dir, num_classes=self.num_classes)
                val_split.set_backbone(backbone)
                val_split.set_patch_size(patch_size)

            else:
                val_split = None

            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, metadata = self.apply_sig, mode = self.mode, data_dir=self.data_dir, num_classes=self.num_classes)
                test_split.set_backbone(backbone)
                test_split.set_patch_size(patch_size)

            else:
                test_split = None
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
            train_split = self.get_split_from_df(backbone, patch_size, all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(backbone, patch_size, all_splits=all_splits, split_key='val')
            test_split = self.get_split_from_df(backbone, patch_size, all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
            # print("****** Normalizing Data ******")
            # scalers = train_split.get_scaler()
            # train_split.apply_scaler(scalers=scalers)
            # val_split.apply_scaler(scalers=scalers)
            # test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split, test_split

    '''
    Added function create_splits from Generic_WSI_Classification_Dataset
    '''
    def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
        settings = {
                    'n_splits' : k,
                    'val_num' : val_num,
                    'test_num': test_num,
                    'label_frac': label_frac,
                    'seed': self.seed,
                    'custom_test_ids': custom_test_ids
                    }

        if self.patient_strat:
            settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)


    '''
    Added function set_splits from Generic_WSI_Classification_Dataset
    '''
    def set_splits(self,start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)

        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))]

            for split in range(len(ids)):
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

        else:
            self.train_ids, self.val_ids, self.test_ids = ids


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None

    '''
    Added functions test_split_gen and save_split from Generic_WSI_Classification_Dataset
    '''

    def test_split_gen(self, return_descriptor=False):

        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
                            columns= columns)
        df = df.reset_index(drop=True)
        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]

        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index = False)


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str = 'omic', 
                 max_seq_len=2500, use_hilbert_index=False, features_already_hilbert=True,
                 use_random_sampling=True, num_eval_views=1, **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False
        self.training_mode = True  # 默认训练模式
        # IHG-Mamba 参数化
        self.max_seq_len = max_seq_len
        self.use_hilbert_index = use_hilbert_index
        self.features_already_hilbert = features_already_hilbert
        self.use_random_sampling = use_random_sampling
        self.num_eval_views = num_eval_views

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        # 1. 精准指向特征文件
                        feature_subdir = getattr(self, 'feature_subdir', 'pt_files')
                        wsi_path = os.path.join(data_dir, feature_subdir, f'{slide_id}.pt')

                        if not os.path.exists(wsi_path):
                            print(f"[Error] 找不到特征文件: {wsi_path}")
                            continue

                        wsi_bag = torch.load(wsi_path)

                        # =========================================================
                        # IHG-Mamba: Hilbert 空间拓扑重排 (条件加载)
                        # =========================================================
                        # 只有当 features_already_hilbert=False 且 use_hilbert_index=True 时才重排
                        if self.use_hilbert_index and not self.features_already_hilbert:
                            hilbert_path = os.path.join(data_dir, 'hilbert', f'{slide_id}_hilbert.pt')

                            if not os.path.exists(hilbert_path):
                                raise FileNotFoundError(f"Missing Hilbert index: {hilbert_path}")

                            hilbert_idx = torch.load(hilbert_path).long()

                            if hilbert_idx.numel() != wsi_bag.shape[0]:
                                raise RuntimeError(
                                    f"Hilbert index length mismatch for {slide_id}: "
                                    f"idx={hilbert_idx.numel()}, feature={wsi_bag.shape[0]}"
                                )

                            wsi_bag = wsi_bag[hilbert_idx]
                            print(f"\n✅ [IHG-Mamba] Hilbert 重排 {slide_id}: {wsi_bag.shape}")

                        # ===== 防 OOM: 限制最大序列长度 =====
                        max_seq_len = getattr(self, 'max_seq_len', 2500)
                        use_random_sampling = getattr(self, 'use_random_sampling', True)
                        sampling_mode = getattr(self, 'sampling_mode', 'random_points')
                        chunk_size = getattr(self, 'chunk_size', 50)
                        eval_chunk_strategy = getattr(self, 'eval_chunk_strategy', 'center')

                        if max_seq_len is not None and max_seq_len > 0 and wsi_bag.shape[0] > max_seq_len:
                            orig_len = wsi_bag.shape[0]

                            if sampling_mode == 'chunk':
                                indices = hilbert_chunk_sample_indices(
                                    n=orig_len,
                                    max_seq_len=max_seq_len,
                                    chunk_size=chunk_size,
                                    training=self.training_mode,
                                    eval_strategy=eval_chunk_strategy,
                                )

                            elif sampling_mode == 'random_points':
                                if self.training_mode and use_random_sampling:
                                    indices = torch.randperm(orig_len)[:max_seq_len]
                                    indices, _ = indices.sort()
                                else:
                                    indices = torch.linspace(0, orig_len - 1, max_seq_len).long()

                            elif sampling_mode == 'uniform_points':
                                indices = torch.linspace(0, orig_len - 1, max_seq_len).long()

                            else:
                                raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

                            wsi_bag = wsi_bag[indices]
                            print(f"[Subsample] {slide_id}: {orig_len} -> {wsi_bag.shape[0]} (mode={sampling_mode})")

                        # ===== order_mode: sampling 之后再打乱顺序 (纯顺序负对照) =====
                        order_mode = getattr(self, 'order_mode', 'keep')
                        if order_mode == 'random_perm':
                            base_seed = getattr(self, 'seed', 1)
                            import hashlib
                            h = int(hashlib.md5(slide_id.encode()).hexdigest(), 16) % (2**31)
                            g = torch.Generator()
                            g.manual_seed(base_seed + h)
                            perm = torch.randperm(wsi_bag.shape[0], generator=g)
                            wsi_bag = wsi_bag[perm]

                        path_features.append(wsi_bag)

                    if len(path_features) > 0:
                        path_features = torch.cat(path_features, dim=0)
                    else:
                        raise RuntimeError(f"No valid features found for case {case_id}. All slide IDs: {slide_ids}")

                    return (path_features, torch.zeros((1, 1)), label, event_time, c)

                # --- 暂时折叠了其他 mode (cluster, omic等)，因为你目前只测纯病理(path) ---
                else:
                    raise NotImplementedError(f'Mode [{self.mode}] not implemented yet.')
            else:
                return slide_ids, label, event_time, c

class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.training_mode = True  # 默认训练模式，外部可切换
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        # IHG-Mamba 参数默认值 (避免 preloading=yes 时属性缺失)
        self.max_seq_len = 2500
        self.use_hilbert_index = False
        self.features_already_hilbert = True
        self.use_random_sampling = True
        self.num_eval_views = 1
        self.feature_subdir = 'pt_files'
        self.sampling_mode = 'random_points'
        self.chunk_size = 50
        self.eval_chunk_strategy = 'center'
        self.order_mode = 'keep'
        self.seed = 1
        #! add from HIPT
        # cluster_dir = "/".join(data_dir.split("/")[0:-1])
        # if os.path.isfile(os.path.join(cluster_dir, 'fast_cluster_ids.pkl')):
        #     with open(os.path.join(cluster_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
        #         self.fname2ids = pickle.load(handle)
        # else:
        #     print("Cluster file missing")
        ### --> Initializing genomic features in Generic Split
        # self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        # self.signatures = signatures

        # with open(os.path.join(data_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
        #     self.fname2ids = pickle.load(handle)

        # def series_intersection(s1, s2):
        #     return pd.Series(list(set(s1) & set(s2)))

        # if self.signatures is not None:
        #     self.omic_names = []
        #     for col in self.signatures.columns:
        #         omic = self.signatures[col].dropna().unique()
        #         omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
        #         omic = sorted(series_intersection(omic, self.genomic_features.columns))
        #         self.omic_names.append(omic)
        #     self.omic_sizes = [len(omic) for omic in self.omic_names]
        # print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--
    
    def set_backbone(self, backbone):
        print('Setting Backbone:', backbone)
        self.backbone = backbone

    def set_patch_size(self, size):
        print('Setting Patchsize:', size)
        self.patch_size = size

    def pre_loading(self, thread=8):
        # set flag
        self.cache_flag = True

        ids = list(range(len(self)))
        from multiprocessing.pool import ThreadPool
        exe = ThreadPool(thread)
        exe.map(self.__getitem__, ids)