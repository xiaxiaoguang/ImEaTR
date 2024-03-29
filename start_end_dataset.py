import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from utils.span_utils import span_xx_to_cxw
from torchtext import vocab
import torch.nn as nn
import h5py


logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]

    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],`
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,load_all=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.load_all = load_all
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()

        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)
        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        # ctx_l 是video modality 的长度，
        # 可以近似认为，video每一段信息被对应的特征均分了
        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            # tef结构就是[[1/x,2/x],...]
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            # 加上position信息
            else:
                model_inputs["video_feat"] = tef
        # query_feat torch.Size([77, 512])
        # video_feat torch.Size([29, 514])
        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l , meta["duration"])  # (#windows, 2)
            if "subs_train" not in self.data_path and "hl" in self.dset_name: 
                # Amazing that this is only for hl and other size is wrong!!
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels_w_annot(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
            else:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0] , meta["duration"], ctx_l)  # only one gt
        # load all the positive and negtive information
        if self.load_all:
            model_inputs["pos_all"],model_inputs["all"]  = \
            self.get_all_different_labels(meta["relevant_windows"][0], meta["duration"], ctx_l) # only for 
        return dict(meta=meta, model_inputs=model_inputs)
    # def equal_2dvector_in_1d(self,a,b):
    #     bsz = a.shape[0]
    #     ret = []
    #     eps = 1e-4
    #     for i in range(bsz):
    #         if torch.max(torch.abs(a[i]-b[i])) < eps:
    #             ret.append(i)
    #     return ret
    def get_all_different_labels(self,windows,duration,ctx_l):
        clip_len = duration/ctx_l
        
        gt_st = int(windows[0] / clip_len)
        gt_ed = max(0, min(int(windows[1] / clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed
        pos_pool = list(range(gt_st,gt_ed+1))

        all_pool = list(range(0,ctx_l))

        return pos_pool, all_pool
    
    def get_saliency_labels_sub_as_query(self, windows, duration, ctx_l, max_n=2):
        clip_len = duration/ctx_l

        gt_st = int(windows[0] / clip_len)
        gt_ed = max(0, min(int(windows[1] / clip_len), ctx_l) - 1)

        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        if len(neg_pool) < max_n:
            neg_clip_indices = [0,0]
        else:
            neg_clip_indices = random.sample(neg_pool, k=max_n)

        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_w_annot(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample easy negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []

        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        
        return pos_clip_indices, neg_clip_indices

    def get_span_labels(self, windows, ctx_l , duration):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        # 这里他没有兼容charades
        # 或者说他直接就假设所有视频长度都一样,这只是qvhighlights的特性
        clip_len = duration / ctx_l
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        # TODO return windows_xx, windows_cxw together for span_loss_type=="ce"
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / clip_len), min(int(w[1] / clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
        q_feat = np.load(q_feat_path)['last_hidden_state'].astype(np.float32)
        q_feat = q_feat[:self.max_q_l]

        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)

        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        # 主要作用就是加载数据，如果v_feat_dirs传输了多个也会一次性都加载出来
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            _feat_path = join(_feat_dir, f"{vid}.npz")
            _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
            if self.normalize_v:
                _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)


def start_end_collate(batch):

    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()

    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["pos_all" , "all"]:
            batched_data[k] = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=torch.float32,val =-1, fixed_length=None)[0]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        else:
            batched_data[k] = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=torch.float32,val = 0, fixed_length=None)            
        # mask 在这里实现！
        # 作用是数据长度不是不统一吗？那我mask掉填充的部分ok了
        # 呃呃半天才发现这个事情尴尬。。
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    )
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]

    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "pos_all" in batched_model_inputs:
        for name in ["pos_all","all"]:
            targets[name]  = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
