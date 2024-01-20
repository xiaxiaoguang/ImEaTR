aux_loss=True,
bsz=32,
ckpt_filepath='results/hl-12_30_11_37-exp/model.ckpt', 
clip_length=2, 
conf_thd=0.0, 
contrastive_align_loss=False, 
contrastive_align_loss_coef=0.0, 
contrastive_hdim=64, 
ctx_mode='video_tef', 
data_ratio=1.0, 
debug=False, 
dec_layers=3, 
device=device(type='cuda'), 
dim_feedforward=1024, 
dropout=0.1, 
dset_name='hl', 
enc_layers=3, 
eos_coef=0.1, 
eval_bsz=50, 
eval_interval=1, 
eval_log_filepath='results/hl-12_30_11_37-exp/eval.log.txt', 
eval_path='data/qvhighlights/highlight_val_release.jsonl', 
eval_split_name='val', eval_untrained=False, event_coef=3, 
exp_id='exp', 
giou_loss_coef=1, 
grad_clip=0.1, 
hidden_dim=256, 
input_dropout=0.5, 
label_loss_coef=4, 
lr=0.0001, 
lr_drop=150, 
lw_saliency=1.0, 
max_after_nms=10, 
max_before_nms=10, 
max_es_cnt=200, 
max_q_l=32, 
max_v_l=75, 
max_windows=5, 
n_epoch=200, 
n_input_proj=2, 
nheads=8, 
nms_thd=-1, 
no_norm_tfeat=False, 
no_norm_vfeat=False, 
no_pin_memory=False, 
no_sort_results=False, 
num_queries=10, 
num_slot_iter=3, 
num_workers=4, 
pin_memory=True, 
position_embedding='sine', 
pre_norm=False, 
query_dim=2, 
results_dir='results/hl-12_30_11_37-exp', 
results_root='results', 
resume=None, 
resume_all=False, 
saliency_margin=0.2, 
save_interval=500, 
seed=429, 
set_cost_class=4, 
set_cost_giou=1, 
set_cost_span=10, 
span_loss_coef=10, 
span_loss_type='l1', 
start_epoch=None, 
t_feat_dim=512, 
t_feat_dir='data/qvhighlights/features/clip_text_features/', 
temperature=0.07, 
tensorboard_log_dir='results/hl-12_30_11_37-exp/tensorboard_log', 
train_log_filepath='results/hl-12_30_11_37-exp/train.log.txt', 
train_path='data/qvhighlights/highlight_train_release.jsonl', 
txt_drop_ratio=0, 
use_tef=True, 
use_txt_pos=False, 
use_video=True, 
v_feat_dim=2818, 
v_feat_dirs=['data/qvhighlights/features/slowfast_features', 'data/qvhighlights/features/clip_features'],
wd=0.0001