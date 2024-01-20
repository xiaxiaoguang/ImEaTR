dset_name=charades
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip
results_root=results
exp_id=CHD

######## data paths

train_path=../data/charades/charades_sta_train_tvr_format.jsonl
eval_path=../data/charades/charades_sta_test_tvr_format.jsonl

eval_split_name=test

######## setup video+text features
feat_root=../data/charades/features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi
#### training
bsz=32


PYTHONPATH=$PYTHONPATH:. python train.py \
--dset_name ${dset_name} \
--debug \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--negative_saliency \
--attention_weights_align \
--max_q_l 77 \
--max_v_l u \
--max_windows 1 \
--clip_length 0 \
${@:1} 
