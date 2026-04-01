#!/bin/bash

# Inference script for 2.5D Brownian Bridge Diffusion Model (CT-to-MRI)
# Generates 3D MRI volumes from CT

module load python
conda activate flashattention
# python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/generate_bbdm_ct2mri.py \
#     --experiment_name "bbdm_ct2mri_SynthRAD_only_mni" \
#     --checkpoint_ver "epoch015" \
#     --test_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/test_metadata_synthrad.csv" \
#     --use_bf16 \
#     --batch_size 1 \
#     --num_workers 4 \
#     --seed 1337
    # --mni  # Uncomment for MNI-registered data

# --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_IntactBrain" \
# --test_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/test_metadata_total_IntactBrain.csv" \

# v1
# python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/generate_bbdm_ct2mri.py \
#     --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_n225_20260302" \
#     --checkpoint_ver "epoch023" \
#     --test_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/test_metadata_total_n225_20260302.csv" \
#     --use_bf16 \
#     --mni \
#     --batch_size 2 \
#     --num_workers 4 \
#     --seed 1337

# v2
# python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/generate_bbdm_ct2mri_v2.py \
#     --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_n226_20260302_v2" \
#     --checkpoint_ver "epoch023" \
#     --test_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/test_metadata_total_n226_20260302.csv" \
#     --use_bf16 \
#     --mni \
#     --use_ista \
#     --num_workers 4 \
#     --seed 1337

# v3 (75, 65, 56, 47, 38, 28, 18, 13, 9)
python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/generate_bbdm_ct2mri_v3.py \
    --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_n226_20260302_v3" \
    --checkpoint_ver "epoch065" \
    --test_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/test_metadata_total_n226_20260302.csv" \
    --train_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_n226_20260302.csv" \
    --use_bf16 \
    --use_ista \
    --mni \
    --num_workers 4 \
    --seed 1337

python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/generate_bbdm_ct2mri_v3.py \
    --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_n226_20260302_v3" \
    --checkpoint_ver "epoch047" \
    --test_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/test_metadata_total_n226_20260302.csv" \
    --train_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_n226_20260302.csv" \
    --use_bf16 \
    --use_ista \
    --mni \
    --num_workers 4 \
    --seed 1337

python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/generate_bbdm_ct2mri_v3.py \
    --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_n226_20260302_v3" \
    --checkpoint_ver "epoch005" \
    --test_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/test_metadata_total_n226_20260302.csv" \
    --train_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_n226_20260302.csv" \
    --use_bf16 \
    --use_ista \
    --mni \
    --num_workers 4 \
    --seed 1337


# python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/generate_bbdm_ct2mri_v3_inference.py \
#     --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_n226_20260302_v3" \
#     --checkpoint_ver "epoch040" \
#     --train_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_n226_20260302.csv" \
#     --ct_input_dir "/pscratch/sd/s/seojw/CT_to_MRI/CT_20260308/" \
#     --use_bf16 \
#     --use_ista \
#     --mni \
#     --num_workers 4 \
#     --seed 1337



# python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/generate_bbdm_ct2mri.py \
#     --experiment_name "bbdm_ct2mri_SynthRAD_only_mni_IntactBrain" \
#     --checkpoint_ver "epoch028" \
#     --test_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/test_metadata_synthrad_IntactBrain.csv" \
#     --use_bf16 \
#     --batch_size 1 \
#     --num_workers 4 \
#     --seed 1337

# python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/generate_bbdm_ct2mri.py \
#     --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_IntactBrain" \
#     --checkpoint_ver "epoch006" \
#     --test_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/test_metadata_total_IntactBrain.csv" \
#     --use_bf16 \
#     --batch_size 2 \
#     --num_workers 4 \
#     --seed 1337