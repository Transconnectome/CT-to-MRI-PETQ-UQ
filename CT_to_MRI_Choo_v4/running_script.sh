#!/bin/bash


#--train_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_IntactBrain_siteAonly.csv" \
#--val_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/val_metadata_total_IntactBrain_siteAonly.csv" \

module load python
conda activate flashattention
# python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/train_bbdm_ct2mri.py \
#       --train_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_n225_20260302.csv" \
#       --val_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/val_metadata_total_n225_20260302.csv" \
#       --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_n225_20260302" \
#       --batch_size 2 \
#       --grad_accum_steps 2 \
#       --max_epochs 10000 \
#       --use_histogram \
#       --mni

# python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/train_bbdm_ct2mri_wandb.py \
#python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/train_bbdm_ct2mri_wandb_v2.py \
# python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/train_bbdm_ct2mri_wandb_v2.py \
#       --train_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_n226_20260302.csv" \
#       --val_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/val_metadata_total_n226_20260302.csv" \
#       --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_n226_20260302_v2" \
#       --resume_ckpt "/pscratch/sd/s/seojw/CT_to_MRI/checkpoints/bbdm_ct2mri_SynthRAD+severance_mni_n226_20260302_v2/bbdm_ct2mri_SynthRAD+severance_mni_n226_20260302_v2_epoch005.pt" \
#       --wandb_project "CT_to_MRI" \
#       --wandb_entity "connectome" \
#       --batch_size 2 \
#       --grad_accum_steps 8 \
#       --max_epochs 10000 \
#       --use_histogram \
#       --mni


# #python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/train_bbdm_ct2mri_wandb_v2.py \

python /pscratch/sd/s/seojw/CT_to_MRI_Choo/Choo/train_bbdm_ct2mri_wandb_v3.py \
      --train_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_n226_20260302.csv" \
      --val_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/val_metadata_total_n226_20260302.csv" \
      --experiment_name "bbdm_ct2mri_SynthRAD+severance_mni_n226_20260302_v3" \
      --wandb_project "CT_to_MRI" \
      --wandb_entity "connectome" \
      --batch_size 2 \
      --grad_accum_steps 8 \
      --max_epochs 10000 \
      --mni



# python /pscratch/sd/s/seojw/CT_to_MRI/train_ldm_ct2t1_tmp.py \
# 	--experiment_name 'ldm_ddpm_ct2t1_saturate_fixed_SynthRAD+severance_mni_IntactBrain_siteAonly_lr1e-4_n225_20260302' \
# 	--vq_ckpt "/pscratch/sd/s/seojw/CT_to_MRI/checkpoints/ct_to_mri_spade_final_latent_2/ct_to_mri_spade_final_latent_2_epoch1250.pt" \
# 	--train_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_n225_20260302.csv" \
#     --val_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/val_metadata_total_n225_20260302.csv" \
#     --mni \
#     --lr 1e-4 \
# 	--val_every 20

# python train_ldm_ct2t1_tmp.py \
#       --train_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/train_metadata_total_IntactBrain_siteAonly.csv" \
#       --val_metadata "/pscratch/sd/s/seojw/CT_to_MRI/metadata/val_metadata_total_IntactBrain_siteAonly.csv" \
# 	  --val_every 10 \
# 	  --lr 1e-5 \
#       --latent \
#       --latent_dir /pscratch/sd/s/seojw/CT_to_MRI/latents \
#       --experiment_name 'ldm_ddpm_ct2t1_saturate_fixed_SynthRAD+severance_mni_IntactBrain_siteAonly_latent_ch256_lr1e-5' \
# 	  --model_channels 256