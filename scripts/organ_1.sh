set -ex
MODEL=organ
SEED=1
ALPHA=1
python scripts/run.py \
    --model $MODEL \
    --data data \
    --checkpoint_dir checkpoints/$MODEL/$MODEL\_$SEED\_$ALPHA \
    --device cuda:0 \
    --metrics data/samples/$MODEL/metrics_$MODEL\_$SEED\_$ALPHA.csv \
    --seed $SEED \
    --gen_path data/samples/$MODEL/$MODEL\_$SEED\_$ALPHA.csv \
    --receptor_file /home/jovyan/receptors/usp7.pdbqt \
    --vina_program /home/jovyan/bin/qvina02 \
    --temp_dir temp_dir \
    --alpha 0.1 \
    --num_sub_proc 16 \
    --box_center 2.860 4.819 92.848 \
    --box_size 17.112 17.038 14.958 \
    --additional_rewards docking \
    --exhaustiveness 8 \
    --num_modes 10 \
    --n_conf 3 \
    --pg_iters 100 \
    --n_batch 8 \
    --discriminator_epochs 2 \
    --save_freq 10
    # --model_load checkpoints/$MODEL/$MODEL\_$SEED\_$ALPHA

