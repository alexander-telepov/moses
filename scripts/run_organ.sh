set -ex
MODEL=organ
SEED=1
mkdir -p checkpoints/$MODEL/$MODEL\_$SEED
python scripts/run.py \
    --model $MODEL \
    --data data \
    --checkpoint_dir checkpoints/$MODEL/$MODEL\_$SEED \
    --device cuda:0 \
    --metrics data/samples/$MODEL/metrics_$MODEL\_$SEED.csv \
    --seed $SEED \
    --gen_path data/samples/$MODEL/$MODEL\_$SEED.csv \
    --receptor_path /home/jovyan//receptors/usp7.pdbqt \
    --vina_path /home/jovyan/bin/qvina02 \
    --temp_dir temp_dir \
    --alpha 0.1 \
    --num_sub_proc 16 \
    --box_center 2.860 4.819 92.848 \
    --box_size 17.112 17.038 14.958 \
    --additional_rewards docking

