python -u ./src/channel_compressed_sensing.py \
    --pretrained-model-dir=./models/channel_H_64_64/ \
    \
    --dataset channel \
    --dataset-dir=/home/zby/datasets/channel/data_5.22/noNoise.tfrecords \
    --input-height 64 \
    --input-width 64 \
    --input-channel 2 \
    --input-type full-input \
    --num-input-images 1 \
    --batch-size 1 \
    --scale-factor 2.5 \
    \
    --measurement-type pilot \
    --noise-std 0.1 \
    --num-measurements 500 \
    \
    --z-dim 128 \
    --pilot-dim 192 \
    --model-types pggan \
    --mloss1_weight 0.0 \
    --mloss2_weight 1.0 \
    --zprior_weight 0.0 \
    --dloss1_weight 0.0 \
    --dloss2_weight 0.0 \
    \
    --optimizer-type adam \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --max-update-iter 1 \
    --num-random-restarts 1 \
    \
    --save-images \
    --not-lazy \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 1
