python -u ./src/channel_compressed_sensing.py \
    --pretrained-model-dir=./models/channel_H_64_16_2/ \
    \
    --dataset channel \
    --dataset-dir=/home/zby/datasets/channel/data_5.19/SNR25.tfrecords \
    --input-type full-input \
    --num-input-images 1 \
    --batch-size 1 \
    --scale-factor 2.5 \
    \
    --measurement-type pilot \
    --noise-std 0.1 \
    --num-measurements 500 \
    \
    --z-dim 80 \
    --pilot-dim 48 \
    --model-types dcgan \
    --mloss1_weight 0.0 \
    --mloss2_weight 1.0 \
    --zprior_weight 0.0 \
    --dloss1_weight 0.0 \
    --dloss2_weight 0.0 \
    \
    --optimizer-type adam \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --max-update-iter 2000 \
    --num-random-restarts 1 \
    \
    --save-images \
    --not-lazy \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 1
