
# Copyright (c) Meta Platforms, Inc. and affiliates.

# tiger
for dataset_name in Office_Products
do   
    /home/aizoo/miniconda3/envs/torch/bin/python run.py \
        dataset=amazon \
        dataset.name=$dataset_name \
        seed=42 \
        device_id=1 \
        method=base \
        test_method=tiger \
        experiment_id="tiger_$dataset_name"
done
