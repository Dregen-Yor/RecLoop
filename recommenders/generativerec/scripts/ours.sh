# Copyright (c) Meta Platforms, Inc. and affiliates.

# liger
# for dataset_name in Beauty Toys_and_Games Sports_and_Outdoors
for dataset_name in Office_Products
do
    python run.py \
        dataset=amazon \
        dataset.name=$dataset_name \
        seed=42 \
        device_id=1 \
        method=setting \
        test_method=liger \
        experiment_id="liger_$dataset_name"
done


# for dataset_name in steam
# do
#     python run.py \
#         dataset=steam \
#         dataset.name=$dataset_name \
#         seed=42 \
#         device_id=0 \
#         method=setting \
#         test_method=liger \
#         experiment_id="liger_$dataset_name"
# done