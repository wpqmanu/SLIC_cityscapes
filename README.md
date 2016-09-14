# SLIC_cityscapes
Pipeline:
0. Prepare folders (original image, ground truth, label map, etc.) Make sure they have same amount of files.
1. Go to SLIC_new_cityscapes_training_server_parallel_spark.py to run modified SLIC algorithm to run superpixels.
2. Go to SLIC_merge_superpixels_parallel.py to merge the superpixels obtained from 1, using connected components.
3. (Optional, for random forest of xgboost): Go to feature_extraction to extract features from the merged superpixels.
4. Run classifier at your choice.