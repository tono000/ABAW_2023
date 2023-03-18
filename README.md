

Affwild2-ABAW3 @ CVPR 2022 

How to run?
1. Create a python 3.9 environment using conda or other tools. `conda create -n abaw5 python=3.9`
2. Activate the environment `conda activate abaw5` and Install packages in requirements.txt with `pip install -r requirements.txt`, or manually.
3. Create npy files `python tools\data_preparation.py --root_dir [Path to root of ABAW5 folder] --out_dir [Path to folder that contain npy files]`
4. Create npy files for test set: Edit line 13-14 in `tools/prepare_test_data.py` and run `python tools/prepare_test_data.py`
5. Edit config file in conf/tmp.yaml, or create a new config file. To use **wandb logger**, install wandb logger, login to wandb, and edit logger param in config file.
6. Run `python main.py --cfg /path/to-config-file`, e.g. `python main.py --cfg /conf/tmp.yaml`

Dataset folder structure in our PC
```
 % In our setting, we set --root_dir and --out_dir to the same folder (dataset),
 % that also use for line 14 in `tools/prepare_test_data.py`
 dataset\
    5th_ABAW_Annotations\
    cropped_aligned\
    testset\
        AU_test_set_release.txt
        EXPR_test_set_release.txt
        VA_test_set_release.txt
    AU.npy
    AU_test.npy
    EXPR.npy
    EXPR_test.npy
    VA.npy
    VA_test.npy
```

`root_video_path` folder contains all video in ABAW5, line 13 in `tools/prepare_test_data.py`

FlashAttention v0.2.8