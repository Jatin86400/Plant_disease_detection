# Plant_disease_detection
A deep learning model for vision based plant disease detection

### Dataset
Download the plant village dataset from the following github repository:
```
git clone https://github.com/spMohanty/PlantVillage-Dataset
```
Once dataset is downloaded, open `data_setup.ipynb` and do the partition of the dataset. 

It will create 3 files, `trainlist.txt`, `testlist.txt`, and `vallist.txt`

Once data partition is created, change `data_dir` in `resnet_tl.py` and also the `image_files` accordingly.

Change the path of `config_file_dir` according to your system.

Once all is done you are ready to train.

### Training 

Use cfgs to control the training hyperparameters.
To start training.
```
python3 resnet_tl.py --w 0 --device cuda:0 -v v2
```
`--device`: `cuda:0` by default. If you donot have gpu, use `cpu`.

`--w`: if 1 it will use wandb for logging the results. If 0 it will be off.

`--v`: The version of cfg file you are using. By defaults `v2`



