# 3D Scene Analogies
Official PyTorch implementation of **Learning 3D Scene Analogies with Neural Contextual Scene Maps (ICCV 2025)** [[Paper]](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Learning_3D_Scene_Analogies_with_Neural_Contextual_Scene_Maps_ICCV_2025_paper.html) [[Video]](https://www.youtube.com/watch?v=WTwbSAqTpE8).

[<img src="overview.png" width="600"/>](overview.png)

From a pair of 3D scenes, **3D scene anologies** are defined as dense maps that connect regions sharing similar spatial context.

In this repository, we provide the implementation and instructions for running our calibration method. If you have any questions regarding the implementation, please leave an issue or contact 82magnolia@snu.ac.kr.

## Installation
First install pytorch and pytorch3d using the following command:
```
conda create -n nrfield python=3.9
conda activate nrfield
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
# For ubuntu 22.04, try the following instead
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 numpy<1.9 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```

Then, install the remaining dependencies with the following command:
```
pip install -r requirements.txt
```

## Dataset Preparation (3D-FRONT & ARKitScenes)

First, run the following command to make a `data/` folder.
```
cd $PATH_TO_REPO
mkdir data/
```

### 3D-FRONT
For 3D-FRONT, download all files from the following [link](https://tianchi.aliyun.com/dataset/65347), unzip, and organize the files in the following structure under the `data/` folder.

    3d_scene_analogies/data
    └── 3D-FRONT
    └── 3D-FRONT-texture
    └── 3D-FRONT-model    

### ARKitScenes
For ARKitScenes, first clone the following [repository](https://github.com/apple/ARKitScenes).
Then, run the following command within the repository to download the `train/test` 3D meshes created from RGB-D fusion (further instructions available [here](https://github.com/apple/ARKitScenes/blob/main/DATA.md)).
```
python3 download_data.py raw --video_id_csv raw/raw_train_val_splits.csv --download_dir ~/Downloads/tmp/ \
--raw_dataset_assets mesh annotation
```
Then, rename `Training` to `train` and `Validation` to `test`, and organize the files in the following structure under the `data/` folder.

    3d_scene_analogies/data
    └── arkit_scenes
        ├── train
        └── test

## Citation
If you find this repository useful, please cite

```bibtex
@InProceedings{Kim_2025_ICCV,
    author    = {Kim, Junho and Bae, Gwangtak and Lee, Eun Sun and Kim, Young Min},
    title     = {Learning 3D Scene Analogies with Neural Contextual Scene Maps},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {7828-7840}
}
```