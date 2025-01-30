#!/bin/bash
# 1. Setup the Eiffel Tower dataset
# Download all year's data for the Eiffel Tower dataset
python3 datasetup.py download --dataset Eiffel-Tower --delete_archive

# Create Binary files for each year's dataset
python3 datasetup.py create_colmap_bin_files --colmap_model_path Eiffel-Tower/2015/sfm
python3 datasetup.py create_colmap_bin_files --colmap_model_path Eiffel-Tower/2016/sfm
python3 datasetup.py create_colmap_bin_files --colmap_model_path Eiffel-Tower/2018/sfm
python3 datasetup.py create_colmap_bin_files --colmap_model_path Eiffel-Tower/2020/sfm

# Downscale images and depth files for each year's dataset
python3 datasetup.py downscale_images --images_path Eiffel-Tower/2015/images --downscale_factor 3
python3 datasetup.py downscale_images --images_path Eiffel-Tower/2016/images --downscale_factor 3
python3 datasetup.py downscale_images --images_path Eiffel-Tower/2018/images --downscale_factor 3
python3 datasetup.py downscale_images --images_path Eiffel-Tower/2020/images --downscale_factor 3



# 2. Setup the SeaThru-NeRF dataset
# Download and extract dataset 
python3 datasetup.py download --dataset_url https://docs.google.com/uc?export=download&id=1RzojBFvBWjUUhuJb95xJPSNP3nJwZWaT --download_root Seathru-NeRF --delete_archive

# Rename the images folders
mv Seathru-NeRF/Curasao/images_wb Seathru-NeRF/Curasao/images
mv Seathru-NeRF/IUI3-RedSea/images_wb Seathru-NeRF/IUI3-RedSea/images
mv Seathru-NeRF/JapaneseGradens-RedSea/images_wb Seathru-NeRF/JapaneseGradens-RedSea/images
mv Seathru-NeRF/Panama/images_wb Seathru-NeRF/Panama/images

# Downscale images and depth files for each year's dataset
python3 datasetup.py downscale_images --images_path Seathru-NeRF/Curasao/images  --downscale_factor 3
python3 datasetup.py downscale_images --images_path Seathru-NeRF/IUI3-RedSea/images --downscale_factor 3
python3 datasetup.py downscale_images --images_path Seathru-NeRF/JapaneseGradens-RedSea/images --downscale_factor 3
python3 datasetup.py downscale_images --images_path Seathru-NeRF/Panama/images  --downscale_factor 3


