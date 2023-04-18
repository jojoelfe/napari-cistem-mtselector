"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
import pathlib
import sqlite3
import pandas as pd
from napari.types import LabelsData, ImageData
from napari.layers import Image, Labels, Points
from pathlib import Path
from dask import delayed
import numpy as np
import dask.array as da
import mrcfile
from napari import Viewer
from rich.progress import track


if TYPE_CHECKING:
    import napari

def read_and_resize_mrc(row, x, y):
    filname = Path(row['project_dir'])/ "Assets" / "Images" / "Scaled" / Path(row['FILENAME']).name
    with mrcfile.open(filname) as mrc:
        data = mrc.data
    if len(data.shape) == 3:
        data = data[0]
    if data.shape[0] != y or data.shape[1] != x:
        data = np.pad(data, pad_width = [y-data.shape[0],x-data.shape[1]])
    return(data)

@magic_factory
def load_cistem_project(project: pathlib.Path, viewer: Viewer) -> Image:
    con = sqlite3.connect(project)
    data = pd.read_sql_query("SELECT IMAGE_ASSETS.IMAGE_ASSET_ID, FILENAME, DEFOCUS1, DEFOCUS2, IMAGE_ASSETS.PIXEL_SIZE, X_SIZE, Y_SIZE, DEFOCUS_ANGLE, IMAGE_ASSETS.SPHERICAL_ABERRATION, ESTIMATED_CTF_PARAMETERS.AMPLITUDE_CONTRAST, IMAGE_ASSETS.VOLTAGE FROM IMAGE_ASSETS INNER JOIN ESTIMATED_CTF_PARAMETERS ON IMAGE_ASSETS.CTF_ESTIMATION_ID = ESTIMATED_CTF_PARAMETERS.CTF_ESTIMATION_ID",con)
    con.close

    x_size = data["X_SIZE"].max()
    y_size = data["Y_SIZE"].max()
    scale = 1200 / (max(x_size,y_size))
    # Round up
    x_size = int(x_size * scale + 0.5)
    y_size = int(y_size * scale + 0.5)
    dtype=float
    lazy_imread = delayed(read_and_resize_mrc)  # lazy reader
    ffa = []
    data['project_dir'] = Path(project).parent

    for i,image in data.iterrows():
        ffa.append(lazy_imread(image, x_size, y_size))
    ffa = [da.from_delayed(lazy_imread, shape=(y_size,x_size), dtype=dtype) for lazy_imread in ffa]
    data['img_id'] = data.index
    data.reset_index(inplace=True)
    print(data)
    stack = da.stack(ffa, axis=0)
    image_obj = Image(stack, name = "cisTEM Images")
    image_obj.metadata['project'] = project
    image_obj.metadata['cistem_data'] = data
    return image_obj

@magic_factory
def filter_matches(image: Image, labels: LabelsData, tm_jobid: int = 1, preview: bool = True) -> Points:
    import tensorstore
    con = sqlite3.connect(image.metadata['project'])
    points = []
    colors = []
    if preview is False:
        max_job_id = pd.read_sql_query("SELECT MAX(TEMPLATE_MATCH_JOB_ID) FROM TEMPLATE_MATCH_LIST",con).iloc[0,0]
        max_tm_id = pd.read_sql_query(f"SELECT MAX(TEMPLATE_MATCH_ID) FROM TEMPLATE_MATCH_LIST",con).iloc[0,0]
    tm_jobs =  pd.read_sql_query(f"SELECT * FROM TEMPLATE_MATCH_LIST WHERE TEMPLATE_MATCH_JOB_ID = {str(tm_jobid)}", con)
    img_with_matches = image.metadata['cistem_data'].join(tm_jobs.set_index('IMAGE_ASSET_ID'), on='IMAGE_ASSET_ID', how='inner')
    ts = False
    if type(labels) == tensorstore.TensorStore:
        ts = True
    for i, row in track(img_with_matches.iterrows(), description="Filtering matches", total=len(img_with_matches)):
        x_size = row["X_SIZE"]
        y_size = row["Y_SIZE"]
        scale = 1200 / (max(x_size,y_size))
        pixel_size = row["PIXEL_SIZE"] / scale
       
        if preview is False:
            # Get row of tm_jobs where IMAGE_ASSET_ID = row['IMAGE_ASSET_ID']
            to_copy = tm_jobs.loc[tm_jobs['IMAGE_ASSET_ID'] == row['IMAGE_ASSET_ID']]
            insert_tm_info = to_copy.copy()
            insert_tm_info['TEMPLATE_MATCH_JOB_ID'] = max_job_id + 1
            insert_tm_info['TEMPLATE_MATCH_ID'] = max_tm_id + 1
            max_tm_id += 1
            insert_tm_info.to_sql(f"TEMPLATE_MATCH_LIST",con,if_exists='append',index=False)
        matches_list = pd.read_sql_query(f"SELECT * FROM TEMPLATE_MATCH_PEAK_LIST_{row['TEMPLATE_MATCH_ID']}",con)
        if preview is False:
            insert_matches_list = matches_list.iloc[:0,:].copy()
        for j,x in matches_list.iterrows():
            points.append([row['img_id'], x['Y_POSITION'] / pixel_size, x['X_POSITION'] / pixel_size])
            
            if labels is None:
                colors.append([1,1,1])
                continue
            if ts:
                lv = labels[row['img_id'],int(x['Y_POSITION'] / pixel_size), int(x['X_POSITION'] / pixel_size)].read().result()
            else:
                lv = labels[row['img_id'],int(x['Y_POSITION'] / pixel_size),int(x['X_POSITION'] / pixel_size)]
            if lv == 0:
                colors.append([1,0,0])
            else:
                colors.append([0,1,0])
                if preview is False:
                    insert_matches_list = insert_matches_list.append(x, ignore_index=True)
        if preview is False:
            insert_matches_list.to_sql(f"TEMPLATE_MATCH_PEAK_LIST_{max_tm_id}",con,if_exists='fail',index=False)
            insert_matches_list = matches_list.iloc[:0,:].copy()
            insert_matches_list.to_sql(f"TEMPLATE_MATCH_PEAK_CHANGE_LIST_{max_tm_id}",con,if_exists='fail',index=False)
    con.close
    return Points(points, name = "cisTEM Matches",face_color=colors)
