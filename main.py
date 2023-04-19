from utils import utils_sat as mk
import ee
from fastapi import FastAPI
import os
import shutil
import geopandas as gpd
import os
import numpy as np
from rasterio.windows import Window
from rasterio.merge import merge
from urllib import request
from zipfile import *
from osgeo import gdal
from unetseg.predict import PredictConfig, predict
from shapely.geometry import box
import glob
import operator
import rasterio as rio
from rasterio.mask import mask
#uvicorn main:app --reload

app = FastAPI()
df = mk.read_renabap()

ee.Authenticate()
ee.Initialize()



START_DATE = '2021-01-01'
END_DATE = '2021-04-30'
CLOUD_FILTER = 1
CLD_PRB_THRESH = 1
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/run/{iy}")
def get_raster_google_earth(iy: int,TCI: bool = False,predict: bool = True,anio: int = 2022):
    _, of = mk.get_raster_gearth(df,iy=iy,anio=anio,delta = 0.01, TCI = TCI,path_output= os.getcwd())
    gdf = df[df.renabap_id==iy].reset_index(drop=True)
    min_lon, min_lat, max_lon, max_lat = gdf.geometry[0].bounds
    pf = mk.croppge(of,iy,anio,min_lon, min_lat, max_lon, max_lat, idx = 0,TCI=TCI)
    
    #http://localhost:8000/raster_google_earth/10?TCI=True&predict=False
    if predict:
        _, predtif = mk.predict_model(pf,anio,iy)
        return predtif
    else:
        return pf 
