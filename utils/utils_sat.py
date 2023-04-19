# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:18:16 2023

@author: HP
"""
import ee
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


START_DATE = '2021-01-01'
END_DATE = '2021-04-30'
CLOUD_FILTER = 1
CLD_PRB_THRESH = 1
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50



def read_renabap():
    
    try:
        print("ANTES")
        df = gpd.read_file('./data/amba.geojson')
        print("DESPUES")
    except Exception as e:
        st.error(f"Error reading GeoPandas dataframe: {e}")
    
    return df

#



def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def croppge(of,nom,anio,min_lon, min_lat, max_lon, max_lat, delta=0, idx=0, TCI = False, remove_original = True):
    nom = str(nom)
    anio = str(anio)
    #min_lon, min_lat, max_lon, max_lat = df.geometry[idx].bounds
    min_lon = min_lon-delta
    min_lat = min_lat-delta
    max_lon = max_lon+delta
    max_lat = max_lat+delta

    bbox = box(min_lon, min_lat, max_lon, max_lat)
    crs = 'EPSG:4326' # WGS84

    with rio.open(of) as src:
        # Crop the raster using the bounding box
        out_image, out_transform = mask(src, [bbox], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "crs": crs})
    if TCI:
        cropped_file = './data/raster_cropped/'+nom+"_"+anio+"_"+"_cropped_TCI.tif"
    else:
        cropped_file = './data/raster_cropped/'+nom+"_"+anio+"_"+"_cropped.tif"
    with rio.open(cropped_file, "w", **out_meta) as dest:
        dest.write(out_image)
    fl_300 = True
    with rio.open(cropped_file) as src:
        print(src.width,src.height)
        if src.width<300 and src.height <300:
            fl_300 = False
    if not fl_300:
            #os.remove(cropped_file)
            croppge(of,nom,anio,min_lon, min_lat, max_lon, max_lat,delta = 0.005,TCI = TCI, remove_original = remove_original)
    #os.remove(of)
    return cropped_file

def custom_merge_works(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
    old_data[:] = np.maximum(old_data, new_data)  # <== NOTE old_data[:] updates the old data array *in place*


def predict_model(path,anio,nom,sw = 3):
    nom = str(nom)
    ge = rio.open(path)
    ix = int(np.ceil(ge.height/300))
    jx = int(np.ceil(ge.width/300))
    ix=ix*sw
    jx=jx*sw
    imagespath = './cropped/images/'
    if not os.path.isdir(imagespath):
        os.mkdir(imagespath)
    for i in range(ix):
        for j in range(jx):
            # Open the raster file
            with rio.open(path) as src:
                # Define the window for the crop
                win_height = 300  # height of the window
                win_width = 300  # width of the window
                row_start = 0+j*(300/sw)  # starting row for the window
                col_start = 0+i*(300/sw)  # starting column for the window
                window = Window(col_start, row_start, win_width, win_height)
                # Read the data for the window
                data = src.read(window=window)
                # Update the metadata for the cropped image
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": win_height,
                    "width": win_width,
                    "transform": rio.windows.transform(window, src.transform)
                })
                # Write the cropped image to a new file

                p = os.path.join(imagespath,f'output_crop_{win_height}_{win_height}_{row_start}_{col_start}.tif')

                with rio.open(p, 'w', **out_meta) as dst:
                    dst.write(data)
    ge.close()
    imlen= int(len(os.listdir(imagespath))/3)
               
    predict_config = PredictConfig(images_path='./cropped/', # ruta a las imagenes sobre las cuales queremos predecir
                                    results_path='./cropped/out/', # ruta de destino para nuestra predicción
                                    batch_size=imlen,
                                    model_path=os.path.join(os.getcwd(),'model', 'model.h5'),  #  ruta al modelo (.h5)
                                    height=160,
                                    width=160,
                                    n_channels=5,
                                    n_classes=1,
                                    class_weights= [1]) 
    predict(predict_config)
    anio = '2021'
    r = './cropped/out/'
    files_to_mosaic = glob.glob(r+"\\*")
    m = [rio.open(files_to_mosaic[x]) for x in range(len(files_to_mosaic))]
    arr, out_trans = merge(m, method = custom_merge_works)
    output_meta = m[0].meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "height": arr.shape[1],
            "width": arr.shape[2],
            "transform": out_trans,
        })


    outputdir = "C:/Users/HP/OneDrive/VizApp/apiPred/apiPred/output_pred/"#"./output_pred/"#os.path.join("C:/Users/HP/OneDrive/pytorch/torchgeo/renabap/output",prov)
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    fname = nom+"_"+str(anio)+".tif"#path.split("\\")[len(path.split("\\"))-1]

    with rio.open(os.path.join(outputdir,fname), "w", **output_meta) as p:
        p.write(arr)


    for k in m:
        k.close()
    for h in files_to_mosaic:
        os.remove(h)
    #shutil.rmtree(imagespath)
    
    return True,os.path.join(outputdir,fname)

def get_raster_gearth(df,iy=6,anio=2021,delta = 0.01, TCI = False,path_output= os.getcwd()):
    nom = str(iy) #df.renabap_id[iy].astype(str)
    sep17 = df[(df.renabap_id==iy)].reset_index(drop=True)
    prov = str(sep17.provincia[0])#.astype(str)
    
    area_barrio = sep17['geometry'].to_crs({'init': 'epsg:4326'})\
           .map(lambda p: p.area )[0]

    min_lon, min_lat, max_lon, max_lat = sep17.geometry[0].bounds
    
    min_lon -= delta
    min_lat -= delta
    max_lon += delta
    max_lat += delta
    
    coords = [[min_lon,min_lat],[min_lon,max_lat],[max_lon,max_lat],[max_lon,min_lat]]
    aoi = ee.Geometry.Polygon(coords)

    AOI = ee.Geometry.Polygon(coords)

    START_DATE = str(anio)+'-09-01'
    END_DATE = str(anio)+'-12-31'
    if anio == 2020:
        START_DATE = str(anio)+'-01-01'
        END_DATE = str(anio)+'-12-31'
    if anio == 2022:
        START_DATE = str(anio)+'-02-01'
        END_DATE = str(anio)+'-03-31'


    s2_sr_cld_col_eval = get_s2_sr_cld_col(AOI, START_DATE, END_DATE)
    a = anio


    ffa_db = ee.Image(ee.ImageCollection('COPERNICUS/S2_SR') 
                   .filterBounds(aoi) 
                   .filterDate(ee.Date(START_DATE), ee.Date(END_DATE))
                   .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)) 
                   .first() 
                   .clip(aoi))
    try:
        if TCI:
            bandas = ['TCI_R','TCI_G','TCI_B']
            link = ffa_db.select('TCI_R','TCI_G','TCI_B').getDownloadURL({
                'scale': 1,
                'crs': 'EPSG:4326',
                'fileFormat': 'GeoTIFF'})
        else:
            bandas = ['B2','B3','B4','B8','B11']
            link = ffa_db.select('B2','B3','B4','B8','B11').getDownloadURL({
                'scale': 1,
                'crs': 'EPSG:4326',
                'fileFormat': 'GeoTIFF'})
    except Exception as e:
        print(e)
        if 'total request size' in str(e).lower():
            t,of = get_raster_gearth(df=df, iy=iy,anio=anio,delta = delta-0.001, TCI = TCI, path_output = path_output)
            if t == True:
                return True,of
        else: 
            return e 
    
    
    file = 'raster.zip'
    
    
    response = request.urlretrieve(link, file)
    
    
    
    if not os.path.isdir(path_output):
        os.mkdir(path_output)
    with ZipFile(file, 'r') as zObject:
        zObject.extractall(path=path_output)
        
        
        
    raster_files = [os.path.join(path_output,i) for i in os.listdir(os.path.join(path_output))]
    rf = []
    for i in bandas:
        for j in raster_files:
            if i in j:
                rf.append(j)
    if not TCI:
        out10m_B11 = os.path.join(os.getcwd(),"B11_10m".join(rf[4].split("B11")))

        src = gdal.Open(rf[0])
        xres, yres = operator.itemgetter(1,5)(src.GetGeoTransform())
        gdal.Warp(out10m_B11, rf[4], xRes=xres, yRes=yres)
        os.remove(rf[4])

    
    raster_files = [os.path.join(path_output,i) for i in os.listdir(os.path.join(path_output))]
    rf = []
    for i in bandas:#['B2','B3','B4','B8','B11']:
        for j in raster_files:
            if i in j:
                rf.append(j)  
    with rio.open(rf[0]) as blue_raster:
        blue = blue_raster.read(1, masked=True)
        out_meta = blue_raster.meta.copy()
        out_meta.update({"count": len(rf)})
        if TCI:
            out_img = os.path.join(path_output,f'./data/raster/{nom}_{anio}_TCI.tif')
        else:
            out_img = os.path.join(path_output,f'./data/raster/{nom}_{anio}.tif')
        file_list = [rio.open(i) for i in rf]
        with rio.open(out_img, 'w', **out_meta) as dest:
            for band_nr, src in enumerate(file_list, start=1):
                dest.write(src.read(1, masked=True), band_nr)
    for k in file_list:
        k.close()
    for h in rf:
        os.remove(h)
    os.remove('raster.zip')
    return True, out_img

