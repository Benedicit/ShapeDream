
import zipfile
import shutil

import objaverse
from tqdm import tqdm

import os
from pathlib import Path

import json,requests
import csv
import pandas as pd
import objaverse.xl as oxl

OBJAVERSE = "objaverse"
REDWOOD = "redwood"
GSO = "gso"
SHAPENET = "shapenet"

working_dir = os.path.dirname(os.path.realpath(__file__))
original_datasets_dirs = {
    OBJAVERSE: "/.objaverse",
    REDWOOD: "/.redwood",
    GSO: "/.gso",
    SHAPENET: "/.shapenet"
}
dataset_dir = working_dir + "/dataset"
SUPPORTED_FILES = ("**/*.glb", "**/*.gltf", "**/*.obj", "**/*.ply", "**/*.stl")

GSO_FILE_COUNT = 1046 # NOTE: Some will come corrupted when downloading via the script...


def download_from_objaverse():
    objaverse_dir = working_dir + original_datasets_dirs[OBJAVERSE]
    os.makedirs(objaverse_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    objaverse_pp = pd.read_json("hf://datasets/cindyxl/ObjaversePlusPlus/annotated_800k.json").copy()

    quality_levels = [2, 3]
    filtered_models = objaverse_pp[
        ((objaverse_pp['style'] == 'realistic') | (objaverse_pp['style'] == 'scanned')) &
        (objaverse_pp['score'].isin(quality_levels)
         & (objaverse_pp['is_multi_object'] == "false")
         & (objaverse_pp['is_scene'] == "false")
         & (objaverse_pp['is_transparent'] == "false")
         & (objaverse_pp['is_figure'] == "false")
         )
        ]

    annotations = oxl.get_annotations()
    temp = pd.DataFrame(objaverse.load_uids(), columns=["UID"]).merge(filtered_models, how="left", on="UID")

    to_download = annotations[annotations.index.isin(filtered_models.index[:2])]



    print(f"Downloading {len(to_download)} filtered models from Objaverse")

    # Download filtered models
    objects = oxl.download_objects(
        objects=to_download,
        download_dir=objaverse_dir,
        download_processes=16,
        save_repo_format=None,
    )




def download_from_shapenet():
    #TODO
    pass



def unpack_gso():
    gso_dir = Path(working_dir + "/.gso")
    zip_files = [f for f in gso_dir.iterdir() if f.suffix.lower() == ".zip"]
    with tqdm(
            total=len(zip_files), unit="B", unit_scale=True, desc="Extracting GSO Files", leave=True
    ) as pbar:
        for file in zip_files:
            try:
                # Extract ZIP file
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(gso_dir)
                mesh_path = gso_dir / "meshes" / "model.obj"
                extracted_mesh = gso_dir / (file.stem + ".obj")
                shutil.move(mesh_path, extracted_mesh)
                shutil.rmtree(gso_dir / "meshes")
                shutil.rmtree(gso_dir / "materials")
                shutil.rmtree(gso_dir / "thumbnails")
                file.unlink()
            except zipfile.BadZipFile:
                print(f"{file} was not a zip file")
                file.unlink()
            except Exception as e:
                print(e)
            finally:
                pbar.update(1)

def generate_template_label_file_gso():
    """
    Helper function; won't be needed by the end user, as the final label file will be provided
    """
    gso_dir = Path(working_dir + "/.gso")
    models = [f for f in gso_dir.iterdir() if f.suffix.lower() == ".obj"]
    csv_path = Path(working_dir + "/gso_labels_template.csv")
    models.sort()
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])
        for model in models:
            writer.writerow([model.name, "object"])




def download_from_gso(unpack: bool = True):
    """
    Based on the script provided by Gazebo
    """

    owner_name = "GoogleResearch"
    collection_name = "Scanned Objects by Google Research"
    print("Downloading models from GSO Dataset")

    page = 1
    count = 0

    # The Fuel server URL.
    base_url ='https://fuel.gazebosim.org/'

    # Fuel server version.
    fuel_version = '1.0'

    # Path to get the models in the collection
    next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)

    # Path to download a single model in the collection
    download_url = base_url + fuel_version + '/{}/models/'.format(owner_name)

    with tqdm(
            total=GSO_FILE_COUNT, unit_scale=False, desc="GSO Models", leave=True
    ) as pbar:
    # Iterate over the pages
        while True:
            url = base_url + fuel_version + next_url

            # Get the contents of the current page.
            r = requests.get(url)

            if not r or not r.text:
                break

            # Convert to JSON
            models = json.loads(r.text)

            # Compute the next page's URL
            page = page + 1
            next_url = f'/models?page={page}&per_page=100&q=collections:{collection_name}'

            # Download each model
            for model in models:
                count += 1
                model_name = model['name']
                download = requests.get(download_url + model_name + '.zip', stream=True)
                with open(f"{working_dir}/.gso/{model_name}.zip", 'wb') as fd:
                    for chunk in download.iter_content(chunk_size=1024*1024):
                        fd.write(chunk)
                pbar.update(1)
    print('Done.')
    if unpack:
        unpack_gso()

def download_from_datasets(datasets: list):
    for d in datasets:
        if d == GSO:
            download_from_gso()
            unpack_gso()
            generate_template_label_file_gso()
        elif d == OBJAVERSE:
            pass
        elif d == SHAPENET:
            pass
        else:
            print(f"Dataset {d} isn't used by us")

if __name__ == '__main__':
    download_from_objaverse()