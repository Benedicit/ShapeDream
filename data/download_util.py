
import zipfile
import shutil
from tqdm import tqdm

import os
from pathlib import Path

import json,requests
import csv

import objaverse.xl as oxl

OBJAVERSE = "objaverse"
REDWOOD = "redwood"
GSO = "gso"

working_dir = os.path.dirname(os.path.realpath(__file__))
original_datasets_dirs = {
    OBJAVERSE: "/.objaverse",
    REDWOOD: "/.redwood",
    GSO: "/.gso",
}
dataset_dir = working_dir + "/dataset"
SUPPORTED_FILES = ("**/*.glb", "**/*.gltf", "**/*.obj", "**/*.ply", "**/*.stl")

GSO_FILE_COUNT = 1046 # NOTE: Some are corrupted when downloading via the script...


def download_from_objaverse():
    objaverse_dir = working_dir + original_datasets_dirs[OBJAVERSE]
    os.makedirs(objaverse_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    annotations = oxl.get_alignment_annotations(download_dir=objaverse_dir)
    #TODO: Scale up
    sampled_df = annotations.groupby('source').apply(lambda x: x.sample(16)).reset_index(drop=True)
    oxl.download_objects(download_dir=objaverse_dir, objects=sampled_df)

def download_from_redwood():
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
    Helper function; won't be needed by end user, as the final label file will be provided
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



owner_name = "GoogleResearch"
collection_name = "Scanned Objects by Google Research"

def download_from_gso(unpack: bool = True):
    """
    Based on the script provided by gazebo
    """

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
                #print ('Downloading (%d) %s' % (count, model_name))
                download = requests.get(download_url + model_name + '.zip', stream=True)
                with open(f"{working_dir}/.gso/{model_name}.zip", 'wb') as fd:
                    for chunk in download.iter_content(chunk_size=1024*1024):
                        fd.write(chunk)
                pbar.update(1)
    print('Done.')
    if unpack:
        unpack_gso()

def download_from_datasets():
    #TODO
    pass

if __name__ == '__main__':
    download_from_gso()
    unpack_gso()
    generate_template_label_file_gso()