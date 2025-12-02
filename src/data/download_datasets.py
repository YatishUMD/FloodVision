import argparse
import os
import subprocess
import ftplib
from huggingface_hub import hf_hub_download

# downloads sen1floods11 using gsutil
def download_sen1floods11(root_dir):
    # destination path for sen1floods11 data
    dest = os.path.join(root_dir, "sen1floods11")
    # create the directory if it doesn't exist
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading Sen1Floods11 to {dest}...")
    try:
        # run the gsutil rsync command to download files from google cloud storage bucket
        # -m enables multi-threading, rsync syncs directories, -r is recursive
        subprocess.run(["gsutil", "-m", "rsync", "-r", "gs://sen1floods11", dest], check=True)
    except FileNotFoundError:
        print("Error: 'gsutil' not found. Install Google Cloud SDK.")

# downloads cloudsen12 subset from huggingface
def download_cloudsen12(root_dir):
    # destination path for cloudsen12 data
    dest = os.path.join(root_dir, "cloudsen12")
    # create the directory if it doesn't exist
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading CloudSEN12 to {dest}...")
    
    # specific part files to download from repo
    files = ["cloudsen12-l1c.0000.part.taco", "cloudsen12-l1c.0001.part.taco"]
    # iterate thru each file and download using huggingface hub api
    for f in files:
        hf_hub_download(repo_id="tacofoundation/CloudSEN12", filename=f, repo_type="dataset", local_dir=dest)

# downloads sen12ms subset via ftp
def download_sen12ms(root_dir):
    # destination path for sen12ms data
    dest = os.path.join(root_dir, "sen12ms")
    # create directory if missing
    os.makedirs(dest, exist_ok=True)
    # specific filename to retrieve from the ftp server
    fname = "ROIs1158_spring_s1.tar.gz"
    # local path where file will be saved
    local_path = os.path.join(dest, fname)
    
    print(f"Downloading {fname} to {dest}...")
    # connect to technical university of munich ftp server
    # login with the provided public credentials
    with ftplib.FTP("dataserv.ub.tum.de") as ftp:
        ftp.login("m1554803", "m1554803")
        with open(local_path, "wb") as f:
            # retrieve the file in binary mode and write to local file
            ftp.retrbinary(f"RETR {fname}", f.write)

if __name__ == "__main__":
    # absolute path of the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # navigate to the project root directory
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    # default data directory inside the project root
    default_data_dir = os.path.join(project_root, "data")

    # argument parser
    parser = argparse.ArgumentParser()
    # argument for selecting which dataset to download (or all)
    # optional argument for output directory defaulting to project/data
    parser.add_argument("dataset", choices=["sen1floods11", "cloudsen12", "sen12ms", "all"], help="Dataset to download")
    parser.add_argument("--output", default=default_data_dir, help="Output directory")
    args = parser.parse_args()

    # trigger particular download based on user selection
    if args.dataset in ["sen1floods11", "all"]: download_sen1floods11(args.output)
    if args.dataset in ["cloudsen12", "all"]: download_cloudsen12(args.output)
    if args.dataset in ["sen12ms", "all"]: download_sen12ms(args.output)
