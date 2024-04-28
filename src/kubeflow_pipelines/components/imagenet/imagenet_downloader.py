import kfp
from kfp.v2.dsl import component, OutputPath

base_image = 'gcr.io/coms-6156-kubeflow/imagenet:latest' 

@component(base_image=base_image)
def download_tiny_imagenet(base_path: OutputPath()):
    import requests
    import zipfile
    import io
    import os

    def download_and_extract(url, save_path):
        """Downloads and extracts zip files from a URL into a given path."""
        print("Starting download and extraction...")
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  
                
                zip_path = os.path.join(save_path, 'tiny-imagenet-200.zip')
                with open(zip_path, 'wb') as file:
                    for data in response.iter_content(chunk_size=1024):  
                        file.write(data)
                
                print("Download complete. Extracting file...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(path=save_path)
                print("Extraction complete.")

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the file: {e}")
        except zipfile.BadZipFile:
            print("Error extracting the file: The downloaded file may be corrupt.")

    # Ensure the base path exists
    os.makedirs(base_path, exist_ok=True)
    print(f"Created directory {base_path}")

    tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    download_and_extract(tiny_imagenet_url, base_path)
