import requests
import zipfile
import io
import os

def download_tiny_imagenet(url, save_path):
    print("Starting download of Tiny ImageNet...")
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raises an HTTPError for bad responses

            if "application/zip" in response.headers.get("Content-Type", ""):
                zip_path = os.path.join(save_path, 'tiny-imagenet-200.zip')
                with open(zip_path, 'wb') as file:
                    for data in response.iter_content(1024):  # 1 Kibibyte blocks
                        file.write(data)

                print("Download complete. Extracting file...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(path=save_path)
                print("Extraction complete.")

            else:
                print("Failed to download: Not a zip file.")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")  # Specific HTTP related errors
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")  # Catch any other errors that might occur
    except zipfile.BadZipFile:
        print("Error extracting the file: The downloaded file may be corrupt.")

def main():
    tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    # Assuming the script is executed from the project root
    save_path = "./data/raw/tiny_imagenet"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory {save_path}")

    download_tiny_imagenet(tiny_imagenet_url, save_path)

if __name__ == "__main__":
    main()
