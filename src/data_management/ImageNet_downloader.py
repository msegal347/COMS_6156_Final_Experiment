import requests, zipfile, io

def download_tiny_imagenet(url, save_path):
    print("Starting download of Tiny ImageNet...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=save_path)
    print("Download and extraction complete.")

def main():
    tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    # Assuming the script is executed from the project root
    save_path = "./data/raw/tiny_imagenet"  
    download_tiny_imagenet(tiny_imagenet_url, save_path)

if __name__ == "__main__":
    main()
