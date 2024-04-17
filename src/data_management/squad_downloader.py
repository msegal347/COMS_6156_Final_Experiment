import requests
import os

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print(f"Downloaded {save_path}")

def validate_file_size(file_path, min_size_kb):
    file_size_kb = os.path.getsize(file_path) / 1024
    if file_size_kb > min_size_kb:
        print(f"Validation passed for {file_path}")
    else:
        print(f"Validation failed for {file_path}, size: {file_size_kb} KB")

def main():
    squad_urls = {
        "SQuAD_1.1_train.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        "SQuAD_1.1_dev.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
        "SQuAD_2.0_train.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "SQuAD_2.0_dev.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    }
    
    # Adjust the base path as per the project structure
    base_path = os.path.join(os.path.dirname(__file__), "../../data/raw")
    
    # Ensure the base path exists
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Created directory {base_path}")

    for filename, url in squad_urls.items():
        save_path = os.path.join(base_path, filename)
        download_file(url, save_path)
        # Assuming a simple file size check for validation; adjust the min_size_kb as needed
        validate_file_size(save_path, 1000)  # Minimum file size in KB

if __name__ == "__main__":
    main()
