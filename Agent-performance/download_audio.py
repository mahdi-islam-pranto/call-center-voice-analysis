# Function to download the audio file from the URL and save it locally
import requests
import os

def download_audio_from_url(url, local_path):
    try:
        response = requests.get(url, stream=True, verify=False)
        # debug
        # print("Status Code:", response.status_code)
        # print("Headers:", response.headers)
        # print("Content-Type:", response.headers.get("Content-Type"))

        if response.status_code == 200:
            with open(local_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

            print("Saved file exists?", os.path.exists(local_path))
            return True
        else:
            print("Failed with status:", response.status_code)
            return False

    except Exception as e:
        print("Download error:", e)
        return False