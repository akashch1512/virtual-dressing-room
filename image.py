import requests

search_query = "shirt transparent background PNG full body image"
api_key = "05a7c7f3ac3c20eac65dee285bd57a372c42fe04e761a2c5b0237fb77a0c4c29"

params = {
    "engine": "google",
    "q": search_query,
    "tbm": "isch",  # image search
    "api_key": api_key
}

response = requests.get("https://serpapi.com/search", params=params)
data = response.json()

for image in data["images_results"]:
    print(image["thumbnail"])  # URL of the image
