import requests
from pprint import pprint
import time
import re

def get_coordinates(place_name, city="Москва"):
    url = "https://nominatim.openstreetmap.org/search"

    headers = {
        "User-Agent": "your@email.com",  # Укажите свой email или название приложения
        "Referer": "https://yourwebsite.com"  # Можно указать любой URL
    }

    params = {
        # "q": place_name,
        "q": f"{place_name}, {city}",
        "format": "json",
        "limit": 1
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    else:
        print(f"Ошибка {response.status_code}: {response.text}")
    return None

user_request = {
    "place": ["park", "cafe"],
    "time": "2 hours",
    "budget": "1500 rubles",
}

url = "https://api.intelligence.io.solutions/api/v1/chat/completions"
headers = {
    "accept": "application/json",
    "Authorization": "Bearer io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6ImZiMTIyMDhjLTM1OWUtNGEzYi05MmE2LTExMzM1YWM5YjJmNCIsImV4cCI6NDg5NjU5MjM2OX0.Bo6sddCvnPaHljwD87AmZWdNBAOiag2HNt1pJdEgCVPFQgkjLwZpy-HhfFai5D-cvxFTUpEowxguq8cun5oPYw",
}

data = {
    "model": "mistralai/Ministral-8B-Instruct-2410",
    "messages": [
        {"role": "system", "content": "You are a route generator."},
        {"role": "user",
         "content": f"Suggest places in Moscow for a walk. Return only a list of place names, one per line. "
                    f"Criteria: places: {', '.join(user_request['place'])}, walk time: {user_request['time']}, budget: {user_request['budget']}."}
    ]
}

try:
    response = requests.post(url, headers=headers, json=data, timeout=200)
    response.raise_for_status()
    api_data = response.json()

    if 'choices' in api_data and api_data['choices']:
        content = api_data['choices'][0]['message']['content']
        places = re.findall(r"^\d*\.*\s*(.+)", content, re.MULTILINE)  # Извлекаем только названия мест
    else:
        print("Error: Unexpected API response format")
        places = []

except requests.exceptions.RequestException as e:
    print(f"API request error: {e}")
    places = []

route = []
for place in places:
    place = place.strip()
    if place:
        coords = get_coordinates(place)
        if coords:
            route.append({"name": place.split('- ')[0], "coordinates": coords})
        time.sleep(1)  # Ограничение запросов (не более 1 запроса в секунду)

pprint(route)
