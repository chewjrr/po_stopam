import sys
import json
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

def main():
    # Читаем входные данные из аргументов
    input_json = sys.argv[1]
    user_request = json.loads(input_json)
    
    # Ваш текущий код для работы с API
    url = "https://api.intelligence.io.solutions/api/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer io-v2-eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJvd25lciI6ImZiMTIyMDhjLTM1OWUtNGEzYi05MmE2LTExMzM1YWM5YjJmNCIsImV4cCI6NDg5NjU5MjM2OX0.Bo6sddCvnPaHljwD87AmZWdNBAOiag2HNt1pJdEgCVPFQgkjLwZpy-HhfFai5D-cvxFTUpEowxguq8cun5oPYw"
    }
    
    data = {
        "model": "mistralai/Ministral-8B-Instruct-2410",
        "messages": [
            {"role": "system", "content": "You are a route generator."},
            {"role": "user", 
             "content": f"Suggest places in Moscow for a walk. Return only a list of place names, one per line. Criteria: places: {', '.join(user_request['places'])}, walk time: {user_request['time']}, budget: {user_request['budget']}."}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=200)
        response.raise_for_status()
        api_data = response.json()
        
        content = api_data['choices'][0]['message']['content']
        places = re.findall(r"^\d*\.*\s*(.+)", content, re.MULTILINE)
        
        route = []
        for place in places:
            place = place.strip()
            if place:
                coords = get_coordinates(place)
                if coords:
                    route.append({"name": place, "coordinates": coords})
                time.sleep(1)
        
        # Выводим результат в stdout для Go
        print(json.dumps(route))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()