# import overpy
# import networkx as nx
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import folium
#
# # --- 1. Запрос данных из OpenStreetMap через Overpass API ---
# api = overpy.Overpass()
#
# def get_graph(lat, lon, radius=1000):
#     """Запрашивает данные о дорогах для пешеходов через Overpass API и строит граф."""
#     query = f"""
#         [out:json];
#         way["highway"~"footway|path|pedestrian|cycleway"]
#         (around:{radius}, {lat}, {lon});
#         (._;>;);
#         out body;
#     """
#     result = api.query(query)
#
#     graph = nx.Graph()
#     for way in result.ways:
#         nodes = way.nodes
#         for i in range(len(nodes) - 1):
#             node1 = nodes[i]
#             node2 = nodes[i + 1]
#             graph.add_node(node1.id, lat=float(node1.lat), lon=float(node1.lon))
#             graph.add_node(node2.id, lat=float(node2.lat), lon=float(node2.lon))
#             graph.add_edge(node1.id, node2.id, weight=1)  # Вес можно настраивать
#
#     return graph
#
#
# # --- 2. Генерация обучающего датасета ---
# def generate_training_data(graph, num_samples=1000):
#     nodes = list(graph.nodes)
#
#     if len(nodes) < 2:
#         raise ValueError("Graph has too few nodes for route generation!")
#
#     data = []
#     for _ in range(num_samples):
#         start, end = random.sample(nodes, 2)
#         try:
#             path = nx.shortest_path(graph, start, end, weight="weight")
#         except nx.NetworkXNoPath:
#             continue  # Если путь не найден, пропускаем
#
#         if path:
#             features = np.array([[graph.nodes[n]["lat"], graph.nodes[n]["lon"]] for n in path])
#             target = np.array(path)
#             data.append((features, target))
#
#     return data
#
#
# # --- 3. Создание нейросети ---
# class RouteNN(nn.Module):
#     def __init__(self, input_size=2, hidden_size=32, output_size=1):
#         super(RouteNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
#
#
# # --- 4. Обучение модели ---
# def train_model(graph):
#     print(f"Количество узлов в графе: {len(graph.nodes)}")
#     print(f"Количество рёбер в графе: {len(graph.edges)}")
#
#     dataset = generate_training_data(graph)
#     model = RouteNN(input_size=2, hidden_size=32, output_size=1)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     num_epochs = 500  # Количество эпох
#
#     for epoch in range(num_epochs):
#         total_loss = 0
#
#         for i, (features, target) in enumerate(dataset):
#             features = torch.tensor(features, dtype=torch.float32)
#             target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)  # Добавляем измерение
#
#             optimizer.zero_grad()
#             output = model(features)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#
#             if i % 50 == 0:
#                 print(f"Эпоха [{epoch + 1}/{num_epochs}], Батч [{i}/{len(dataset)}], Loss: {loss.item():.4f}")
#
#         # Выводим средний loss за эпоху
#         print(f"Эпоха [{epoch + 1}/{num_epochs}] завершена. Средний Loss: {total_loss / len(dataset):.6f}")
#
#     print("Обучение завершено!")
#     return model
#
#
# # --- 5. Генерация маршрута через нейросеть ---
# def generate_route(graph, model, start, end):
#     input_data = np.array([[graph.nodes[start]['lat'], graph.nodes[start]['lon']]])
#     input_tensor = torch.tensor(input_data, dtype=torch.float32)
#
#     predicted_node = int(model(input_tensor).item())
#     if predicted_node not in graph.nodes:
#         predicted_node = start  # Фоллбэк, если модель выдает некорректный узел
#
#     try:
#         route = nx.shortest_path(graph, start, predicted_node, weight="weight") + \
#                 nx.shortest_path(graph, predicted_node, end, weight="weight")[1:]
#     except nx.NetworkXNoPath:
#         route = nx.shortest_path(graph, start, end, weight="weight")  # Если путь не найден
#
#     return route
#
#
# # --- 6. Визуализация маршрута ---
# def plot_route(graph, route):
#     m = folium.Map(location=[graph.nodes[route[0]]['lat'], graph.nodes[route[0]]['lon']], zoom_start=15)
#     coords = [(graph.nodes[n]['lat'], graph.nodes[n]['lon']) for n in route]
#     folium.PolyLine(coords, color="blue", weight=5).add_to(m)
#     m.save("generated_route.html")
#
#
# # --- Основной запуск ---
# # Указываем координаты центра поиска
# center_lat, center_lon = 55.758694, 37.658033  # Москва
#
# graph = get_graph(center_lat, center_lon)
#
# if graph.nodes:
#     model = train_model(graph)
#     start, end = list(graph.nodes)[0], list(graph.nodes)[-1]
#     route = generate_route(graph, model, start, end)
#     plot_route(graph, route)
#     print("Маршрут создан! Открывайте generated_route.html")
# else:
#     print("Ошибка: Граф пустой, маршруты не могут быть созданы.")



import overpy
import networkx as nx
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import folium

# --- 1. Запрос данных из OpenStreetMap через Overpass API ---
api = overpy.Overpass()

def get_graph(lat, lon, radius=1000):
    """Запрашивает данные о дорогах для пешеходов через Overpass API и строит граф."""
    query = f"""
        [out:json];
        way["highway"~"footway|path|pedestrian|cycleway"]
        (around:{radius}, {lat}, {lon});
        (._;>;);
        out body;
    """
    result = api.query(query)

    graph = nx.Graph()
    for way in result.ways:
        nodes = way.nodes
        for i in range(len(nodes) - 1):
            node1 = nodes[i]
            node2 = nodes[i + 1]
            graph.add_node(node1.id, lat=float(node1.lat), lon=float(node1.lon))
            graph.add_node(node2.id, lat=float(node2.lat), lon=float(node2.lon))
            graph.add_edge(node1.id, node2.id, weight=1)  # Вес можно настраивать

    return graph

def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

# --- 2. Генерация обучающего датасета ---
def generate_training_data(graph, num_samples=1000):
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        raise ValueError("Graph has too few nodes for route generation!")

    # Определяем границы координат для нормализации
    lat_values = [graph.nodes[n]["lat"] for n in nodes]
    lon_values = [graph.nodes[n]["lon"] for n in nodes]
    min_lat, max_lat = min(lat_values), max(lat_values)
    min_lon, max_lon = min(lon_values), max(lon_values)

    data = []
    for _ in range(num_samples):
        start, end = random.sample(nodes, 2)

        try:
            path = nx.shortest_path(graph, start, end, weight="length")
        except nx.NetworkXNoPath:
            continue

        if path:
            features = np.array([
                [normalize_data(graph.nodes[n]["lat"], min_lat, max_lat),
                 normalize_data(graph.nodes[n]["lon"], min_lon, max_lon)]
                for n in path[:-1]  # Берём все, кроме последнего
            ])
            target = np.array([
                normalize_data(graph.nodes[path[-1]]["lat"], min_lat, max_lat),
                normalize_data(graph.nodes[path[-1]]["lon"], min_lon, max_lon)
            ])  # Цель – координаты последнего узла маршрута

            data.append((features, target))

    return data, (min_lat, max_lat, min_lon, max_lon)

# --- 3. Создание нейросети ---
class RouteNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=2):  # Теперь output_size=2
        super(RouteNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Теперь возвращает [lat, lon]

# --- 4. Обучение модели ---
def train_model(graph):
    print(f"Количество узлов в графе: {len(graph.nodes)}")
    print(f"Количество рёбер в графе: {len(graph.edges)}")

    dataset, norm_params = generate_training_data(graph)
    model = RouteNN(input_size=2, hidden_size=32, output_size=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 100

    for epoch in range(num_epochs):
        total_loss = 0

        for features, target in dataset:
            # Берём только первую координату маршрута
            start_point = features[0]

            start_tensor = torch.tensor(start_point, dtype=torch.float32).unsqueeze(0)  # (1, 2)
            target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)  # (1, 2)

            optimizer.zero_grad()
            output = model(start_tensor)  # (1, 2)

            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Эпоха [{epoch+1}/{num_epochs}] завершена. Средний Loss: {total_loss / len(dataset):.6f}")

    print("Обучение завершено!")
    return model, norm_params

# --- 5. Генерация маршрута через нейросеть ---
def find_nearest_node(graph, lat, lon):
    """Находит ближайший узел к заданным координатам"""
    return min(graph.nodes, key=lambda n: (graph.nodes[n]["lat"] - lat) ** 2 + (graph.nodes[n]["lon"] - lon) ** 2)


def generate_route(graph, model, start, end, norm_params):
    min_lat, max_lat, min_lon, max_lon = norm_params

    input_data = np.array([[
        normalize_data(graph.nodes[start]['lat'], min_lat, max_lat),
        normalize_data(graph.nodes[start]['lon'], min_lon, max_lon)
    ]])
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    predicted_coords = model(input_tensor).detach().numpy()[0]  # Получаем lat/lon

    # Де-нормализация предсказанных координат
    pred_lat = predicted_coords[0] * (max_lat - min_lat) + min_lat
    pred_lon = predicted_coords[1] * (max_lon - min_lon) + min_lon

    predicted_node = find_nearest_node(graph, pred_lat, pred_lon)  # Ищем ближайший узел в графе

    try:
        route = nx.shortest_path(graph, start, predicted_node, weight="length") + \
                nx.shortest_path(graph, predicted_node, end, weight="length")[1:]
    except nx.NetworkXNoPath:
        route = nx.shortest_path(graph, start, end, weight="length")

    return route, norm_params

# --- 6. Визуализация маршрута ---
def plot_route(graph, route):
    m = folium.Map(location=[graph.nodes[route[0]]['lat'], graph.nodes[route[0]]['lon']], zoom_start=15)
    coords = [(graph.nodes[n]['lat'], graph.nodes[n]['lon']) for n in route]
    folium.PolyLine(coords, color="blue", weight=5).add_to(m)
    m.save("generated_route.html")

# --- Основной запуск ---
# Указываем координаты центра поиска
center_lat, center_lon = 55.758694, 37.658033  # Москва

graph = get_graph(center_lat, center_lon)

if graph.nodes:
    model, norm_params = train_model(graph)
    start, end = list(graph.nodes)[0], list(graph.nodes)[-1]
    route, norm_params = generate_route(graph, model, start, end, norm_params)
    plot_route(graph, route)
    print("Маршрут создан! Открывайте generated_route.html")
else:
    print("Ошибка: Граф пустой, маршруты не могут быть созданы.")
