import overpy
import folium
import os
import webbrowser
import networkx as nx

# Инициализация Overpass API
api = overpy.Overpass()

# Определите начальную и конечную точки
start_latlng = (55.758694, 37.658033)
mid_latlng = (55.76008799171476, 37.6512448101309)
end_latlng = (55.766218, 37.635638)

# Запрос к Overpass API для получения дорог в области
query = f"""
    [out:json];
    way
      ["highway"~"footway|path|pedestrian|steps"]
      (around:1000, {start_latlng[0]}, {start_latlng[1]}, {end_latlng[0]}, {end_latlng[1]});
    out body;
    >;
    out skel qt;
"""
result = api.query(query)

# Создание графа
graph = nx.Graph()

# Добавление узлов и рёбер в граф
for way in result.ways:
    nodes = way.nodes
    for i in range(len(nodes) - 1):
        node1 = nodes[i]
        node2 = nodes[i + 1]
        # Добавляем узлы и рёбра в граф
        graph.add_node(node1.id, lat=float(node1.lat), lon=float(node1.lon))
        graph.add_node(node2.id, lat=float(node2.lat), lon=float(node2.lon))
        # Расстояние между узлами (можно использовать Haversine для точности)
        graph.add_edge(node1.id, node2.id, weight=1)  # Вес можно настроить

# Нахождение ближайших узлов к начальной и конечной точкам
def find_nearest_node(graph, lat, lon):
    nearest_node = None
    min_distance = float('inf')
    for node in graph.nodes:
        node_lat = graph.nodes[node]['lat']
        node_lon = graph.nodes[node]['lon']
        distance = (node_lat - lat) ** 2 + (node_lon - lon) ** 2
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    return nearest_node

start_node = find_nearest_node(graph, start_latlng[0], start_latlng[1])
mid_node = find_nearest_node(graph, mid_latlng[0], mid_latlng[1])
end_node = find_nearest_node(graph, end_latlng[0], end_latlng[1])

# Поиск кратчайшего пути
shortest_path_toMid = nx.shortest_path(graph, start_node, mid_node, weight='weight')
shortest_path_toEnd = nx.shortest_path(graph, mid_node, end_node, weight='weight')

full_route = shortest_path_toMid + shortest_path_toEnd[1:]

# Извлечение координат маршрута
route_coords = []
for node_id in full_route:
    node = graph.nodes[node_id]
    route_coords.append((node['lat'], node['lon']))

# Создание карты с помощью Folium
m = folium.Map(location=start_latlng, zoom_start=18)

# Добавление маршрута на карту
folium.PolyLine(route_coords, color="purple", weight=5, opacity=0.7).add_to(m)

folium.Marker(start_latlng, popup="Начальная точка", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(mid_latlng, popup="Промежуточная точка", icon=folium.Icon(color="blue")).add_to(m)
folium.Marker(end_latlng, popup="Конечная точка", icon=folium.Icon(color="red")).add_to(m)

# Сохранение карты в HTML-файл
file_path = "shortest_route.html"
m.save(file_path)

# Автоматическое открытие файла в браузере
webbrowser.open('file://' + os.path.abspath(file_path))
#nvnvnv