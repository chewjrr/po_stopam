import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:provider/provider.dart';
import 'package:startap_maps/services/auth/auth_service.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final MapController _mapController = MapController();
  final Distance _distanceCalculator = const Distance();

  LatLng _currentPosition = const LatLng(55.7558, 37.6176);
  double _zoomLevel = 13.0;
  List<Marker> _markers = [];
  List<LatLng> _routePoints = [];
  final List<String> _filterOptions = ['Кафе', 'Парки', 'Музеи', 'ТЦ'];
  String? _selectedCategory;
  LatLng? _startPoint;
  LatLng? _endPoint;
  bool _isRouteBuilding = false;
  double _routeDistance = 0.0;
  int _routeDuration = 0;
  bool _isLoadingRoute = false;

  @override
  void initState() {
    super.initState();
    _determinePosition();
    _addSampleMarkers();
  }

  Future<void> _determinePosition() async {
    // Здесь должна быть реализация получения текущего местоположения
    setState(() => _currentPosition = const LatLng(55.7558, 37.6176));
  }

  void _addSampleMarkers() {
    setState(() {
      _markers = [
        Marker(
          point: _currentPosition,
          width: 40,
          height: 40,
          builder: (ctx) => const Icon(Icons.location_pin, color: Colors.red, size: 40),
        ),
      ];
    });
  }

  void _startRouteBuilding(String category) {
    setState(() {
      _selectedCategory = category;
      _isRouteBuilding = true;
      _startPoint = null;
      _endPoint = null;
      _routePoints.clear();
      _markers.removeWhere((m) => m.point != _currentPosition);
    });
  }

  void _addPoint(LatLng point) async {
    if (!_isRouteBuilding || _selectedCategory == null) return;

    setState(() {
      _isLoadingRoute = true;
      if (_startPoint == null) {
        _startPoint = point;
        _markers.add(_createMarker(point, Colors.green, Icons.location_pin));
      } else {
        _startPoint = point;
        _markers.removeWhere((m) => m.key == const ValueKey('start'));
        _markers.add(_createMarker(point, Colors.green, Icons.location_pin));
      }
    });

    try {
      final nearest = await _findNearestPoint(point, _selectedCategory!);
      if (nearest != null) {
        await _buildRoute(point, nearest);
      }
    } finally {
      setState(() => _isLoadingRoute = false);
    }
  }

  Marker _createMarker(LatLng point, Color color, IconData icon) {
    return Marker(
      point: point,
      width: 40,
      height: 40,
      key: icon == Icons.location_pin ? const ValueKey('start') : const ValueKey('end'),
      builder: (ctx) => Icon(icon, color: color, size: 40),
    );
  }

  Future<LatLng?> _findNearestPoint(LatLng startPoint, String category) async {
    await Future.delayed(const Duration(milliseconds: 300));

    final places = {
      'Кафе': [
        const LatLng(55.7604, 37.6186),
        const LatLng(55.7517, 37.6178),
      ],
      'Парки': [
        const LatLng(55.7338, 37.5889),
        const LatLng(55.7905, 37.5836),
      ],
      'Музеи': [
        const LatLng(55.7480, 37.6085),
        const LatLng(55.7157, 37.5542),
      ],
      'ТЦ': [
        const LatLng(55.7507, 37.6172),
        const LatLng(55.7580, 37.6225),
      ],
    };

    if (!places.containsKey(category)) return null;

    final points = places[category]!;
    LatLng nearest = points.first;
    double minDistance = _distanceCalculator(startPoint, nearest);

    for (final point in points) {
      final distance = _distanceCalculator(startPoint, point);
      if (distance < minDistance) {
        minDistance = distance;
        nearest = point;
      }
    }

    return nearest;
  }

  Future<void> _buildRoute(LatLng start, LatLng end) async {
    setState(() => _isLoadingRoute = true);

    try {
      _routePoints = _generateRoutePoints(start, end);
      _routeDistance = _distanceCalculator(start, end) / 1000;
      _routeDuration = (_routeDistance * 15).toInt();

      setState(() {
        _endPoint = end;
        _markers.removeWhere((m) => m.key == const ValueKey('end'));
        _markers.add(_createMarker(end, Colors.blue, Icons.flag));
      });
    } finally {
      setState(() => _isLoadingRoute = false);
    }
  }

  List<LatLng> _generateRoutePoints(LatLng start, LatLng end) {
    final List<LatLng> points = [];
    const int steps = 10;

    for (int i = 0; i <= steps; i++) {
      final ratio = i / steps;
      points.add(LatLng(
        start.latitude + (end.latitude - start.latitude) * ratio,
        start.longitude + (end.longitude - start.longitude) * ratio,
      ));
    }

    return points;
  }

  Future<void> _saveRoute() async {
    if (_startPoint == null || _endPoint == null || _routePoints.isEmpty) return;

    final nameController = TextEditingController(
      text: 'Маршрут в $_selectedCategory',
    );

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Сохранить маршрут'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: nameController,
              decoration: const InputDecoration(
                labelText: 'Название маршрута',
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                Column(
                  children: [
                    const Icon(Icons.directions_walk, size: 20),
                    Text('${_routeDistance.toStringAsFixed(1)} км'),
                  ],
                ),
                Column(
                  children: [
                    const Icon(Icons.access_time, size: 20),
                    Text('$_routeDuration мин'),
                  ],
                ),
                Column(
                  children: [
                    Icon(_getCategoryIcon(_selectedCategory!), size: 20),
                    Text(_selectedCategory!),
                  ],
                ),
              ],
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Отмена'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Сохранить'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      await _saveRouteToFirebase(
        name: nameController.text,
        start: _startPoint!,
        end: _endPoint!,
        points: _routePoints,
        distance: _routeDistance,
        duration: _routeDuration,
        category: _selectedCategory!,
      );
    }
  }

  Future<void> _saveRouteToFirebase({
    required String name,
    required LatLng start,
    required LatLng end,
    required List<LatLng> points,
    required double distance,
    required int duration,
    required String category,
  }) async {
    try {
      final user = _auth.currentUser;
      if (user == null) throw 'Пользователь не авторизован';

      await _firestore
          .collection('users')
          .doc(user.uid)
          .collection('saved_routes')
          .add({
        'name': name,
        'date': DateTime.now().toIso8601String(),
        'start': {'lat': start.latitude, 'lng': start.longitude},
        'end': {'lat': end.latitude, 'lng': end.longitude},
        'points': points.map((p) => {'lat': p.latitude, 'lng': p.longitude}).toList(),
        'distance': distance,
        'duration': duration,
        'category': category,
      });

      setState(() {
        _isRouteBuilding = false;
        _selectedCategory = null;
        _startPoint = null;
        _endPoint = null;
        _routePoints.clear();
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Маршрут сохранен!')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Ошибка: $e')),
      );
    }
  }

  IconData _getCategoryIcon(String category) {
    switch (category) {
      case 'Кафе': return Icons.local_cafe;
      case 'Парки': return Icons.park;
      case 'Музеи': return Icons.museum;
      case 'ТЦ': return Icons.shopping_cart;
      default: return Icons.place;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              center: _currentPosition,
              zoom: _zoomLevel,
              onTap: (_, point) => _addPoint(point),
            ),
            children: [
              TileLayer(
                urlTemplate: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                subdomains: const ['a', 'b', 'c'],
              ),
              MarkerLayer(markers: _markers),
              if (_routePoints.isNotEmpty)
                PolylineLayer(
                  polylines: [
                    Polyline(
                      points: _routePoints,
                      color: Colors.blue.withOpacity(0.7),
                      strokeWidth: 5,
                    ),
                  ],
                ),
            ],
          ),

          if (_isLoadingRoute)
            const Center(child: CircularProgressIndicator()),

          Positioned(
            top: 50,
            left: 16,
            right: 16,
            child: Card(
              elevation: 3,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.search),
                        const SizedBox(width: 8),
                        Expanded(
                          child: TextField(
                            decoration: const InputDecoration(
                              hintText: 'Поиск мест',
                              border: InputBorder.none,
                            ),
                          ),
                        ),
                      ],
                    ),
                    if (_selectedCategory == null)
                      SizedBox(
                        height: 40,
                        child: ListView(
                          scrollDirection: Axis.horizontal,
                          children: _filterOptions.map((category) {
                            return Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 4),
                              child: ActionChip(
                                label: Text(category),
                                onPressed: () => _startRouteBuilding(category),
                              ),
                            );
                          }).toList(),
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ),

          if (_isRouteBuilding && _startPoint != null)
            Positioned(
              bottom: 20,
              left: 0,
              right: 0,
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 16),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.1),
                      blurRadius: 10,
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.directions, color: Colors.blue),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Маршрут в $_selectedCategory',
                                style: const TextStyle(
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              Text(
                                '${_routeDistance.toStringAsFixed(1)} км • $_routeDuration мин',
                                style: const TextStyle(color: Colors.grey),
                              ),
                            ],
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.close),
                          onPressed: () {
                            setState(() {
                              _isRouteBuilding = false;
                              _selectedCategory = null;
                              _routePoints.clear();
                              _markers.removeWhere((m) => m.key != null);
                            });
                          },
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(
                          child: OutlinedButton(
                            onPressed: () {
                              setState(() {
                                _isRouteBuilding = false;
                                _selectedCategory = null;
                                _routePoints.clear();
                                _markers.removeWhere((m) => m.key != null);
                              });
                            },
                            child: const Text('Отмена'),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: ElevatedButton(
                            onPressed: _saveRoute,
                            child: const Text('Сохранить'),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            heroTag: 'location',
            onPressed: _determinePosition,
            child: const Icon(Icons.my_location),
          ),
          const SizedBox(height: 16),
          FloatingActionButton(
            heroTag: 'zoom_in',
            onPressed: () {
              setState(() => _zoomLevel += 1);
              _mapController.move(_currentPosition, _zoomLevel);
            },
            child: const Icon(Icons.add),
          ),
          const SizedBox(height: 16),
          FloatingActionButton(
            heroTag: 'zoom_out',
            onPressed: () {
              setState(() {
                if (_zoomLevel > 1) {
                  _zoomLevel -= 1;
                  _mapController.move(_currentPosition, _zoomLevel);
                }
              });
            },
            child: const Icon(Icons.remove),
          ),
        ],
      ),
    );
  }
}
