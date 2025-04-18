import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:provider/provider.dart';
import 'package:startap_maps/models/route_model.dart';
import 'package:startap_maps/services/auth/auth_service.dart';
import 'package:startap_maps/services/map_service.dart';
import 'package:startap_maps/services/route_service.dart';
import 'package:startap_maps/widgets/map_widget.dart';
import 'package:startap_maps/widgets/route_bottom_sheet.dart';
import 'package:startap_maps/widgets/search_bar.dart';
import 'package:startap_maps/widgets/save_route_dialog.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late final MapService _mapService;
  late final RouteService _routeService;
  final List<String> _filterOptions = ['Кафе', 'Парки', 'Музеи', 'ТЦ'];

  LatLng _currentPosition = const LatLng(55.7558, 37.6176);
  double _zoomLevel = 13.0;
  List<Marker> _markers = [];
  List<LatLng> _routePoints = [];
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
    _mapService = Provider.of<MapService>(context, listen: false);
    _routeService = Provider.of<RouteService>(context, listen: false);
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

  Future<void> _addPoint(LatLng point) async {
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
      final nearest = await _routeService.findNearestPoint(point, _selectedCategory!);
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

  Future<void> _buildRoute(LatLng start, LatLng end) async {
    setState(() => _isLoadingRoute = true);

    try {
      _routePoints = await _routeService.buildRoute(start, end);
      _routeDistance = _mapService.distanceCalculator(start, end) / 1000;
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

  void _cancelRouteBuilding() {
    setState(() {
      _isRouteBuilding = false;
      _selectedCategory = null;
      _routePoints.clear();
      _markers.removeWhere((m) => m.key != null);
    });
  }

  Future<void> _saveRoute() async {
    if (_startPoint == null || _endPoint == null || _routePoints.isEmpty) return;

    final nameController = TextEditingController(
      text: 'Маршрут в $_selectedCategory',
    );

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => SaveRouteDialog(
        nameController: nameController,
        distance: _routeDistance,
        duration: _routeDuration,
        category: _selectedCategory!,
        categoryIcon: _getCategoryIcon(_selectedCategory!),
        onSave: () => Navigator.pop(context, true),
        onCancel: () => Navigator.pop(context, false),
      ),
    );

    if (confirmed == true) {
      final user = Provider.of<AuthService>(context, listen: false).currentUser;
      if (user == null) return;

      final route = RouteModel(
        name: nameController.text,
        start: _startPoint!,
        end: _endPoint!,
        points: _routePoints,
        distance: _routeDistance,
        duration: _routeDuration,
        category: _selectedCategory!,
        date: DateTime.now(),
      );

      try {
        await _routeService.firestoreService.saveRoute(route, user.uid);
        _cancelRouteBuilding();
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Маршрут сохранен!')),
        );
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка: $e')),
        );
      }
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

  void _zoomIn() {
    setState(() => _zoomLevel += 1);
    _mapService.controller.move(_currentPosition, _zoomLevel);
  }

  void _zoomOut() {
    setState(() {
      if (_zoomLevel > 1) {
        _zoomLevel -= 1;
        _mapService.controller.move(_currentPosition, _zoomLevel);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          MapWidget(
            currentPosition: _currentPosition,
            zoomLevel: _zoomLevel,
            markers: _markers,
            routePoints: _routePoints,
            onTap: _addPoint,
            mapController: _mapService.controller,
          ),
          Positioned(
            top: 50,
            left: 16,
            right: 16,
            child: SearchBarWidget(
              filterOptions: _filterOptions,
              onFilterSelected: _selectedCategory == null
                  ? _startRouteBuilding
                  : null,
            ),
          ),
          if (_isLoadingRoute)
            const Center(child: CircularProgressIndicator()),
          if (_isRouteBuilding && _startPoint != null)
            Positioned(
              bottom: 20,
              left: 0,
              right: 0,
              child: RouteBottomSheet(
                category: _selectedCategory!,
                distance: _routeDistance,
                duration: _routeDuration,
                onCancel: _cancelRouteBuilding,
                onSave: _saveRoute,
                categoryIcon: _getCategoryIcon(_selectedCategory!),
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
            onPressed: _zoomIn,
            child: const Icon(Icons.add),
          ),
          const SizedBox(height: 16),
          FloatingActionButton(
            heroTag: 'zoom_out',
            onPressed: _zoomOut,
            child: const Icon(Icons.remove),
          ),
        ],
      ),
    );
  }
}