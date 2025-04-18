import 'package:latlong2/latlong.dart';
import 'package:startap_maps/services/map_service.dart';
import 'package:startap_maps/services/firestore_service.dart';

class RouteService {
  final MapService _mapService;
  final FirestoreService _firestoreService;

  RouteService(this._mapService, this._firestoreService);
  FirestoreService get firestoreService => _firestoreService;

  Future<LatLng?> findNearestPoint(LatLng point, String category) async {
    // Здесь должна быть реализация поиска ближайшей точки через API
    // Временная заглушка с тестовыми данными
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
    double minDistance = _mapService.distanceCalculator(point, nearest);

    for (final p in points) {
      final distance = _mapService.distanceCalculator(point, p);
      if (distance < minDistance) {
        minDistance = distance;
        nearest = p;
      }
    }

    return nearest;
  }

  Future<List<LatLng>> buildRoute(LatLng start, LatLng end) async {
    return _mapService.generateRoutePoints(start, end);
  }
}