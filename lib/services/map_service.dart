import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

class MapService {
  final MapController _controller = MapController();
  final Distance _distanceCalculator = const Distance();

  MapController get controller => _controller;
  Distance get distanceCalculator => _distanceCalculator;

  LatLng generateIntermediatePoint(LatLng start, LatLng end, double ratio) {
    return LatLng(
      start.latitude + (end.latitude - start.latitude) * ratio,
      start.longitude + (end.longitude - start.longitude) * ratio,
    );
  }

  List<LatLng> generateRoutePoints(LatLng start, LatLng end, {int steps = 10}) {
    final List<LatLng> points = [];
    for (int i = 0; i <= steps; i++) {
      points.add(generateIntermediatePoint(start, end, i / steps));
    }
    return points;
  }
}