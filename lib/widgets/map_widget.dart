import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

class MapWidget extends StatelessWidget {
  final LatLng currentPosition;
  final double zoomLevel;
  final List<Marker> markers;
  final List<LatLng> routePoints;
  final Function(LatLng) onTap;
  final MapController mapController;

  const MapWidget({
    super.key,
    required this.currentPosition,
    required this.zoomLevel,
    required this.markers,
    required this.routePoints,
    required this.onTap,
    required this.mapController,
  });

  @override
  Widget build(BuildContext context) {
    return FlutterMap(
      mapController: mapController,
      options: MapOptions(
        center: currentPosition,
        zoom: zoomLevel,
        onTap: (_, point) => onTap(point),
      ),
      children: [
        TileLayer(
          urlTemplate: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
          subdomains: const ['a', 'b', 'c'],
        ),
        MarkerLayer(markers: markers),
        if (routePoints.isNotEmpty)
          PolylineLayer(
            polylines: [
              Polyline(
                points: routePoints,
                color: Colors.blue.withOpacity(0.7),
                strokeWidth: 5,
              ),
            ],
          ),
      ],
    );
  }
}