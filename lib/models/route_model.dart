import 'package:latlong2/latlong.dart';

class RouteModel {
  final String id;
  final String name;
  final LatLng start;
  final LatLng end;
  final List<LatLng> points;
  final double distance;
  final int duration;
  final String category;
  final DateTime date;

  RouteModel({
    this.id = '',
    required this.name,
    required this.start,
    required this.end,
    required this.points,
    required this.distance,
    required this.duration,
    required this.category,
    required this.date,
  });

  Map<String, dynamic> toMap() {
    return {
      'name': name,
      'start': {'lat': start.latitude, 'lng': start.longitude},
      'end': {'lat': end.latitude, 'lng': end.longitude},
      'points': points.map((p) => {'lat': p.latitude, 'lng': p.longitude}).toList(),
      'distance': distance,
      'duration': duration,
      'category': category,
      'date': date.toIso8601String(),
    };
  }

  factory RouteModel.fromMap(String id, Map<String, dynamic> map) {
    return RouteModel(
      id: id,
      name: map['name'],
      start: LatLng(map['start']['lat'], map['start']['lng']),
      end: LatLng(map['end']['lat'], map['end']['lng']),
      points: List<LatLng>.from(map['points'].map((p) => LatLng(p['lat'], p['lng']))),
      distance: map['distance'],
      duration: map['duration'],
      category: map['category'],
      date: DateTime.parse(map['date']),
    );
  }
}