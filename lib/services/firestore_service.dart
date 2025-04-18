import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:startap_maps/models/route_model.dart';

class FirestoreService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  Future<void> saveRoute(RouteModel route, String userId) async {
    try {
      await _firestore
          .collection('users')
          .doc(userId)
          .collection('saved_routes')
          .add(route.toMap());
    } catch (e) {
      throw 'Ошибка сохранения маршрута: $e';
    }
  }

  Stream<List<RouteModel>> getSavedRoutes(String userId) {
    return _firestore
        .collection('users')
        .doc(userId)
        .collection('saved_routes')
        .orderBy('date', descending: true)
        .snapshots()
        .map((snapshot) => snapshot.docs
        .map((doc) => RouteModel.fromMap(doc.id, doc.data()))
        .toList());
  }
}