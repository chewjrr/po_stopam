import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class MyRoutesPage extends StatefulWidget {
  const MyRoutesPage({super.key});

  @override
  State<MyRoutesPage> createState() => _MyRoutesPageState();
}

class _MyRoutesPageState extends State<MyRoutesPage> {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Мои маршруты'),
      ),
      body: StreamBuilder<QuerySnapshot>(
        stream: _firestore
            .collection('users')
            .doc(_auth.currentUser?.uid)
            .collection('saved_routes')
            .snapshots(),
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return Center(child: Text('Ошибка: ${snapshot.error}'));
          }

          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (snapshot.data?.docs.isEmpty ?? true) {
            return const Center(child: Text('У вас нет сохраненных маршрутов'));
          }

          return ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: snapshot.data?.docs.length,
            itemBuilder: (context, index) {
              var route = snapshot.data?.docs[index];
              return _buildRouteCard(route?.data() as Map<String, dynamic>);
            },
          );
        },
      ),
    );
  }

  Widget _buildRouteCard(Map<String, dynamic> routeData) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              routeData['name'] ?? 'Без названия',
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Дата: ${routeData['date'] ?? 'Не указана'}',
              style: TextStyle(color: Colors.grey[600]),
            ),
            const SizedBox(height: 8),
            Text(
              'Дистанция: ${routeData['distance'] ?? '0'} км',
            ),
            const SizedBox(height: 8),
            Text(
              'Время: ${routeData['duration'] ?? '0'} мин',
            ),
          ],
        ),
      ),
    );
  }
}