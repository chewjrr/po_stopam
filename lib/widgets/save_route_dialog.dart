import 'package:flutter/material.dart';

class SaveRouteDialog extends StatelessWidget {
  final TextEditingController nameController;
  final double distance;
  final int duration;
  final String category;
  final IconData categoryIcon;
  final VoidCallback onSave;
  final VoidCallback onCancel;

  const SaveRouteDialog({
    super.key,
    required this.nameController,
    required this.distance,
    required this.duration,
    required this.category,
    required this.categoryIcon,
    required this.onSave,
    required this.onCancel,
  });

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
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
                  Text('${distance.toStringAsFixed(1)} км'),
                ],
              ),
              Column(
                children: [
                  const Icon(Icons.access_time, size: 20),
                  Text('$duration мин'),
                ],
              ),
              Column(
                children: [
                  Icon(categoryIcon, size: 20),
                  Text(category),
                ],
              ),
            ],
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: onCancel,
          child: const Text('Отмена'),
        ),
        ElevatedButton(
          onPressed: onSave,
          child: const Text('Сохранить'),
        ),
      ],
    );
  }
}