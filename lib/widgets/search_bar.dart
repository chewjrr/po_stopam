import 'package:flutter/material.dart';

class SearchBarWidget extends StatelessWidget {
  final ValueChanged<String>? onSearch;
  final List<String> filterOptions;
  final ValueChanged<String>? onFilterSelected;

  const SearchBarWidget({
    super.key,
    this.onSearch,
    required this.filterOptions,
    this.onFilterSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
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
                    onChanged: onSearch,
                    decoration: const InputDecoration(
                      hintText: 'Поиск мест',
                      border: InputBorder.none,
                    ),
                  ),
                ),
              ],
            ),
            if (onFilterSelected != null)
              SizedBox(
                height: 40,
                child: ListView(
                  scrollDirection: Axis.horizontal,
                  children: filterOptions.map((category) {
                    return Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 4),
                      child: ActionChip(
                        label: Text(category),
                        onPressed: () => onFilterSelected!(category),
                      ),
                    );
                  }).toList(),
                ),
              ),
          ],
        ),
      ),
    );
  }
}