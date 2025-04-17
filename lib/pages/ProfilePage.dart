import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:startap_maps/pages/ChatPage.dart';
import 'package:startap_maps/pages/MyRoutesPage.dart';
import '../services/auth/auth_service.dart';

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  @override
  Widget build(BuildContext context) {
    final authService = Provider.of<AuthService>(context);
    final User? user = FirebaseAuth.instance.currentUser;
    final String email = user?.email ?? 'Email недоступен';

    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 32),
              const Text(
                "Профиль",
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue,
                ),
              ),
              const SizedBox(height: 24),
              _buildUserInfo(email),
              const SizedBox(height: 40),
              _buildMenuSection(context, user, authService),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildUserInfo(String email) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.2),
            spreadRadius: 2,
            blurRadius: 5,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Row(
        children: [
          const Icon(Icons.email, size: 24, color: Colors.blue),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Email',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  email,
                  style: const TextStyle(
                    fontSize: 16,
                    color: Colors.black,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }


  Widget _buildMenuSection(BuildContext context, User? user, AuthService authService) {
    return Column(
      children: [
        _buildMenuButton(
          icon: Icons.directions,
          text: "Мои маршруты",
          onTap: () => Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => const MyRoutesPage()),
          ),
        ),
        _buildMenuButton(
          icon: Icons.favorite,
          text: "Любимые места",
          onTap: () {},
        ),
        _buildMenuButton(
          icon: Icons.settings,
          text: "Настройки",
          onTap: () {},
        ),
        const SizedBox(height: 40),
        _buildFooterButton(
          "Ответы на вопросы",
          onTap: () {
            if (user != null) {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => ChatPage(
                    receiverUserID: user.uid,
                    receiverUserEmail: user.email ?? '',
                  ),
                ),
              );
            }
          },
        ),
        _buildFooterButton("Политика конфиденциальности"),
        _buildFooterButton("Пользовательское соглашение"),
        _buildFooterButton(
          "Выход",
          color: const Color(0xFFFD3535),
          onTap: () => authService.signOut(),
        ),
        const SizedBox(height: 24),
      ],
    );
  }

  Widget _buildMenuButton({
    required IconData icon,
    required String text,
    required VoidCallback onTap,
  }) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        leading: Icon(icon, size: 28, color: Colors.blue),
        title: Text(
          text,
          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
        ),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }

  Widget _buildFooterButton(
      String text, {
        Color color = Colors.black,
        VoidCallback? onTap,
      }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: InkWell(
        onTap: onTap,
        child: Text(
          text,
          style: TextStyle(
            fontSize: 16,
            color: color,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
    );
  }
}
