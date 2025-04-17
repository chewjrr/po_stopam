import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:startap_maps/services/auth/auth_service.dart';

import '../components/my_button.dart';
import '../components/my_text_field.dart';

class RegisterPage extends StatefulWidget {
  final void Function()? onTap;
  const RegisterPage({super.key, required this.onTap});

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final emailController = TextEditingController();
  final passwordController = TextEditingController();
  final confirmPasswordController = TextEditingController();

  Future<void> singUp() async {
    if (passwordController.text != confirmPasswordController.text) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Password do not match!"),),);
      return;
    }
    final authService = Provider.of<AuthService>(context, listen:  false);

    try {
      await authService.singInWithEmailandPassword(emailController.text, passwordController.text,);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.toString()),),);
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: Colors.grey[300],
        body: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 25.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [

                  SizedBox(height: 50),

                  Icon(
                    Icons.map,
                    size:  100,
                    color: Colors.grey[800],
                  ),

                  SizedBox(height: 50),

                  const Text(
                    "Let`s create an account for you!",
                    style: TextStyle(
                      fontSize: 16,
                    ),
                  ),

                  SizedBox(height: 25),

                  MyTextField(
                    hintText: "Email",
                    controller: emailController,
                    obscureText: false,
                  ),

                  SizedBox(height: 10),

                  MyTextField(
                    hintText: "Password",
                    controller: passwordController,
                    obscureText: true,
                  ),

                  SizedBox(height: 10),

                  MyTextField(
                    hintText: "Confirm password",
                    controller: confirmPasswordController,
                    obscureText: true,
                  ),

                  SizedBox(height: 25),

                  MyButton(onTap: singUp, text: "Sing Un"),

                  SizedBox(height: 50),

                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text("Already a member?"),
                      SizedBox(width: 4),
                      GestureDetector(
                        onTap: widget.onTap,
                        child: Text(
                          "Login now",
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],),
                ],
              ),
            ),
          ),
        )
    );
  }
}

