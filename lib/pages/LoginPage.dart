import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:startap_maps/components/my_button.dart';
import 'package:startap_maps/components/my_text_field.dart';
import 'package:startap_maps/services/auth/auth_service.dart';

import '../services/auth/auth_service.dart';

class LoginPage extends StatefulWidget {
  final void Function()? onTap;
  const LoginPage({super.key, required this.onTap,});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final emailController = TextEditingController();
  final passwordController = TextEditingController();

  void singIn() async{
    final authService = Provider.of<AuthService>(context, listen: false);

    try {
      await authService.singInWithEmailandPassword(emailController.text, passwordController.text,);
    } catch (e){
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.toString(),),),);
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
                  "Welcome back, you have been missed!",
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

                SizedBox(height: 25),

                MyButton(onTap: singIn, text: "Sing In"),

                SizedBox(height: 50),

                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                  Text("Not a member?"),
                  SizedBox(width: 4),
                  GestureDetector(
                    onTap: widget.onTap,
                    child: Text(
                      "Register now",
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
