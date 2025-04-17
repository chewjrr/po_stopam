import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import '../chat/chat_service.dart';

class ChatPage extends StatefulWidget {
  final String receiverUserEmail;
  final String receiverUserID;
  const ChatPage({super.key, required this.receiverUserEmail, required this.receiverUserID,});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final ChatService _chatService = ChatService();
  final FirebaseAuth _firebaseAuth = FirebaseAuth.instance;
  final TextEditingController _textController = TextEditingController();

  void _sendMessage() async {
    if (_textController.text.isNotEmpty) {
      await _chatService.sendMessage(_textController.text);
      _textController.clear();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Чат тех-поддержки'),
      ),
      body: Column(
        children: [
          Expanded(child: _buildMessageList()),
          _buildMessageInput(),
        ],
      ),
    );
  }

  Widget _buildMessageList() {
    return StreamBuilder(
      stream: _chatService.getMessages(_firebaseAuth.currentUser!.uid),
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return Text('Error ${snapshot.error.toString()}');
        }
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Text("Loading..");
        }
        return ListView(
          children: snapshot.data!.docs.map((document) => _buildMessageItem(document)).toList(),
        );
      },
    );
  }

  Widget _buildMessageItem(DocumentSnapshot document) {
    Map<String, dynamic> data = document.data() as Map<String, dynamic>;
    var alignment = (data['senderId'] == _firebaseAuth.currentUser!.uid
        ? Alignment.centerRight
        : Alignment.centerLeft);

    var messageStyle = TextStyle(
      fontSize: 20,
      color: (data['senderId'] == _firebaseAuth.currentUser!.uid
          ? Colors.black
          : Colors.white),
    );

    var timeStyle = TextStyle(
      fontSize: 15,
      color: (data['senderId'] == _firebaseAuth.currentUser!.uid
          ? Colors.black
          : Colors.white),
    );

    var containerDecoration = BoxDecoration(
      color: (data['senderId'] == _firebaseAuth.currentUser!.uid
          ? Colors.grey[120]
          : const Color(0xFF1A6FEE)),
      borderRadius: BorderRadius.circular(10),
      border: Border.all(
        color: (data['senderId'] == _firebaseAuth.currentUser!.uid
            ? const Color(0xFF1A6FEE)
            : Colors.transparent),
        width: 2.0,
      ),
    );

    DateTime timestamp = (data['timestamp'] as Timestamp).toDate();
    String formattedTime = "${timestamp.hour}:${timestamp.minute}";

    return Container(
      alignment: alignment,
      child: Padding(
        padding: const EdgeInsets.only(left: 10, right: 10, top: 5, bottom: 5),
        child: IntrinsicWidth(
          child: Container(
            decoration: containerDecoration,
            padding: const EdgeInsets.all(10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Align(
                  alignment: alignment,
                  child: Text(
                    formattedTime,
                    style: timeStyle,
                  ),
                ),
                Align(
                  alignment: alignment,
                  child: Text(
                    data['message'],
                    style: messageStyle,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMessageInput() {
    return Row(
      children: [
        Expanded(
          child: Padding(
            padding: const EdgeInsets.only(left: 8.0, bottom: 8.0),
            child: TextField(
              controller: _textController,
              decoration: const InputDecoration(
                hintText: 'Введите сообщение...',
                border: OutlineInputBorder(
                  borderSide: BorderSide(color: Color(0xFF1A6FEE), width: 2.0),
                ),
                focusedBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Color(0xFF1A6FEE), width: 2.0),
                ),
                enabledBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Color(0xFF1A6FEE), width: 2.0),
                ),
                hintStyle: TextStyle(color: Colors.grey),
              ),
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: IconButton(
            onPressed: _sendMessage,
            icon: const Icon(
              Icons.arrow_forward_ios_outlined,
              size: 30,
              color: Color(0xFF1A6FEE),
            ),
          ),
        ),
      ],
    );
  }
}
