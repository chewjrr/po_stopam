import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/widgets.dart';
import '../models/Message.dart';

class ChatService extends ChangeNotifier {
  final FirebaseAuth _firebaseAuth = FirebaseAuth.instance;
  final FirebaseFirestore _fireStore = FirebaseFirestore.instance;

  Future<void> sendMessage(String message) async {
    final String currentUserId = _firebaseAuth.currentUser!.uid;
    final Timestamp timestamp = Timestamp.now();

    Message newMessage = Message(
      message: message,
      receiverId: "admin",
      senderId: currentUserId,
      timestamp: timestamp,
    );

    List<String> ids = [currentUserId, "admin"];
    ids.sort();
    String chatRoomId = ids.join("_");

    await _fireStore.collection('chat_rooms').doc(chatRoomId).collection('messages').add(newMessage.toMap());
  }

  Stream<QuerySnapshot> getMessages(String userId) {
    List<String> ids = [userId, "admin"];
    ids.sort();
    String chatRoomId = ids.join("_");

    return _fireStore.collection("chat_rooms")
        .doc(chatRoomId).collection('messages')
        .orderBy("timestamp", descending: false).snapshots();
  }
}
