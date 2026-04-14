import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'utils/theme.dart';
import 'utils/env.dart';
import 'screens/auth/login_screen.dart';
import 'screens/dashboard/subject_dashboard_screen.dart';
import 'services/url_config_service.dart';
import 'services/auth_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize Supabase ONLY if the placeholders have been replaced
  if (!Env.supabaseUrl.contains('YOUR_SUPABASE')) {
    await Supabase.initialize(
      url: Env.supabaseUrl,
      anonKey: Env.supabaseAnonKey,
    );
  }

  // Restore persisted backend API URL
  UrlConfigService().loadSavedUrl();

  // AuthService constructor auto-restores session from localStorage
  // (called implicitly via singleton factory)
  AuthService();

  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
      systemNavigationBarColor: Colors.white,
      systemNavigationBarIconBrightness: Brightness.dark,
    ),
  );

  runApp(const ClassroomMonitorApp());
}

class ClassroomMonitorApp extends StatefulWidget {
  const ClassroomMonitorApp({super.key});

  @override
  State<ClassroomMonitorApp> createState() => _ClassroomMonitorAppState();
}

class _ClassroomMonitorAppState extends State<ClassroomMonitorApp> {
  ThemeMode _themeMode = ThemeMode.light;

  void toggleTheme() {
    setState(() {
      _themeMode =
          _themeMode == ThemeMode.light ? ThemeMode.dark : ThemeMode.light;
    });
  }

  @override
  Widget build(BuildContext context) {
    // If a session already exists in localStorage, jump straight to dashboard
    final isLoggedIn = AuthService().isLoggedIn;

    return MaterialApp(
      title: 'Classroom Monitor',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: _themeMode,
      home: isLoggedIn
          ? const SubjectDashboardScreen()
          : const LoginScreen(),
    );
  }
}
