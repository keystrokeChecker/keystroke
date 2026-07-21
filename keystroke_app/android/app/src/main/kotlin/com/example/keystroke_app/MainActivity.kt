curl.exe http://127.0.0.1:8000/health
curl.exe -F "file=@data\session1.wav" -F "method=yamnet" -F "threshold=0.4" -F "delta=0.05" http://127.0.0.1:8000/analyze
package com.example.keystroke_app

import io.flutter.embedding.android.FlutterActivity

class MainActivity : FlutterActivity()
