$python = "C:\Program Files\Python314\python.exe"
$log = "C:\Users\Tursi\Desktop\keyboard\keystroke\keystroke_backend\backend_log.txt"

Set-Content $log "BACKEND START: $(Get-Date)"

# Install dependencies
"--- Installing requirements ---" | Tee-Object -FilePath $log -Append
& $python -m pip install -r requirements.txt 2>&1 | Tee-Object -FilePath $log -Append

# Get local IP so user knows where to point the app
"--- Network Info ---" | Tee-Object -FilePath $log -Append
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.*" } | Select-Object -First 1).IPAddress
"Local IP: $ip" | Tee-Object -FilePath $log -Append
"App backend URL: http://${ip}:8000" | Tee-Object -FilePath $log -Append

"--- Starting FastAPI server on 0.0.0.0:8000 ---" | Tee-Object -FilePath $log -Append
& $python -m uvicorn app:app --host 0.0.0.0 --port 8000 2>&1 | Tee-Object -FilePath $log -Append
