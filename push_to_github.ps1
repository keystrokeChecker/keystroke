$log = "C:\Users\Tursi\Desktop\keyboard\keystroke\push_log.txt"
Set-Location "C:\Users\Tursi\Desktop\keyboard\keystroke"

"=== Adding remote ===" | Tee-Object -FilePath $log
git remote add origin https://github.com/keystrokeChecker/keystroke.git 2>&1 | Tee-Object -FilePath $log -Append

"=== Creating and switching to feature/ml branch ===" | Tee-Object -FilePath $log -Append
git checkout -b feature/ml 2>&1 | Tee-Object -FilePath $log -Append

"=== Staging all changes ===" | Tee-Object -FilePath $log -Append
git add -A 2>&1 | Tee-Object -FilePath $log -Append

"=== Committing ===" | Tee-Object -FilePath $log -Append
git commit -m "feat: complete ML pipeline, APK build, backend integration and dataset tooling" 2>&1 | Tee-Object -FilePath $log -Append

"=== Pushing to feature/ml ===" | Tee-Object -FilePath $log -Append
git push -u origin feature/ml 2>&1 | Tee-Object -FilePath $log -Append

"=== DONE ===" | Tee-Object -FilePath $log -Append
