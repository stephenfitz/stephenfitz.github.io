#!/bin/bash
# python3 -m http.server &
live-server --no-browser &
sleep 1
url="http://localhost:8000"
if command -v xdg-open >/dev/null; then
  xdg-open "$url"
elif command -v open >/dev/null; then
  # open "$url"
  open -na "Google Chrome" --args --new-window http://localhost:8080
else
  echo "Please open $url manually"
fi
