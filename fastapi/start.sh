uvicorn server:app --host "0.0.0.0" --reload  &
./ngrok http 8000 &
docker logs -f e2e
