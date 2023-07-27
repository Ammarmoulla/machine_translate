import os
from www import app
import routes
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import io
import requests


telegram_token = "6645018983:AAG2nTpOuCxwdgfMZTlxkmlBxPchFrm8fec"
chat_id = "903737895"
ngrok_token = "2ST8bvXhOYVVsUKv1qe1jOnqgNW_5N8Pt75i32ijBJh3NDcNJ"
port = 8000

def send_telegram(text):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    response = requests.post(url, data=data)

if __name__ == '__main__':

    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(port).public_url
    print("Public URL:", public_url)
    send_telegram("Server Production is Ready 🎉🎉🎉 "
                  +"\n The <b>Public</b> URL below 🔥🚀🔥🚀🔥🚀"
                  + f"\n{public_url}/docs\n.")
    
    nest_asyncio.apply()
    uvicorn.run(app, port=port)
    

