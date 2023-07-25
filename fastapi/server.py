import os
from www import app
import routes
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import io
port = 8000

if __name__ == '__main__':

    ngrok.set_auth_token("2ST8bvXhOYVVsUKv1qe1jOnqgNW_5N8Pt75i32ijBJh3NDcNJ")
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)
