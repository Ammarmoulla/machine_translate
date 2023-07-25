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
    public_url = ngrok.connect(8000).public_url
    print("Public URL:", public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)
    
    allowed_requests = 2
    request_count = 0
    while True:
        request_count += 1

        if request_count > allowed_requests:
            ngrok.disconnect(public_url)
            ngrok.kill()
            break
