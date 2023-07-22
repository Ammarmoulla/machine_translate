from www import app
from .lang import *

@app.get('/')
async def index():
    result = {
        "code":"200",
        "message":"Hello World"
    }
    return result

