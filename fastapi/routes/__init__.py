from www import app
from .lang import *

@app.get('/')
async def index():
    result = {
        "code":"200",
        "message":"I am Ready"
    }
    return result

