import os
from www import app
from waitress import serve
import routes

port = os.getenv("PORT")

if __name__ == '__main__':
    print("Server is running on port: " + port)
    serve(app, host='0.0.0.0', port=port)
