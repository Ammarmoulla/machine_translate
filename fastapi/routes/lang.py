import os
import psutil
from www import app
import handlers.lang as lang_handler
from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

@app.post('/lang')
async def get_lang(english_text: TextRequest):
    
    en_text = english_text.text
    return lang_handler.en_to_fr(en_text)


@app.get("/stop")
def stop_server():
    parent_pid = os.getpid()
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()

