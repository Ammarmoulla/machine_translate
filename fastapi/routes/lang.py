from www import app
import handlers.lang as lang_handler
from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

@app.post('/lang')
async def get_lang(english_text: TextRequest):
    
    en_text = english_text.text
    return lang_handler.en_to_fr(en_text)

