import requests
OLLAMA_URL='http://localhost:11434/api/generate'; LG_MODEL='llama-guard3'
def classify_safety(text:str)->dict:
    r=requests.post(OLLAMA_URL, json={'model':LG_MODEL,'prompt':text,'stream':False}, timeout=30); r.raise_for_status()
    out=r.json().get('response',''); verdict='unsafe' if 'unsafe' in out.lower() else 'safe'; return {'verdict':verdict,'raw':out}
def enforce_input_guard(text:str):
    if classify_safety(text)['verdict']!='safe': raise ValueError('Blocked by safety policy (input)')
def enforce_output_guard(text:str)->str:
    return text if classify_safety(text)['verdict']=='safe' else 'I can’t provide that content.'