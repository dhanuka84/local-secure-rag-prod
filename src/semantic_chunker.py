from typing import List, Dict
import re
import tiktoken
class SemanticChunker:
    def __init__(self, size_tokens=300, overlap=50, enc='cl100k_base'):
        self.size=size_tokens; self.overlap=overlap; self.enc=tiktoken.get_encoding(enc)
    def chunk(self, text:str, meta:Dict)->List[Dict]:
        parts=re.split(r'\n(?=(SECTION\s+\d+|Article\s+\d+|###\s+|##\s+|#\s+))', text, flags=re.I)
        sections=[''.join(parts[i:i+2]).strip() for i in range(0,len(parts),2)] or [text]
        chunks=[]
        for s in sections:
            toks=self.enc.encode(s)
            if len(toks)<=self.size: chunks.append(self._mk(s,meta)); continue
            for i in range(0,len(toks), self.size-self.overlap):
                part=self.enc.decode(toks[i:i+self.size]); chunks.append(self._mk(part,meta))
        return chunks
    def _mk(self,s,meta): return {'text':s,'metadata':{**meta,'tok_len':len(self.enc.encode(s))}}