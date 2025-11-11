from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
_analyzer=AnalyzerEngine(); _anonymizer=AnonymizerEngine()
def redact_pii(text:str)->str:
    ents=_analyzer.analyze(text=text, language='en'); return _anonymizer.anonymize(text=text, analyzer_results=ents).text