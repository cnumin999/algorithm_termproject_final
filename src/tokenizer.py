# src/tokenizer.py
import re
from konlpy.tag import Komoran

tagger = Komoran()

def extract_terms(text):
    if not text:
        return []
    
    result = []
    
    # 1. 괄호 안 영문 추출
    english_in_brackets = re.findall(r'\(([a-zA-Z]+)\)', text)
    for eng in english_in_brackets:
        eng_lower = eng.lower()
        if eng_lower not in result:
            result.append(eng_lower)
    
    # 2. 괄호 제거 후 Komoran 형태소 분석
    text_clean = re.sub(r'\([^)]*\)', '', text)  # 괄호와 내용 제거
    tokens = tagger.pos(text_clean)
    
    for w, t in tokens:
        # 명사(NNG, NNP), 외국어(SL), 동사(VV), 형용사(VA), 
        # 부사(MAG), 어근(XR), 미지정어(NA), 명사추정(NNB) 포함
        if t in {'NNG', 'NNP', 'SL', 'VV', 'VA', 'MAG', 'XR', 'NA', 'NNB'}:
            term = w.lower() if t == 'SL' else w
            
            # 1글자 제외 (조사 잔여물 방지)
            if len(term) >= 2 and term not in result:
                result.append(term)
    
    return result