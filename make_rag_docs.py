import os
import json
import pandas as pd
from utils import check_raid, extract_phase_in_title


def dnf_datas(folder_path):
    data = data = {file.replace('.json', '') : pd.read_json(folder_path + file) for file in os.listdir(folder_path) if file.split('.')[-1] == 'json'}
    return data

folder_path = 'data/'
datasets = dnf_datas(folder_path)

from langchain_core.documents import Document
import re
from soynlp.normalizer import repeat_normalize

def limit_repetitions(text):
    # 모든 문자에 대해 연속되는 경우 3회 이상 반복되는 것을 2회로 제한
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def remove_urls(text):
    # URL 패턴을 찾기 위한 정규 표현식
    url_pattern = r'https?:\/\/(?:www\.)?[\w/\-?=%.]+\.[\w/\-?=%.]+'
    # 텍스트에서 URL 제거
    no_url_text = re.sub(url_pattern, '', text)
    return no_url_text

def cleaning_text(text):
    
    result = remove_urls(text)
    
    result = re.sub("&amp;nbsp;", " ", result)
    result = re.sub('\xa0', '', result)
    result = re.sub('\n', ' ', result)
    # result = re.sub('[ㄱ-ㅎ가-힣]')
    
    result = limit_repetitions(result)
    
    return result

def make_metadata(data, key):
    metadata = dict()
    
    
    if 'homepage' in key:
        metadata['title'] = data['title']
        
        raid = ['[레이드] 아스라한 : 무의 장막', '[레이드] 기계 혁명 : 바칼 레이드 (하드 모드)',
        '[레이드] 기계 혁명 : 바칼 레이드', '[레이드] 기계 혁명 : 개전 (1~4인)',]
        region = ['[레기온] 어둑섬', '[레기온] 대마법사의 차원회랑 ', '[레기온] 빼앗긴 땅, 이스핀즈',]
        dungeon = ['[특수 던전] 이면 경계', '[휘장/젬]코드네임 게이볼그 ', '상급 던전',]
        
        equiment = ['보조 특성 (장비 특성)', '커스텀', '융합 장비',
        '장비 관련 기타 유용한 정보', '장비 공통', '장비 세팅', '장비 세팅(버퍼)',]
        
        
        if data['category1'] in raid or data['category1'] in region or data['category1'] in dungeon:
            metadata['category1'] = '공략'
            if data['category1'] in raid:
                metadata['category2'] = '레이드'
            elif data['category1'] in region:
                metadata['category2'] = '레기온'
            elif data['category1'] in dungeon:
                metadata['category2'] = '기타 던전'
            
            metadata['category3'] = data['category1']
        
        elif data['category1'] in equiment:
            metadata['category1'] = '장비'
            metadata['category2'] = ''
            metadata['category3'] = data['category1']
        
        else:
            metadata['category1'] = '직업'
            metadata['category2'] = ''
            metadata['category3'] = data['category1']
            
    elif 'wiki' in key:
        for key, value in data['metadata'].items():
            metadata[key] = value        
        
    return metadata
    
    
def make_rag_data(data, key):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=3000,
        chunk_overlap=500,
    )

    def chunking2docs(content):
        return text_splitter.create_documents([content])

    
    metadata = make_metadata(data, key)
       
    page_content = data['content']
    
    docs = chunking2docs(page_content)
    
    result = []
    for idx, doc in enumerate(docs):
        metadata['content_part'] = idx+1
        result.append(Document(page_content=doc.page_content, metadata=metadata))
        
    return result

def get_rag_data1():
    docs = []
    for key, df in datasets.items():
        for _, row in df.iterrows():
            docs += make_rag_data(row, key)
            
    return docs