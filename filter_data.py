import pandas as pd

df = pd.read_json('/Users/seokholee/이모저모/DNF_chatbot_RAG/data/dnf_homepage_tip.json')


RAID = ['[레이드] 아스라한 : 무의 장막', '[레이드] 기계 혁명 : 바칼 레이드 (하드 모드)',
       '[레이드] 기계 혁명 : 바칼 레이드', '[레이드] 기계 혁명 : 개전 (1~4인)',]
REGION = ['[레기온] 어둑섬', '[레기온] 대마법사의 차원회랑 ', '[레기온] 빼앗긴 땅, 이스핀즈',]
DUNGEON = ['[특수 던전] 이면 경계', '[휘장/젬]코드네임 게이볼그 ', '상급 던전',]

EQUIMENT = ['보조 특성 (장비 특성)', '커스텀', '융합 장비',
'장비 관련 기타 유용한 정보', '장비 공통', '장비 세팅', '장비 세팅(버퍼)',]

SELECT_CATEGORY1 = ['공략', '장비', '직업']
SELECT_CATEGORY2 = ['레이드', '레기온', '기타 던전']

cols = df['category1'].unique()
JOB = [col for col in cols if col not in ['[레이드] 아스라한 : 무의 장막', '[레이드] 기계 혁명 : 바칼 레이드 (하드 모드)',
       '[레이드] 기계 혁명 : 바칼 레이드', '[레이드] 기계 혁명 : 개전 (1~4인)', '[특수 던전] 이면 경계',
       '[레기온] 어둑섬', '[레기온] 대마법사의 차원회랑 ', '[레기온] 빼앗긴 땅, 이스핀즈',
       '[휘장/젬]코드네임 게이볼그 ', '상급 던전', '보조 특성 (장비 특성)', '커스텀', '융합 장비',
       '장비 관련 기타 유용한 정보', '장비 공통', '장비 세팅', '장비 세팅(버퍼)', '\xa0',]]

CATEGORY3_DICT = {
    '레이드' : RAID,
    '레기온' : REGION,
    '기타 던전' : DUNGEON,
    '장비' : EQUIMENT,
    '직업' : JOB,
}