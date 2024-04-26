from filter_data import *
import re

def get_phase(raid, mode):
    
    options = PHASE[KO2EN_WORD[raid]][:] 
    
    if mode == '하드':
        try:
            options += PHASE[KO2EN_WORD[raid]+'_HARD']
        except:
            pass
        
    options += ['전체']
    return  options


def check_raid(text):
    
    RAID_TYPE1, RAID_TYPE2, RAID_TYPE3 = '군단장', '카제로스', '에픽'
    
    if '카멘' in text or '노멘' in text or '하멘' in text:
        return '카멘', RAID_TYPE1
    
    if '아브' in text or '노브' in text or '하브' in text:
        return '아브렐슈드', RAID_TYPE1
    
    if '일리' in text or '노칸' in text or '하칸' in text:
        return '일리아칸', RAID_TYPE1
    
    if '에키' in text or '노키' in text or '하키' in text:
        return '에키드나', RAID_TYPE2
    
    if '베히' in text:
        return '베히모스', RAID_TYPE3
    
    return '', ''
    
def extract_phase_in_title(text):
    # '관문'이나 '관' 바로 앞에 오는 숫자 찾기
    match = re.search(r'(\d+)\s*관문?', text)
    if match:
        return int(match.group(1))  # 찾은 숫자를 정수로 변환하여 반환
    else:
        return None  # 숫자가 없는 경우 None 반환