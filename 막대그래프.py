import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (matplotlib가 웹에서 자동 지원하지 않을 수 있으니 시스템폰트 설치 시 필요)
plt.rcParams['axes.unicode_minus'] = False

# 3자리 행동코드 매핑 테이블
behavior_mapping = {
    '111': '수면',
    '112': '잠못이룸',
    '121': '식사하기',
    '122': '간식및음료',
    '131': '자기치료',
    '132': '아파서쉼',
    '133': '의료서비스',
    '141': '개인위생',
    '142': '외모관리',
    '143': '이미용서비스',
    '149': '기타개인유지',
    '210': '법인일',
    '221': '농림어업일',
    '222': '제조업일',
    '223': '서비스업일',
    '229': '기타기업일',
    '241': '일중휴식',
    '242': '일관련연수',
    '249': '기타일관련',
    '311': '학교수업',
    '312': '수업간휴식',
    '313': '자율학습',
    '314': '학교행사',
    '319': '기타학교활동',
    '321': '학원수강',
    '322': '온라인강의',
    '323': '스스로학습',
    '329': '기타학습',
    '411': '식사준비',
    '412': '간식만들기',
    '413': '설거지정리',
    '421': '세탁하기',
    '422': '세탁물건조',
    '423': '다림질정리',
    '424': '의류수선',
    '431': '청소',
    '432': '정리',
    '433': '쓰레기처리',
    '461': '반려동물돌봄',
    '462': '식물돌보기',
    '463': '동식물서비스',
    '471': '매장쇼핑',
    '472': '온라인쇼핑',
    '473': '서비스구입',
    '474': '온라인서비스',
    '479': '기타쇼핑',
    '511': '신체적돌보기',
    '512': '간호하기',
    '513': '훈육가르치기',
    '514': '책읽어주기',
    '515': '아이놀아주기',
    '516': '상담방문',
    '519': '기타돌보기',
    '711': '대면교제',
    '712': '화상교제',
    '713': '문자교제',
    '714': 'SNS교제',
    '719': '기타교제',
    '731': '개인종교활동',
    '732': '종교모임',
    '739': '기타종교',
    '821': '책읽기',
    '822': '신문보기',
    '823': '잡지보기',
    '824': '방송시청',
    '825': '비디오시청',
    '826': '라디오듣기',
    '827': '음악듣기',
    '828': '인터넷검색',
    '829': '기타미디어',
    '831': '걷기산책',
    '832': '달리기조깅',
    '833': '등산',
    '834': '자전거',
    '835': '개인운동',
    '836': '구기운동',
    '837': '낚시사냥',
    '839': '기타스포츠',
    '841': '집단게임',
    '842': 'PC게임',
    '843': '모바일게임',
    '849': '기타게임',
    '851': '휴식',
    '852': '담배피우기',
    '891': '개인취미',
    '892': '교양학습',
    '893': '유흥',
    '899': '기타여가',
    '910': '개인유지이동',
    '921': '출근',
    '922': '퇴근',
    '923': '출장이동',
    '924': '일이동',
    '930': '학습이동',
    '940': '가정관리이동',
    '951': '아이돌봄이동',
    '952': '미성년돌봄이동',
    '953': '성인돌봄이동',
    '954': '독립성인돌봄이동',
    '960': '자원봉사이동',
    '970': '교제이동',
    '980': '문화여가이동'
}

def map_code_to_name(code):
    return behavior_mapping.get(code, f"미분류({code})")

def extract_first_three_digits(code):
    try:
        code_str = str(code)
        digits = ''.join(filter(str.isdigit, code_str))
        if len(digits) >= 3:
            return digits[:3]
        return digits.ljust(3, '0')
    except:
        return '000'

def is_main_activity_column(col):
    return '주행동시간대' in col and '동시행동시간대' not in col

def parse_time_from_column(col):
    # 시간 문자열에서 분 단위로 키 생성 (예: 08:10 -> 8*60+10=490)
    import re
    patterns = [
        r'(오전|오후)\s*(\d{1,2}):(\d{2})',
        r'(\d{1,2}):(\d{2})',
        r'(\d{1,2})시\s*(\d{1,2})분?'
    ]
    for pattern in patterns:
        m = re.search(pattern, col)
        if m:
            if len(m.groups()) == 3:
                period, hour, minute = m.group(1), int(m.group(2)), int(m.group(3))
                if period == '오전' and hour == 12:
                    hour = 0
                elif period == '오후' and hour != 12:
                    hour += 12
                return hour * 60 + minute
            elif len(m.groups()) == 2:
                hour, minute = int(m.group(1)), int(m.group(2))
                return hour * 60 + minute
    return 9999

def group_hourly(analysis_results):
    hourly_data = {}
    for res in analysis_results:
        hour = res['sort_key'] // 60
        if hour not in hourly_data:
            hourly_data[hour] = {'total_count':0, 'behavior_counts':{}}
        hourly_data[hour]['total_count'] += res['total_count']
        for b in res['top_behaviors']:
            hourly_data[hour]['behavior_counts'][b['code']] = hourly_data[hour]['behavior_counts'].get(b['code'],0) + b['count']
    hourly_results = []
    for hour in range(24):
        if hour in hourly_data:
            total = hourly_data[hour]['total_count']
            bc = hourly_data[hour]['behavior_counts']
            behaviors = [{'code':c, 'name':map_code_to_name(c), 'count':cnt, 'percentage':cnt*100/total} for c,cnt in bc.items()]
            behaviors.sort(key=lambda x: x['percentage'], reverse=True)
            top3 = behaviors[:3]
            hourly_results.append({'hour':hour, 'total_count':total, 'top_behaviors':top3})
        else:
            hourly_results.append({'hour':hour, 'total_count':0, 'top_behaviors':[]})
    return hourly_results

# Streamlit 앱 시작 ----------------------------------------------------

st.title("📊 생활시간조사 재실자 행동 분석 시스템")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write(f"데이터 {len(data):,}건 로드 완료")
        
        # 자동 컬럼 매핑 (간단 버전)
        columns = data.columns.str.lower()
        region_col = next((c for c in data.columns if '시도' in c or 'region' in c.lower()), None)
        weekday_col = next((c for c in data.columns if '요일' in c or 'weekday' in c.lower()), None)
        household_col = next((c for c in data.columns if '가구원' in c or 'household' in c.lower()), None)
        gender_col = next((c for c in data.columns if '성별' in c or 'sex' in c.lower() or 'gender' in c.lower()), None)
        age_col = next((c for c in data.columns if '연령' in c or 'age' in c.lower()), None)
        marriage_col = next((c for c in data.columns if '혼인' in c or 'marriage' in c.lower()), None)
        
        # 조건 선택 UI
        region = st.selectbox("행정구역 선택", options=["전체"] + sorted(data[region_col].dropna().unique().astype(str).tolist()) if region_col else ["전체"])
        weekday = st.selectbox("요일 선택", options=["전체"] + sorted(data[weekday_col].dropna().unique().astype(str).tolist()) if weekday_col else ["전체"])
        household = st.selectbox("가구원수 선택", options=["전체"] + sorted(data[household_col].dropna().unique().astype(str).tolist()) if household_col else ["전체"])
        gender = st.selectbox("성별 선택", options=["전체"] + sorted(data[gender_col].dropna().unique().astype(str).tolist()) if gender_col else ["전체"])
        age = st.selectbox("연령대 선택", options=["전체"] + sorted(data[age_col].dropna().unique().astype(str).tolist()) if age_col else ["전체"])
        marriage = st.selectbox("혼인상태 선택", options=["전체"] + sorted(data[marriage_col].dropna().unique().astype(str).tolist()) if marriage_col else ["전체"])
        
        def filter_df(df):
            tmp = df.copy()
            if region_col and region != "전체":
                tmp = tmp[tmp[region_col].astype(str) == region]
            if weekday_col and weekday != "전체":
                tmp = tmp[tmp[weekday_col].astype(str) == weekday]
            if household_col and household != "전체":
                tmp = tmp[tmp[household_col].astype(str) == household]
            if gender_col and gender != "전체":
                tmp = tmp[tmp[gender_col].astype(str) == gender]
            if age_col and age != "전체":
                tmp = tmp[tmp[age_col].astype(str) == age]
            if marriage_col and marriage != "전체":
                tmp = tmp[tmp[marriage_col].astype(str) == marriage]
            return tmp
        
        if st.button("분석 실행"):
            filtered = filter_df(data)
            st.write(f"조건 필터링 후 데이터 수: {len(filtered):,}개")
            if len(filtered) == 0:
                st.warning("조건에 맞는 데이터가 없습니다.")
            else:
                # 주행동시간대 컬럼 선택
                time_cols = [col for col in filtered.columns if is_main_activity_column(col)]
                time_cols.sort(key=parse_time_from_column)
                analysis_results = []
                for col in time_cols:
                    codes = filtered[col].dropna()
                    if len(codes) == 0:
                        continue
                    three_digits = codes.apply(extract_first_three_digits)
                    vc = three_digits.value_counts()
                    total = len(three_digits)
                    top_behaviors = []
                    for i, (code, cnt) in enumerate(vc.items()):
                        p = cnt * 100 / total
                        top_behaviors.append({'rank': i+1, 'code': code, 'name': map_code_to_name(code), 'count': cnt, 'percentage': p})
                    analysis_results.append({'time': col, 'total_count': total, 'top_behaviors': top_behaviors, 'sort_key': parse_time_from_column(col)})
                analysis_results.sort(key=lambda x: x['sort_key'])
                hourly_results = group_hourly(analysis_results)
                
                # 결과 텍스트 보여주기
                for hr in hourly_results:
                    st.markdown(f"### {hr['hour']:02d}:00~{hr['hour']:02d}:59 (총 {hr['total_count']}개)")
                    for i, b in enumerate(hr['top_behaviors']):
                        st.markdown(f"{i+1}위: **{b['name']}** ({b['percentage']:.1f}%)")
                
                # 막대그래프 그리기
                hours = [hr['hour'] for hr in hourly_results]
                bar_width = 0.28
                
                def get_behavior_data(idx):
                    return [hr['top_behaviors'][idx]['percentage'] if len(hr['top_behaviors']) > idx else 0 for hr in hourly_results]
                def get_behavior_labels(idx):
                    return [hr['top_behaviors'][idx]['name'] if len(hr['top_behaviors']) > idx else "" for hr in hourly_results]
                
                b1 = get_behavior_data(0)
                b2 = get_behavior_data(1)
                b3 = get_behavior_data(2)
                l1 = get_behavior_labels(0)
                l2 = get_behavior_labels(1)
                l3 = get_behavior_labels(2)
                
                x = np.arange(len(hours))
                fig, ax = plt.subplots(figsize=(16, 8))
                
                bars1 = ax.bar(x - bar_width, b1, bar_width, label=l1[0] if l1 else "1위 행동")
                bars2 = ax.bar(x, b2, bar_width, label=l2[0] if l2 else "2위 행동")
                bars3 = ax.bar(x + bar_width, b3, bar_width, label=l3[0] if l3 else "3위 행동")
                
                font_size = 9
                rotation = 30
                
                def label_bars(bars, labels, data, y_offset=0):
                    for bar, label, val in zip(bars, labels, data):
                        if val > 2 and label:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2, height + y_offset, f'{label}\n{val:.1f}%', ha='center', va='bottom', rotation=rotation, fontsize=font_size,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='none'))
                
                label_bars(bars1, l1, b1)
                label_bars(bars2, l2, b2, y_offset=3)
                label_bars(bars3, l3, b3, y_offset=6)
                
                ax.set_xticks(x)
                ax.set_xticklabels([f"{hour}:00" for hour in hours], rotation=rotation)
                ax.set_ylabel("비율 (%)")
                ax.set_title("시간대별 상위 3개 행동 비율")
                ax.set_ylim(0, max(max(b1), max(b2), max(b3)) * 1.5)
                ax.legend(["1위 행동", "2위 행동", "3위 행동"])
                ax.grid(True, linestyle='--', alpha=0.3)
                
                st.pyplot(fig)
    except Exception as e:
        st.error(f"오류 발생: {e}")
