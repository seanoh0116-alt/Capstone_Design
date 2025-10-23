import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

class LifeTimeAnalyzer:
    def __init__(self):
        self.data = None
        self.column_mapping = {}
        self.analysis_results = []
        self.zoom_factor = 1.0  # 줌 배율 추가
        
        # 3자리 행동코드 매핑 테이블
        self.behavior_mapping = {
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
        
        # 메인/서브 색상 팔레트 정의
        self.colors = {
            'main_primary': '#2E86AB',
            'main_secondary': '#A23B72',
            'main_accent': '#F18F01',
            'sub_light': '#F0F3FF',
            'sub_medium': '#C1D3FE',
            'sub_dark': '#1A365D',
            'neutral_light': '#F7FAFC',
            'neutral_medium': '#E2E8F0',
            'neutral_dark': '#2D3748',
            'success': '#38A169',
            'warning': '#D69E2E',
            'error': '#E53E3E'
        }
        self.create_main_window()
    
    def get_behavior_name(self, code):
        """3자리 코드를 한글명으로 변환 (짧게 줄임)"""
        full_name = self.behavior_mapping.get(code, f"미분류({code})")
        # 긴 이름을 줄여서 겹침 방지
        if len(full_name) > 5:
            return full_name[:5] + ".."
        return full_name
    
    def bind_mousewheel(self, widget, canvas):
        """마우스 휠 스크롤 및 줌 바인딩"""
        def _on_mousewheel(event):
            # Ctrl 키가 눌려있는지 확인
            if event.state & 0x4:  # Ctrl 키가 눌려있음
                # 줌 기능
                if event.delta > 0:
                    self.zoom_factor *= 1.1  # 확대
                else:
                    self.zoom_factor *= 0.9  # 축소
                
                # 줌 범위 제한
                self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))
                
                # 차트 다시 그리기
                self.refresh_chart()
            else:
                # 일반 스크롤
                if event.delta:
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                else:
                    if event.num == 4:
                        canvas.yview_scroll(-1, "units")
                    elif event.num == 5:
                        canvas.yview_scroll(1, "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)
            # Ctrl+휠 이벤트도 바인딩
            canvas.bind_all("<Control-MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Control-Button-4>", _on_mousewheel)
            canvas.bind_all("<Control-Button-5>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
            canvas.unbind_all("<Control-MouseWheel>")
            canvas.unbind_all("<Control-Button-4>")
            canvas.unbind_all("<Control-Button-5>")
        
        widget.bind('<Enter>', _bind_to_mousewheel)
        widget.bind('<Leave>', _unbind_from_mousewheel)
    
    def refresh_chart(self):
        """줌 변경 시 차트 다시 그리기"""
        if hasattr(self, 'hourly_results') and self.hourly_results:
            # 기존 차트 제거
            for widget in self.chart_scrollable_frame.winfo_children():
                widget.destroy()
            
            # 새로운 크기로 차트 생성
            self._create_bar_chart_with_zoom()
    
    def create_main_window(self):
        """메인 창 생성"""
        self.window = tk.Tk()
        self.window.title("🏠 생활시간조사 재실자 행동 분석 시스템 (텍스트 겹침 해결)")
        self.window.geometry("1600x1100")
        self.window.configure(bg=self.colors['neutral_light'])
        
        # 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')
        
        # 메인 컨테이너
        main_container = tk.Frame(self.window, bg=self.colors['neutral_light'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 헤더 섹션
        header_frame = tk.Frame(main_container, bg=self.colors['main_primary'], relief='flat', bd=0)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 제목
        title_frame = tk.Frame(header_frame, bg=self.colors['main_primary'])
        title_frame.pack(pady=15)
        
        title_label = tk.Label(title_frame, text="📊 생활시간조사 재실자 행동 분석 시스템", 
                              font=('Arial', 22, 'bold'), bg=self.colors['main_primary'], fg='white')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="통계청 마이크로데이터 기반 시간대별 행동 패턴 분석 (텍스트 겹침 해결)", 
                                 font=('Arial', 11), bg=self.colors['main_primary'], fg=self.colors['sub_light'])
        subtitle_label.pack(pady=(5, 0))
        
        # 버튼 섹션
        button_section = tk.Frame(main_container, bg=self.colors['sub_light'], relief='flat', bd=1)
        button_section.pack(fill=tk.X, pady=(0, 15))
        
        button_frame = tk.Frame(button_section, bg=self.colors['sub_light'])
        button_frame.pack(pady=15)
        
        # 버튼들
        self.create_medium_button(button_frame, "📁 CSV 파일 불러오기", self.load_csv, 
                                bg=self.colors['main_secondary'], fg='white', row=0, col=0)
        self.create_medium_button(button_frame, "🔍 행동 분석 실행", self.analyze_data, 
                                bg=self.colors['success'], fg='white', row=0, col=1)
        self.create_medium_button(button_frame, "📊 막대 그래프 생성", self.create_bar_charts, 
                                bg=self.colors['main_accent'], fg='white', row=0, col=2)
        
        # 줌 리셋 버튼 추가
        self.create_medium_button(button_frame, "🔍 줌 리셋", self.reset_zoom, 
                                bg=self.colors['warning'], fg='white', row=0, col=3)
        
        # 조건 선택 프레임
        self.create_compact_condition_frame(main_container)
        
        # 메인 컨텐츠 영역
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # 결과 텍스트 탭
        self.create_text_tab()
        
        # 차트 탭
        self.create_large_chart_tab()
        
        # 상태바
        self.create_status_bar()
        
        self.window.mainloop()
    
    def reset_zoom(self):
        """줌 리셋"""
        self.zoom_factor = 1.0
        if hasattr(self, 'hourly_results') and self.hourly_results:
            self.refresh_chart()
            self.update_status(f"✅ 줌 리셋 완료 - 배율: {self.zoom_factor:.1f}x")
        else:
            messagebox.showwarning("경고", "먼저 막대 그래프를 생성해주세요.")
    
    def create_medium_button(self, parent, text, command, bg, fg, row, col):
        """중간 크기의 스타일 버튼"""
        btn = tk.Button(parent, text=text, command=command, 
                       font=('Arial', 14, 'bold'), bg=bg, fg=fg,
                       width=20, height=2, relief='flat', bd=0,
                       activebackground=self.darken_color(bg), activeforeground=fg,
                       cursor='hand2')
        btn.grid(row=row, column=col, padx=8, pady=8)
        
        def on_enter(e):
            btn.config(bg=self.lighten_color(bg), relief='raised', bd=2)
        def on_leave(e):
            btn.config(bg=bg, relief='flat', bd=0)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
    
    def darken_color(self, color):
        """색상을 자연스럽게 어둡게"""
        color_map = {
            self.colors['main_secondary']: '#8B2F5A',
            self.colors['success']: '#2F855A',
            self.colors['main_accent']: '#C05621',
            self.colors['warning']: '#B7791F'
        }
        return color_map.get(color, color)
    
    def lighten_color(self, color):
        """색상을 자연스럽게 밝게"""
        color_map = {
            self.colors['main_secondary']: '#B44C7A',
            self.colors['success']: '#48BB78',
            self.colors['main_accent']: '#F6AD55',
            self.colors['warning']: '#F6E05E'
        }
        return color_map.get(color, color)
    
    def create_compact_condition_frame(self, parent):
        """컴팩트한 조건 선택 프레임"""
        condition_frame = tk.LabelFrame(parent, text="🎯 재실자 특성 선택", 
                                       font=('Arial', 10, 'bold'), 
                                       bg=self.colors['neutral_light'], 
                                       fg=self.colors['neutral_dark'], 
                                       relief='solid', bd=1, padx=15, pady=10)
        condition_frame.pack(fill="x", pady=(0, 10))
        
        for i in range(3):
            condition_frame.columnconfigure(i, weight=1)
        
        conditions = [
            ("🏙️행정구역:", ['전체', '11-서울', '21-부산', '22-대구', '23-인천', '24-광주', 
                          '25-대전', '26-울산', '29-세종', '31-경기', '32-강원',
                          '33-충북', '34-충남', '35-전북', '36-전남', '37-경북', 
                          '38-경남', '39-제주']),
            ("📅요일:", ['전체', '0-일요일', '1-월요일', '2-화요일', '3-수요일', 
                       '4-목요일', '5-금요일', '6-토요일']),
            ("👨‍👩‍👧‍👦가구원수:", ['전체', '1명', '2명', '3명', '4명', '5명', '6명', '7명', '8명', '9명']),
            ("👤성별:", ['전체', '1-남자', '2-여자']),
            ("🎂연령코드:", ['전체', '1-10대', '2-20대', '3-30대', '4-40대', '5-50대', '6-60대이상']),
            ("💑혼인상태:", ['전체', '1-미혼', '2-배우자있음', '3-사별', '4-이혼'])
        ]
        
        self.condition_vars = {}
        
        for i, (label_text, values) in enumerate(conditions):
            row = i // 3
            col = i % 3
            
            item_frame = tk.Frame(condition_frame, bg=self.colors['neutral_light'])
            item_frame.grid(row=row, column=col, sticky="ew", pady=3, padx=5)
            
            label = tk.Label(item_frame, text=label_text, 
                           font=('Arial', 9, 'bold'), 
                           bg=self.colors['neutral_light'], 
                           fg=self.colors['sub_dark'])
            label.pack(anchor="w")
            
            var = tk.StringVar()
            combo = ttk.Combobox(item_frame, textvariable=var, values=values, 
                               state="readonly", width=18, font=('Arial', 8))
            combo.pack(fill="x", pady=(2, 0))
            combo.current(0)
            
            var_name = label_text.replace(':', '').replace('🏙️', '').replace('📅', '').replace('👨‍👩‍👧‍👦', '').replace('👤', '').replace('🎂', '').replace('💑', '')
            self.condition_vars[var_name] = var
        
        # 변수 매핑
        self.region_var = self.condition_vars['행정구역']
        self.weekday_var = self.condition_vars['요일']
        self.household_var = self.condition_vars['가구원수']
        self.gender_var = self.condition_vars['성별']
        self.age_var = self.condition_vars['연령코드']
        self.marriage_var = self.condition_vars['혼인상태']
    
    def create_text_tab(self):
        """텍스트 결과 탭"""
        self.text_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.text_frame, text="📋 분석 결과")
        
        text_container = tk.Frame(self.text_frame, bg=self.colors['sub_light'], relief='solid', bd=1)
        text_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.result_text = tk.Text(text_container, width=80, height=20, 
                                  font=('Consolas', 9), 
                                  bg=self.colors['neutral_light'], 
                                  fg=self.colors['neutral_dark'],
                                  relief='flat', wrap=tk.WORD, padx=15, pady=15,
                                  selectbackground=self.colors['sub_medium'])
        
        text_scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=self.result_text.yview)
        
        self.result_text.pack(side="left", fill=tk.BOTH, expand=True)
        text_scrollbar.pack(side="right", fill="y")
        self.result_text.config(yscrollcommand=text_scrollbar.set)
        
        welcome_msg = """
🎉 생활시간조사 분석 프로그램에 오신 것을 환영합니다!

📋 사용 순서:
1️⃣ '📁 CSV 파일 불러오기' → 2️⃣ 재실자 특성 선택 → 3️⃣ '🔍 행동 분석 실행' → 4️⃣ '📊 막대 그래프 생성'

💡 새로운 기능: 
- 시간대별 상위 3개 행동을 막대그래프로 표시
- Ctrl+마우스 휠로 그래프 줌 인/아웃 가능
- 🔍 줌 리셋 버튼으로 원래 크기로 복원
- 텍스트 겹침 문제 해결 (회전, 크기조절, 위치조정)
        """
        self.result_text.insert(tk.END, welcome_msg)
        self.result_text.config(state=tk.DISABLED)
    
    def create_large_chart_tab(self):
        """확대된 차트 탭 (줌 기능 포함)"""
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="📊 막대 그래프")
        
        chart_container = tk.Frame(self.chart_frame, bg=self.colors['sub_light'], relief='solid', bd=1)
        chart_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chart_canvas = tk.Canvas(chart_container, bg=self.colors['neutral_light'])
        
        v_scrollbar = ttk.Scrollbar(chart_container, orient="vertical", command=self.chart_canvas.yview)
        v_scrollbar.pack(side="right", fill="y")
        
        h_scrollbar = ttk.Scrollbar(chart_container, orient="horizontal", command=self.chart_canvas.xview)
        h_scrollbar.pack(side="bottom", fill="x")
        
        self.chart_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.chart_canvas.pack(side="left", fill="both", expand=True)
        
        self.chart_scrollable_frame = ttk.Frame(self.chart_canvas)
        
        self.chart_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.chart_canvas.configure(scrollregion=self.chart_canvas.bbox("all"))
        )
        
        self.chart_canvas.create_window((0, 0), window=self.chart_scrollable_frame, anchor="nw")
        
        # 마우스 휠 스크롤 및 줌 바인딩 추가
        self.bind_mousewheel(chart_container, self.chart_canvas)
        self.bind_mousewheel(self.chart_canvas, self.chart_canvas)
        
        initial_label = tk.Label(self.chart_scrollable_frame, 
                               text="📊 시간대별 막대 그래프가 여기에 표시됩니다\n\n분석 실행 후 '📊 막대 그래프 생성' 버튼을 클릭하세요\n🖱️ 마우스 휠로 스크롤, Ctrl+휠로 줌 가능\n📝 텍스트 겹침 문제 해결됨",
                               font=('Arial', 16), fg=self.colors['sub_dark'], 
                               bg=self.colors['neutral_light'])
        initial_label.pack(expand=True, pady=150)
    
    def create_status_bar(self):
        """상태바"""
        status_frame = tk.Frame(self.window, bg=self.colors['sub_dark'], height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="📌 준비됨 - CSV 파일을 불러와주세요", 
                                   bg=self.colors['sub_dark'], fg='white', font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 줌 배율 표시 추가
        self.zoom_label = tk.Label(status_frame, text=f"🔍 줌: {self.zoom_factor:.1f}x", 
                                  bg=self.colors['sub_dark'], fg=self.colors['sub_medium'], font=('Arial', 8))
        self.zoom_label.pack(side=tk.RIGHT, padx=(0, 20), pady=5)
        
        version_label = tk.Label(status_frame, text="v8.0 | 통계청 생활시간조사 분석기 (텍스트 겹침 해결)", 
                               bg=self.colors['sub_dark'], fg=self.colors['sub_medium'], font=('Arial', 8))
        version_label.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def update_status(self, message):
        """상태바 업데이트"""
        self.status_label.config(text=message)
        self.zoom_label.config(text=f"🔍 줌: {self.zoom_factor:.1f}x")
        self.window.update()
    
    def extract_first_three_digits(self, code):
        """행동코드에서 앞 3자리 추출"""
        try:
            code_str = str(code)
            digits = ''.join(filter(str.isdigit, code_str))
            
            if len(digits) >= 3:
                return digits[:3]
            else:
                return digits.ljust(3, '0')
        except:
            return '000'
    
    def group_by_hour(self, analysis_results):
        """10분 단위 데이터를 1시간 단위로 그룹화"""
        hourly_data = {}
        
        for result in analysis_results:
            hour = result['sort_key'] // 60
            
            if hour not in hourly_data:
                hourly_data[hour] = {
                    'total_count': 0,
                    'behavior_counts': {}
                }
            
            hourly_data[hour]['total_count'] += result['total_count']
            
            for behavior in result['top_behaviors']:
                code = behavior['code']
                count = behavior['count']
                
                if code not in hourly_data[hour]['behavior_counts']:
                    hourly_data[hour]['behavior_counts'][code] = 0
                hourly_data[hour]['behavior_counts'][code] += count
        
        hourly_results = []
        for hour in range(24):
            if hour in hourly_data:
                data = hourly_data[hour]
                total_count = data['total_count']
                behavior_counts = data['behavior_counts']
                
                behaviors = []
                for code, count in behavior_counts.items():
                    percentage = (count / total_count) * 100
                    name = self.get_behavior_name(code)
                    behaviors.append({
                        'code': code,
                        'name': name,
                        'count': count,
                        'percentage': percentage
                    })
                
                behaviors.sort(key=lambda x: x['percentage'], reverse=True)
                top_3 = behaviors[:3]
                
                hourly_results.append({
                    'hour': hour,
                    'total_count': total_count,
                    'top_behaviors': top_3
                })
            else:
                hourly_results.append({
                    'hour': hour,
                    'total_count': 0,
                    'top_behaviors': []
                })
        
        return hourly_results
    
    # 나머지 함수들은 이전과 동일하므로 생략하고 핵심 함수들만 포함
    def find_column_mapping(self):
        """실제 컬럼명 찾기"""
        patterns = {
            'region': ['SIDO', '시도', '행정구역', '지역', 'region'],
            'weekday': ['WEEKDAY', '요일', 'day', '요일구분'],
            'household': ['HH_SIZE', '가구원수', '전체가구원수', 'household'],
            'gender': ['SEX', '성별', 'gender', '성별코드'],
            'age': ['AGE_GRP', '연령코드', '연령그룹', 'age_code', 'age_group'],
            'marriage': ['MARITAL', '혼인', '혼인상태', 'marriage']
        }
        
        self.column_mapping = {}
        
        for key, possible_names in patterns.items():
            for col in self.data.columns:
                col_lower = col.lower()
                for pattern in possible_names:
                    pattern_lower = pattern.lower()
                    
                    if key == 'age':
                        if '연령코드' in col or 'age_grp' in col_lower or 'age_code' in col_lower:
                            self.column_mapping[key] = col
                            break
                        elif pattern_lower in col_lower and ('그룹' in col or 'group' in col_lower or 'grp' in col_lower):
                            self.column_mapping[key] = col
                            break
                    else:
                        if pattern_lower in col_lower:
                            self.column_mapping[key] = col
                            break
                
                if key in self.column_mapping:
                    break
        
        return self.column_mapping
    
    def is_main_activity_column(self, column_name):
        """주행동시간대 컬럼인지 확인"""
        if '주행동시간대' in column_name and '동시행동시간대' not in column_name:
            return True
        return False
    
    def parse_time_from_column(self, column_name):
        """시간 파싱"""
        patterns = [
            r'(오전|오후)\s*(\d{1,2}):(\d{2})',
            r'(\d{1,2}):(\d{2})',
            r'(\d{1,2})시\s*(\d{1,2})분?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, column_name)
            if match:
                if len(match.groups()) == 3:
                    period = match.group(1)
                    hour = int(match.group(2))
                    minute = int(match.group(3))
                    
                    if period == '오전':
                        if hour == 12:
                            hour = 0
                        hour_24 = hour
                    else:
                        if hour == 12:
                            hour_24 = 12
                        else:
                            hour_24 = hour + 12
                    
                    if hour_24 > 23:
                        hour_24 = hour_24 - 24
                    
                elif len(match.groups()) == 2:
                    hour = int(match.group(1))
                    minute = int(match.group(2)) if match.group(2) else 0
                    hour_24 = hour
                    
                    if hour_24 > 23:
                        hour_24 = hour_24 % 24
                
                if minute > 59:
                    minute = minute % 60
                
                return hour_24 * 60 + minute
        
        return 9999
    
    def load_csv(self):
        """CSV 파일 로드"""
        self.update_status("📁 CSV 파일 선택 중...")
        
        file_path = filedialog.askopenfilename(
            title="CSV 파일 선택",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.update_status("📊 데이터 로딩 중...")
            try:
                for encoding in ['cp949', 'euc-kr', 'utf-8', 'latin1']:
                    try:
                        self.data = pd.read_csv(file_path, encoding=encoding)
                        self.result_text.config(state=tk.NORMAL)
                        self.result_text.delete(1.0, tk.END)
                        
                        self.result_text.insert(tk.END, "🎉 CSV 파일 로드 성공!\n")
                        self.result_text.insert(tk.END, f"📊 총 데이터 수: {len(self.data):,}개\n")
                        self.result_text.insert(tk.END, f"🔤 사용된 인코딩: {encoding}\n\n")
                        
                        mapping = self.find_column_mapping()
                        
                        self.result_text.insert(tk.END, "🔍 컬럼 매핑 결과:\n")
                        for key, col_name in mapping.items():
                            self.result_text.insert(tk.END, f"   ✓ {key}: {col_name}\n")
                        
                        missing_keys = []
                        for key in ['region', 'weekday', 'household', 'gender', 'age', 'marriage']:
                            if key not in mapping:
                                missing_keys.append(key)
                        
                        if missing_keys:
                            self.result_text.insert(tk.END, f"⚠️ 찾지 못한 컬럼: {missing_keys}\n")
                        
                        self.check_time_columns()
                        
                        self.result_text.insert(tk.END, "\n🎯 조건을 선택하고 '🔍 행동 분석 실행'을 눌러주세요!\n")
                        self.result_text.config(state=tk.DISABLED)
                        
                        self.update_status(f"✅ 데이터 로드 완료 - {len(self.data):,}개 레코드")
                        return
                        
                    except UnicodeDecodeError:
                        continue
                
                messagebox.showerror("오류", "지원하는 인코딩으로 파일을 읽을 수 없습니다.")
                self.update_status("❌ 파일 로드 실패")
                
            except Exception as e:
                messagebox.showerror("오류", f"파일 로드 중 오류가 발생했습니다:\n{str(e)}")
                self.update_status("❌ 파일 로드 실패")
        else:
            self.update_status("📌 준비됨 - CSV 파일을 불러와주세요")
    
    def check_time_columns(self):
        """시간대 컬럼 확인"""
        main_time_columns = []
        concurrent_time_columns = []
        
        for col in self.data.columns:
            if self.is_main_activity_column(col):
                main_time_columns.append(col)
            elif '동시행동시간대' in col:
                concurrent_time_columns.append(col)
        
        main_time_columns.sort(key=self.parse_time_from_column)
        
        self.result_text.insert(tk.END, f"🕐 발견된 주행동시간대 컬럼: {len(main_time_columns)}개\n")
        self.result_text.insert(tk.END, f"🚫 제외된 동시행동시간대 컬럼: {len(concurrent_time_columns)}개\n")
        self.result_text.insert(tk.END, f"📖 등록된 행동코드: {len(self.behavior_mapping)}개\n")
        
        if len(main_time_columns) > 0:
            self.result_text.insert(tk.END, f"📝 주행동시간대 컬럼 예시 (시간순 정렬):\n")
            for i, col in enumerate(main_time_columns[:10]):
                time_part = col.replace('(주행동시간대) ', '')
                time_key = self.parse_time_from_column(col)
                hour = time_key // 60
                minute = time_key % 60
                self.result_text.insert(tk.END, f"   {i+1}. {time_part} (정렬키: {hour:02d}:{minute:02d})\n")
            if len(main_time_columns) > 10:
                self.result_text.insert(tk.END, f"   ... 외 {len(main_time_columns)-10}개\n")
    
    def analyze_data(self):
        """데이터 분석 실행"""
        if self.data is None:
            messagebox.showwarning("경고", "먼저 CSV 파일을 불러와주세요.")
            return
        
        self.update_status("🔍 데이터 분석 중...")
        
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "🔍 시간대별 행동 분석을 시작합니다...\n\n")
        
        # 선택된 조건 표시
        self.result_text.insert(tk.END, "📋 선택된 조건:\n")
        self.result_text.insert(tk.END, f"   🏙️ 행정구역: {self.region_var.get()}\n")
        self.result_text.insert(tk.END, f"   📅 요일: {self.weekday_var.get()}\n")
        self.result_text.insert(tk.END, f"   👨‍👩‍👧‍👦 가구원수: {self.household_var.get()}\n")
        self.result_text.insert(tk.END, f"   👤 성별: {self.gender_var.get()}\n")
        self.result_text.insert(tk.END, f"   🎂 연령코드: {self.age_var.get()}\n")
        self.result_text.insert(tk.END, f"   💑 혼인상태: {self.marriage_var.get()}\n\n")
        
        # 데이터 필터링
        filtered_data = self.data.copy()
        original_count = len(filtered_data)
        
        self.result_text.insert(tk.END, f"📊 전체 데이터: {original_count:,}개\n")
        
        # 각 조건별 필터링
        if self.region_var.get() != '전체' and 'region' in self.column_mapping:
            region_code = int(self.region_var.get().split('-')[0])
            col_name = self.column_mapping['region']
            before_count = len(filtered_data)
            filtered_data = filtered_data[filtered_data[col_name] == region_code]
            after_count = len(filtered_data)
            self.result_text.insert(tk.END, f"🔽 행정구역 필터: {before_count:,} → {after_count:,}개\n")
        
        if self.weekday_var.get() != '전체' and 'weekday' in self.column_mapping:
            weekday_code = int(self.weekday_var.get().split('-')[0])
            col_name = self.column_mapping['weekday']
            before_count = len(filtered_data)
            filtered_data = filtered_data[filtered_data[col_name] == weekday_code]
            after_count = len(filtered_data)
            self.result_text.insert(tk.END, f"🔽 요일 필터: {before_count:,} → {after_count:,}개\n")
        
        if self.household_var.get() != '전체' and 'household' in self.column_mapping:
            household_code = int(self.household_var.get().replace('명', ''))
            col_name = self.column_mapping['household']
            before_count = len(filtered_data)
            filtered_data = filtered_data[filtered_data[col_name] == household_code]
            after_count = len(filtered_data)
            self.result_text.insert(tk.END, f"🔽 가구원수 필터: {before_count:,} → {after_count:,}개\n")
        
        if self.gender_var.get() != '전체' and 'gender' in self.column_mapping:
            gender_code = int(self.gender_var.get().split('-')[0])
            col_name = self.column_mapping['gender']
            before_count = len(filtered_data)
            filtered_data = filtered_data[filtered_data[col_name] == gender_code]
            after_count = len(filtered_data)
            self.result_text.insert(tk.END, f"🔽 성별 필터: {before_count:,} → {after_count:,}개\n")
        
        if self.age_var.get() != '전체' and 'age' in self.column_mapping:
            age_code = int(self.age_var.get().split('-')[0])
            col_name = self.column_mapping['age']
            before_count = len(filtered_data)
            filtered_data = filtered_data[filtered_data[col_name] == age_code]
            after_count = len(filtered_data)
            self.result_text.insert(tk.END, f"🔽 연령코드 필터: {before_count:,} → {after_count:,}개\n")
        
        if self.marriage_var.get() != '전체' and 'marriage' in self.column_mapping:
            marriage_code = int(self.marriage_var.get().split('-')[0])
            col_name = self.column_mapping['marriage']
            before_count = len(filtered_data)
            filtered_data = filtered_data[filtered_data[col_name] == marriage_code]
            after_count = len(filtered_data)
            self.result_text.insert(tk.END, f"🔽 혼인상태 필터: {before_count:,} → {after_count:,}개\n")
        
        final_count = len(filtered_data)
        
        if final_count == 0:
            self.result_text.insert(tk.END, "\n❌ 선택한 조건에 맞는 데이터가 없습니다.\n")
            self.result_text.config(state=tk.DISABLED)
            self.update_status("⚠️ 조건에 맞는 데이터 없음")
            return
        
        self.result_text.insert(tk.END, f"\n✅ 필터링 완료: {original_count:,}개 → {final_count:,}개\n")
        self.result_text.insert(tk.END, f"📈 필터링 비율: {(final_count/original_count)*100:.1f}%\n\n")
        
        # 시간대별 분석
        self.collect_time_codes(filtered_data)
        
        self.result_text.config(state=tk.DISABLED)
        self.update_status(f"✅ 분석 완료 - {final_count:,}개 데이터 분석됨")
    
    def collect_time_codes(self, filtered_data):
        """시간대별 3자리 코드 수집 및 분석"""
        self.result_text.insert(tk.END, "🕐 주행동시간대 분석 중... (1시간 단위로 집계)\n\n")
        
        main_time_columns = []
        for col in filtered_data.columns:
            if self.is_main_activity_column(col):
                main_time_columns.append(col)
        
        if len(main_time_columns) == 0:
            self.result_text.insert(tk.END, "❌ 주행동시간대 컬럼을 찾을 수 없습니다.\n")
            return
        
        main_time_columns.sort(key=self.parse_time_from_column)
        
        self.analysis_results = []
        
        for time_col in main_time_columns:
            codes = filtered_data[time_col].dropna()
            if len(codes) > 0:
                three_digit_codes = codes.apply(self.extract_first_three_digits)
                code_counts = three_digit_codes.value_counts()
                total_count = len(three_digit_codes)
                
                all_behaviors = []
                for i, (code, count) in enumerate(code_counts.items()):
                    percentage = (count / total_count) * 100
                    behavior_name = self.get_behavior_name(code)
                    all_behaviors.append({
                        'rank': i + 1,
                        'code': code,
                        'name': behavior_name,
                        'count': count,
                        'percentage': percentage
                    })
                
                time_name = time_col.replace('(주행동시간대) ', '')
                
                result = {
                    'time': time_name,
                    'total_count': total_count,
                    'unique_codes': len(code_counts),
                    'top_behaviors': all_behaviors,
                    'sort_key': self.parse_time_from_column(time_col)
                }
                
                self.analysis_results.append(result)
        
        self.analysis_results.sort(key=lambda x: x['sort_key'])
        
        # 1시간 단위로 그룹화
        self.hourly_results = self.group_by_hour(self.analysis_results)
        
        if self.hourly_results:
            self.result_text.insert(tk.END, f"📈 분석된 시간대: 0시부터 23시까지 24시간\n")
            self.result_text.insert(tk.END, f"🔤 행동코드를 한글명으로 표시합니다.\n\n")
            self.result_text.insert(tk.END, "🏆 시간대별 상위 3개 행동 (1시간 단위):\n")
            self.result_text.insert(tk.END, "=" * 90 + "\n")
            
            for result in self.hourly_results:
                hour = result['hour']
                total_count = result['total_count']
                top_behaviors = result['top_behaviors']
                
                self.result_text.insert(tk.END, f"\n📅 {hour:02d}:00 ~ {hour:02d}:59 (총 {total_count:,}개)\n")
                self.result_text.insert(tk.END, "-" * 60 + "\n")
                
                if top_behaviors:
                    for i, behavior in enumerate(top_behaviors):
                        self.result_text.insert(tk.END, 
                            f"   {i+1}위: {behavior['name']} ({behavior['code']}) "
                            f"({behavior['count']:,}개, {behavior['percentage']:5.1f}%)\n")
                else:
                    self.result_text.insert(tk.END, "   데이터 없음\n")
            
            self.result_text.insert(tk.END, f"\n💡 '📊 막대 그래프 생성' 버튼을 눌러 시각화를 확인하세요!\n")
            self.result_text.insert(tk.END, f"🖱️ 막대그래프 창에서 Ctrl+마우스 휠로 줌 인/아웃 가능!\n")
            self.result_text.insert(tk.END, f"📝 텍스트 겹침 문제가 해결되었습니다!\n")
    
    def create_bar_charts(self):
        """시간대별 막대그래프 생성"""
        if not hasattr(self, 'hourly_results') or not self.hourly_results:
            messagebox.showwarning("경고", "먼저 데이터 분석을 실행해주세요.")
            return
        
        self.update_status("📊 막대 그래프 생성 중...")
        
        # 기존 차트 제거
        for widget in self.chart_scrollable_frame.winfo_children():
            widget.destroy()
        
        # 줌 적용된 차트 생성
        self._create_bar_chart_with_zoom()
        
        # 차트 탭으로 전환
        self.notebook.select(self.chart_frame)
        
        self.update_status(f"✅ 막대 그래프 생성 완료 - 24시간 시간대별 분석 (줌: {self.zoom_factor:.1f}x)")
        
        messagebox.showinfo("완료", f"🎉 시간대별 막대 그래프가 생성되었습니다!\n"
                                   f"📊 00:00부터 23:00까지 1시간 단위로 상위 3개 행동을 표시합니다.\n"
                                   f"📈 각 막대 위에 행동명과 비율이 표시됩니다.\n"
                                   f"🔍 Ctrl+마우스 휠로 줌 인/아웃 가능합니다!\n"
                                   f"📝 텍스트 겹침 문제가 해결되었습니다!")
    
    def _create_bar_chart_with_zoom(self):
        """텍스트 겹침 해결된 막대그래프 생성"""
        # 막대그래프 색상 팔레트
        bar_colors = [self.colors['main_primary'], self.colors['main_secondary'], self.colors['main_accent']]
        
        # 줌 팩터에 따른 크기 조정
        base_width = 32  # 기본 너비 확대
        base_height = 14  # 기본 높이 확대
        fig_width = base_width * self.zoom_factor
        fig_height = base_height * self.zoom_factor
        
        # 전체 24시간을 하나의 큰 그래프로 생성
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor(self.colors['neutral_light'])
        
        # 시간대별 데이터 준비
        hours = list(range(24))
        behavior_1_data = []
        behavior_2_data = []
        behavior_3_data = []
        behavior_1_labels = []
        behavior_2_labels = []
        behavior_3_labels = []
        
        for result in self.hourly_results:
            top_behaviors = result['top_behaviors']
            
            # 상위 3개 행동의 비율 추출
            if len(top_behaviors) >= 1:
                behavior_1_data.append(top_behaviors[0]['percentage'])
                behavior_1_labels.append(top_behaviors[0]['name'])
            else:
                behavior_1_data.append(0)
                behavior_1_labels.append('')
            
            if len(top_behaviors) >= 2:
                behavior_2_data.append(top_behaviors[1]['percentage'])
                behavior_2_labels.append(top_behaviors[1]['name'])
            else:
                behavior_2_data.append(0)
                behavior_2_labels.append('')
            
            if len(top_behaviors) >= 3:
                behavior_3_data.append(top_behaviors[2]['percentage'])
                behavior_3_labels.append(top_behaviors[2]['name'])
            else:
                behavior_3_data.append(0)
                behavior_3_labels.append('')
        
        # 막대 너비 (줌에 따라 조정, 더 넓게)
        bar_width = 0.28 * self.zoom_factor
        x_pos = np.arange(len(hours))
        
        # 막대그래프 생성
        bars1 = ax.bar(x_pos - bar_width, behavior_1_data, bar_width, 
                      label='1위 행동', color=bar_colors[0], alpha=0.8)
        bars2 = ax.bar(x_pos, behavior_2_data, bar_width, 
                      label='2위 행동', color=bar_colors[1], alpha=0.8)
        bars3 = ax.bar(x_pos + bar_width, behavior_3_data, bar_width, 
                      label='3위 행동', color=bar_colors[2], alpha=0.8)
        
        # 텍스트 겹침 해결을 위한 개선된 라벨 표시
        font_size = max(7, min(11, 8 * self.zoom_factor))  # 폰트 크기 조정
        rotation_angle = 35  # 회전 각도 (45도보다 약간 작게)
        
        def add_improved_labels_on_bars(bars, labels, data, offset_y=0):
            for bar, label, value in zip(bars, labels, data):
                if value > 2 and label:  # 2% 이상인 경우만 표시 (겹침 방지)
                    height = bar.get_height()
                    # Y 위치를 막대 위로 조정 (겹침 방지)
                    y_pos = height + max(1, 2 * self.zoom_factor) + offset_y
                    
                    # 텍스트 표시 (회전, 크기 조정, 위치 조정)
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{label}\n{value:.1f}%',
                           ha='center', va='bottom', 
                           fontsize=font_size, 
                           fontweight='bold',
                           color=self.colors['neutral_dark'], 
                           rotation=rotation_angle,
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', 
                                   alpha=0.8, 
                                   edgecolor='none'))  # 배경 박스로 가독성 향상
        
        # 각 막대에 라벨 추가 (Y 위치를 다르게 하여 겹침 방지)
        add_improved_labels_on_bars(bars1, behavior_1_labels, behavior_1_data, offset_y=0)
        add_improved_labels_on_bars(bars2, behavior_2_labels, behavior_2_data, offset_y=3)
        add_improved_labels_on_bars(bars3, behavior_3_labels, behavior_3_data, offset_y=6)
        
        # 그래프 설정 (줌에 따라 폰트 크기 조정)
        title_font_size = max(14, min(26, 20 * self.zoom_factor))
        label_font_size = max(12, min(20, 16 * self.zoom_factor))
        tick_font_size = max(10, min(16, 12 * self.zoom_factor))
        
        ax.set_xlabel('시간대', fontsize=label_font_size, fontweight='bold')
        ax.set_ylabel('비율 (%)', fontsize=label_font_size, fontweight='bold')
        ax.set_title(f'🕐 시간대별 상위 3개 행동 패턴 (00:00 ~ 23:59) - 줌: {self.zoom_factor:.1f}x', 
                    fontsize=title_font_size, fontweight='bold', pad=25, color=self.colors['main_primary'])
        
        # X축 설정 (회전 각도 조정)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{h:02d}시' for h in hours], rotation=30, ha='right', fontsize=tick_font_size)
        
        # Y축 설정 (여유 공간 확보)
        max_value = max(max(behavior_1_data), max(behavior_2_data), max(behavior_3_data))
        ax.set_ylim(0, max_value * 1.5)  # 텍스트 공간 확보를 위해 더 많은 여유 공간
        ax.tick_params(axis='y', labelsize=tick_font_size)
        
        # 범례 (줌에 따라 크기 조정)
        legend_font_size = max(10, min(18, 14 * self.zoom_factor))
        ax.legend(loc='upper right', fontsize=legend_font_size, frameon=True, fancybox=True, shadow=True)
        
        # 그리드 (더 부드럽게)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_facecolor(self.colors['neutral_light'])
        
        # 레이아웃 조정 (텍스트 공간 확보)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12, top=0.85, left=0.08, right=0.95)
        
        # tkinter에 차트 추가
        canvas = FigureCanvasTkAgg(fig, self.chart_scrollable_frame)
        canvas.draw()
        chart_widget = canvas.get_tk_widget()
        chart_widget.pack(pady=20, padx=20)
        
        # 차트 위젯에 마우스 휠 스크롤 및 줌 바인딩
        self.bind_mousewheel(chart_widget, self.chart_canvas)
        
        # 스크롤 영역 업데이트
        self.chart_scrollable_frame.update_idletasks()
        self.chart_canvas.configure(scrollregion=self.chart_canvas.bbox("all"))

# 프로그램 실행 - 여기가 중요한 부분입니다!
if __name__ == "__main__":
    app = LifeTimeAnalyzer()  # 이 부분이 완전해야
