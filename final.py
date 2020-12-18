import speech_recognition as sr
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from pandas_datareader import data
pd.options.mode.chained_assignment = None  # default='warn'
import pickle


def stock_addition(A, B, C):  # 주식의 종가와 예측가를 저장해줍니다.
    try:
        with open('word.txt', 'rb') as f:
            stock = pickle.load(f)
        with open('word.txt', 'wb') as f:
            stock[A] = (B,C)
            pickle.dump(stock, f)
    except:
        with open('word.txt', 'wb') as f:
            pickle.dump({A: B}, f)


def stockbook():  #저장된 주식의 종가와 예측가격을 출력합니다.
    try:
        with open('word.txt', 'rb') as f:
            stock = pickle.load(f)
            for key, val in stock.items():
                print("{key} -> 현재주가,예상주가: {value}".format(key=key, value=val))
    except FileNotFoundError:
        print("저장되지 않았습니다.")

def crawling(search):  # daum포털사이트 단어사전을 크롤링해서 뜻,예문을 출력해줍니다. 추가 기능도 있습니다.
    # 종목 타입에 따라 download url이 다름. 종목코드 뒤에 .KS .KQ등이 입력되어야해서 Download Link 구분 필요
    stock_type = {
        'kospi': 'stockMkt',
        'kosdaq': 'kosdaqMkt'
    }

    # 회사명으로 주식 종목 코드를 획득할 수 있도록 하는 함수

    # download url 조합
    def get_download_stock(market_type=None):
        market_type = stock_type[market_type]

        download_link = 'http://kind.krx.co.kr/corpgeneral/corpList.do'
        download_link = download_link + '?method=download'
        download_link = download_link + '&marketType=' + market_type
        df = pd.read_html(download_link, header=0)[0]
        return df;

    # kospi 종목코드 목록 다운로드
    def get_download_kospi():
        df = get_download_stock('kospi')

        df.종목코드 = df.종목코드.map('{:06d}.KS'.format)
        return df

    # kosdaq 종목코드 목록 다운로드
    def get_download_kosdaq():
        df = get_download_stock('kosdaq')
        df.종목코드 = df.종목코드.map('{:06d}.KQ'.format)
        return df
    # kospi, kosdaq 종목코드 각각 다운로드
    kospi_df = get_download_kospi()
    kosdaq_df = get_download_kosdaq()
    # data frame merge
    code_df = pd.concat([kospi_df, kosdaq_df])
    # data frame정리
    code_df = code_df[['회사명', '종목코드']]
    # data frame title 변경 '회사명' = name, 종목코드 = 'code'
    code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'})
    # # data frame정리
    # code_df = code_df[['회사명', '종목코드']]
    # # data frame title 변경 '회사명' = name, 종목코드 = 'code'
    # code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'})
    # # 종목코드는 6자리로 구분되기때문에 0을 채워 6자리로 변경
    # code_df.code = code_df.code.map('{:06d}'.format)
    code = get_code(code_df, search)
    # # yahoo의 주식 데이터 종목은 코스피는 .KS, 코스닥은 .KQ가 붙습니다.
    # # 삼성전자의 경우 코스피에 상장되어있기때문에 '종목코드.KS'로 처리하도록 한다.
    # code = code + '.KS'
    # get_data_yahoo API를 통해서 yahho finance의 주식 종목 데이터를 가져온다.
    df = data.DataReader(code, 'yahoo', start='2020-1-1')
    df_ = df[:]
    days = (df_.index[-1] - df_.index[0]).days
    mu = ((((df_['Close'][-1]) / df_['Close'][1])) ** (
            365.0 / days)) - 1

    print('mu =', str(round(mu, 4) * 100) + "%")
    # 주가 수익률의 연간 volatility 계산해주기
    df_['Returns'] = df_['Adj Close'].pct_change()
    vol = df_['Returns'].std() * math.sqrt(252)
    print("Annual Volatility =", str(round(vol, 4) * 100) + "%")

    result = []
    S = df_['Close'][-1]
    T = 252  # 1년 영업일 기준 일 수
    dt = 1 / T
    s_path_multi = np.zeros(shape=(1000, 30))
    # 30영업일 이후
    s_path_multi[:, 0] = S

    for i in range(1000):
        Z = np.random.standard_normal(size=1000)
        for j in range(1, 30):  # 배열 크기에 따라 기간
            s_path_multi[i, j] = s_path_multi[i, j - 1] * np.exp((mu - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z[j])
        result.append(s_path_multi[i, -1])

    print("금일 ", search, "주가", S)
    print("몬테카를로 시뮬레이션으로 계산한 30 영업일 뒤 주가", search, "주가", round(np.mean(result), 2))

    # 주가 그래프를 그려주는 코드입니다.
    plt.figure(figsize=(10, 8))

    for i in range(s_path_multi.shape[0]):
        plt.plot(s_path_multi[i], color='b', linewidth=1.0)

    plt.xlabel('Time')
    plt.ylabel('Interest rate')
    plt.grid(True)
    plt.axis('tight')
    plt.show()
    plt.savefig('./search.png',dpi=300)
    plt.hist(result, bins=50)
    plt.show()
    plt.savefig('./search_1.png',dpi=300)
    add = input("종목에 추가하시겠습니까? y/n: ")
    if (add == 'y' or add == 'Y'):
        stock_addition(search, S, round(np.mean(result), 2))
        print('추가되었습니다.')

def get_code(df, name):
    code = df.query("name=='{}'".format(name))['code'].to_string(index=False)
# 위와같이 code명을 가져오면 앞에 공백이 붙어있는 상황이 발생하여 앞뒤로 strip() 하여 공백 제거
    code = code.strip()
    return code


def stock_simulation():  # 주식 시뮬레이션을 합니다. search변수에 받아서 crawiling함수로 보내줍니다.
    print("====================================")
    print("1.음성인식 2. 직접입력 ")
    print("====================================")
    choice =int(input())
    if(choice==1):
        print("주식을 말하세요!")
        rec = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            audio = rec.listen(source)
        search = rec.recognize_google(audio, language='ko-KR')
        search = search.replace(" ",'')
        # replace 함수를 통해 모든 공백 제거
        print(search)
        crawling(search)
    elif (choice==2):
        search = input("주식을 입력해주세요! : ")
        crawling(search)
while (1):
    print("====================================")
    print("1.종목 출력 2.시뮬레이션 3.종료  ")
    print("====================================")
    n = int(input("번호를 선택하세요 : "))
    if (n == 1):
        stockbook()
    elif (n == 2):
        stock_simulation()
    elif (n == 3):
        break

