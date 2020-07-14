#-*-coding:utf-8-*-
import dolphindb as ddb
import talib as ta
import re
s=ddb.session()
import pandas as pd
s.connect('10.0.60.55',8509,'admin','123456')
z=ddb.session()
z.connect('10.0.40.33',8505,'admin','123456')

class GetData():

    def _get_alpha_str(self,s):  # 获取字母
        result = ''.join(re.findall(r'[A-Za-z]', s))
        return result

    def Future_hist_Mcandle(self,product,dateStart,dateEnd,type='main',ktype='1'):
        if type=='index': #指数
            sql = f"TL_Get_Future_Index([`{product}],{dateStart},{dateEnd},`{ktype})"
        if type=='main':#主力K线
            sql=f"TL_Get_Future_Main([`{product}],{dateStart},{dateEnd},`{ktype})"
        if type=='main_weight':#主力K线加权,向后赋权
            sql=f'TL_Get_Future_Main_w([`{product}],{dateStart},{dateEnd},`{ktype})'
        data = s.run(sql)
        return data
    
    def Future_hist_Mswitch(self, product, dateStart, dateEnd):
        sql = f"select * \
            from loadTable('dfs://G2FutureMainSwitch2','FutureMainSwitch')  \
            where between(tradingDay,{dateStart}:{dateEnd}), product = `{product}, switched = true \
            order by tradingDay"
        data = s.run(sql)
        return data

    def Future_hist_Mtick(self,product,dateStart,dateEnd,type='main'):
        if type=='main': #主力TICK
            sql=f"select  * from loadTable('dfs://G2FutureMainTick2','FutureMainTick') where between(tradingDay,{dateStart}:{dateEnd}),product in [`{product}]; "
        elif type=='index': #主力tick指数
            sql=f"select  * from loadTable('dfs://G2FutureIndex2','FutureIndex') where between(tradingDay,{dateStart}:{dateEnd}),product in [`{product}]; "
        data = s.run(sql)
        return data

    def Future_hist_tick(self,symbol,dateStart,dateEnd):#期货历史TICK
        p=self._get_alpha_str(symbol)
        sql=f"select * from loadTable('dfs://FUTURE_TAQ','FUTURE_TAQ') where  between (tradingDay,{dateStart}:{dateEnd}),product= `{p},symbol=`{symbol}"
        data = s.run(sql)
        return data

    def Future_realtime_tick(self,symbols): #期货实时数据
        sql=f"select last(timestamp(add(timestamp(date),time))) as date,last(product) as product,last(preClose) as preClose,last(last) as last ,last(volume) as volume,last(oi) as oi,last(limitDown) as limitDown," \
            f"last(limitUp) as limitUp,last(askPrice1) as askPrice1,last(bidPrice1) as bidPrice1,last(askVolume1) as askVolume1,last(bidVolume1) as bidVolume1 from Future_stream where symbol in {symbols} group by symbol"
        return z.run(sql)

    def Future_hist_candle(self,symbol,dateStart,dateEnd,ktype='1'): #期货历史K线
        sql=f"TL_Get_Future(`{symbol},{dateStart},{dateEnd},`{ktype})"
        return s.run(sql)

    def Hist_candle_spread(self,symbol1,c1,symbol2,c2,dateStart,dateEnd,ktype='1'): #历史k线价差 不限期货
        sql=f"Get_History_Spread({dateStart},{dateEnd},`{symbol1},{c1},`{symbol2},{c2},{int(ktype)})"
        return s.run(sql)

    def Hist_Tick_spread(self,symbol1,c1,symbol2,c2,dateStart,dateEnd): #历史tick价差，不限期货
        sql=f"Get_History_Spread_Tick({dateStart},{dateEnd},`{symbol1},{c1},`{symbol2},{c2})"
        return s.run(sql)

    def Stock_sh_trade(self,symbol,dateStart,dateEnd):#上交所分笔成交
        sql=f"stock_l2_trade(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    def Stock_sz_trade(self,symbol,dateStart,dateEnd):#深交所分笔成交
        sql=f"stock_szl2_trade(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    def Stock_sz_order(self,symbol,dateStart,dateEnd):#深交所分笔委托
        sql=f"stock_szl2_order(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    def Stock_sh_order(self,symbol,dateStart,dateEnd):#上交所分笔委托
        sql=f"stock_shl2_order(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    def Stock_candle(self,symbol,dateStart,dateEnd,ktype='1'): #股票分时数据
        sql=f"stock_candle(`{symbol},{dateStart},{dateEnd},`{ktype})"
        return s.run(sql)

    def Stock_l2_Tick(self,symbol,dateStart,dateEnd): #股票L2行情
        sql=f"stock_tick(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    def Stock_index_Tick(self,symbol,dateStart,dateEnd):  #证券指数
        sql=f"stock_index_tick(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    def Stock_index_candle(self,symbol,dateStart,dateEnd,ktype='1'):#证券指数K线
        sql=f"stock_index_candle(`{symbol},{dateStart},{dateEnd},`{ktype})"
        return s.run(sql)

    def Option_Candle(self,underlying,dateStart,dateEnd,ktype='1'):#期权k线
        sql=f"TL_option_candle(`{underlying},{dateStart},{dateEnd},`{ktype})"
        return s.run(sql)

    def Option_Tick(self,underlying,symbol,dateStart,dateEnd):
        sql=f"TL_option_TICK(`{underlying},`{symbol},{dateStart},{dateEnd})" #期权TICK
        return s.run(sql)

    def clearcache(self): #dolphin清除缓存
        s.run("pnodeRun(clearAllCache)")



class Make_Factors():
    def Indicators(self,df):
        df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
        dfindicator = pd.DataFrame({
            'Y': df.Close.shift(-1) / df.Close - 1,
            'Return': df.Close / df.Close.shift(1) - 1,
            'Return1': df.Close.shift(1) / df.Close.shift(2) - 1,
            'Return2': df.Close.shift(2) / df.Close.shift(3) - 1,
            'HC': df.High / df.Close, 'LC': df.Low / df.Close, 'HL': df.High / df.Low, 'OL': df.Open / df.Low,
            ############Overlap Studies Functions
            'DEMA': ta.DEMA(df.Close, timeperiod=30),
            'EMA': ta.EMA(df.Close, timeperiod=30),
            'HT_TRENDLINE': ta.HT_TRENDLINE(df.Close),
            'KAMA': ta.KAMA(df.Close, timeperiod=30),
            'MA': ta.MA(df.Close, timeperiod=30, matype=0),
            # 'MAVP': ta.MAVP(df.Close, periods, minperiod=2, maxperiod=30, matype=0),
            'MIDPOINT': ta.MIDPOINT(df.Close, timeperiod=14),
            'MIDPRICE': ta.MIDPRICE(df.High, df.Low, timeperiod=14),
            'SAR': ta.SAR(df.High, df.Low, acceleration=0, maximum=0),
            'SAREXT': ta.SAREXT(df.High, df.Low, startvalue=0, offsetonreverse=0, accelerationinitlong=0,
                                accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0,
                                accelerationmaxshort=0),
            'T3': ta.T3(df.Close, timeperiod=5, vfactor=0),
            'SMA': ta.SMA(df.Close, timeperiod=30),
            'TEMA': ta.TEMA(df.Close, timeperiod=30),
            'TRIMA': ta.TRIMA(df.Close, timeperiod=30),
            'WMA': ta.WMA(df.Close, timeperiod=30),
            #######  Momentum Indicator Functions
            'ADX': ta.ADX(df.High, df.Low, df.Close, timeperiod=14),
            'ADXR': ta.ADXR(df.High, df.Low, df.Close, timeperiod=14),
            'APO': ta.APO(df.Close, fastperiod=12, slowperiod=26, matype=0),
            # aroondown, aroonup = AROON(df.High, df.Low, timeperiod=14),
            'AROONOSC': ta.AROONOSC(df.High, df.Low, timeperiod=14),
            'BOP': ta.BOP(df.Open, df.High, df.Low, df.Close),
            'CCI': ta.CCI(df.High, df.Low, df.Close, timeperiod=14),
            'CMO': ta.CMO(df.Close, timeperiod=14),
            'DX': ta.DX(df.High, df.Low, df.Close, timeperiod=14),
            # 'MFI': ta.MFI(df.High, df.Low, df.Close, df.Volume, timeperiod=14),
            'MINUS_DI': ta.MINUS_DI(df.High, df.Low, df.Close, timeperiod=14),
            'MINUS_DM': ta.MINUS_DM(df.High, df.Low, timeperiod=14),
            'MOM': ta.MOM(df.Close, timeperiod=10),
            'RSI': ta.RSI(df.Close, timeperiod=14),
            ######Volume Indicator Functions
            # 'AD': ta.AD(df.High, df.Low, df.Close, df.Volume),
            # 'ADOSC': ta.ADOSC(df.High, df.Low, df.Close, df.Volume, fastperiod=3, slowperiod=10),
            # 'OBV': ta.OBV(df.Close, df.Volume),
            ######Volatility Indicator Functions
            'ATR': ta.ATR(df.High, df.Low, df.Close, timeperiod=14),
            'NATR': ta.NATR(df.High, df.Low, df.Close, timeperiod=14),
            'TRANGE': ta.TRANGE(df.High, df.Low, df.Close),
            ####Price Transform Functions
            'AVGPRICE': ta.AVGPRICE(df.Open, df.High, df.Low, df.Close),
            'MEDPRICE': ta.MEDPRICE(df.High, df.Low),
            'TYPPRICE': ta.TYPPRICE(df.High, df.Low, df.Close),
            'WCLPRICE': ta.WCLPRICE(df.High, df.Low, df.Close),
            ######Cycle Indicator Functions
            'HT_DCPERIOD': ta.HT_DCPERIOD(df.Close),
            'HT_DCPHASE': ta.HT_DCPHASE(df.Close),
            'HT_TRENDMODE': ta.HT_TRENDMODE(df.Close),
            #######Statistic Functions
            'BETA': ta.BETA(df.High, df.Low, timeperiod=5),
            'CORREL': ta.CORREL(df.High, df.Low, timeperiod=30),
            'LINEARREG': ta.LINEARREG(df.Close, timeperiod=14),
            'LINEARREG_ANGLE': ta.LINEARREG_ANGLE(df.Close, timeperiod=14),
            'LINEARREG_INTERCEPT': ta.LINEARREG_INTERCEPT(df.Close, timeperiod=14),
            'LINEARREG_SLOPE': ta.LINEARREG_SLOPE(df.Close, timeperiod=14),
            'STDDEV': ta.STDDEV(df.Close, timeperiod=5, nbdev=1),
            'TSF': ta.TSF(df.Close, timeperiod=14),
            'VAR': ta.VAR(df.Close, timeperiod=5, nbdev=1),
            #######Math Transform Functions
            # 'ACOS': ta.ACOS(df.Close), 'ASIN': ta.ASIN(df.Close), 'COSH': ta.COSH(df.Close),
            # 'EXP': ta.EXP(df.Close), 'SINH': ta.SINH(df.Close),
            'CEIL': ta.CEIL(df.Close), 'COS': ta.COS(df.Close), 'ATAN': ta.ATAN(df.Close),
            'FLOOR': ta.FLOOR(df.Close), 'LN': ta.LN(df.Close), 'SIN': ta.SIN(df.Close),
            'LOG10': ta.LOG10(df.Close),
            'SQRT': ta.SQRT(df.Close), 'TAN': ta.TAN(df.Close), 'TANH': ta.TANH(df.Close),
            ###### Math Operator Functions
            'ADD': ta.ADD(df.High, df.Low), 'SUB': ta.SUB(df.High, df.Low),
            'MULT': ta.MULT(df.High, df.Low), 'DIV': ta.DIV(df.High, df.Low),
            'MAX': ta.MAX(df.Close, timeperiod=30), 'MIN': ta.MIN(df.Close, timeperiod=30),
            'MAXINDEX': ta.MAXINDEX(df.Close, timeperiod=30),
            'MININDEX': ta.MININDEX(df.Close, timeperiod=30),
            'SUM': ta.SUM(df.Close, timeperiod=30)
        })
        return dfindicator

    def Pattern(self,df):
        df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
        dfpattern = pd.DataFrame({
            'Y': df.Close.shift(-1) / df.Close - 1,
            'TwoCrows': ta.CDL3INSIDE(df.Open, df.High, df.Low, df.Close),  # D
            'ThreeBlackCrows': ta.CDL3BLACKCROWS(df.Open, df.High, df.Low, df.Close),  # D
            'ThreeInsideUD': ta.CDL3INSIDE(df.Open, df.High, df.Low, df.Close),  # U
            'ThreeLineStrike': ta.CDL3LINESTRIKE(df.Open, df.High, df.Low, df.Close),  # D
            'ThreeOutsideUD': ta.CDL3OUTSIDE(df.Open, df.High, df.Low, df.Close),  # U
            'ThreeStarsInTheSouth': ta.CDL3STARSINSOUTH(df.Open, df.High, df.Low, df.Close),  # U
            'ThreeAdvancingWhiteSoldiers': ta.CDL3WHITESOLDIERS(df.Open, df.High, df.Low, df.Close),  # U
            'AdvanceBlock': ta.CDLADVANCEBLOCK(df.Open, df.High, df.Low, df.Close),  # U
            'AbandonedBaby': ta.CDLABANDONEDBABY(df.Open, df.High, df.Low, df.Close, penetration=0),  # R
            'BeltHold': ta.CDLBELTHOLD(df.Open, df.High, df.Low, df.Close),  # U
            'Breakaway': ta.CDLBREAKAWAY(df.Open, df.High, df.Low, df.Close),  # U
            'ClosingMarubozu': ta.CDLCLOSINGMARUBOZU(df.Open, df.High, df.Low, df.Close),  # M
            'ConcealingBabySwallow': ta.CDLCONCEALBABYSWALL(df.Open, df.High, df.Low, df.Close),  # U
            'Counterattack': ta.CDLCOUNTERATTACK(df.Open, df.High, df.Low, df.Close),  #
            'DarkCloudCover': ta.CDLDARKCLOUDCOVER(df.Open, df.High, df.Low, df.Close, penetration=0),  # D
            'Doji': ta.CDLDOJI(df.Open, df.High, df.Low, df.Close),  #
            'DojiStar': ta.CDLDOJISTAR(df.Open, df.High, df.Low, df.Close),  # R
            'DragonflyDoji': ta.CDLDRAGONFLYDOJI(df.Open, df.High, df.Low, df.Close),  # R
            'EngulfingPattern': ta.CDLENGULFING(df.Open, df.High, df.Low, df.Close),  # R
            'EveningDojiStar': ta.CDLEVENINGDOJISTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RD
            'EveningStar': ta.CDLEVENINGSTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RU
            'UDgapSideWhiteLines': ta.CDLGAPSIDESIDEWHITE(df.Open, df.High, df.Low, df.Close),  # M
            'GravestoneDoji': ta.CDLGRAVESTONEDOJI(df.Open, df.High, df.Low, df.Close),  # RU
            'Hammer': ta.CDLHAMMER(df.Open, df.High, df.Low, df.Close),  # R
            'HangingMan': ta.CDLHANGINGMAN(df.Open, df.High, df.Low, df.Close),  # R
            'HaramiPattern': ta.CDLHARAMI(df.Open, df.High, df.Low, df.Close),  # RU
            'HaramiCrossPattern': ta.CDLHARAMICROSS(df.Open, df.High, df.Low, df.Close),  # R
            'HighWaveCandle': ta.CDLHIGHWAVE(df.Open, df.High, df.Low, df.Close),  # R
            'HikkakePattern': ta.CDLHIKKAKE(df.Open, df.High, df.Low, df.Close),  # R
            'ModifiedHikkakePattern': ta.CDLHIKKAKEMOD(df.Open, df.High, df.Low, df.Close),  # M
            'HomingPigeon': ta.CDLHOMINGPIGEON(df.Open, df.High, df.Low, df.Close),  # R
            'IdenticalThreeCrow': ta.CDLIDENTICAL3CROWS(df.Open, df.High, df.Low, df.Close),  # D
            'InNeckPattern': ta.CDLINNECK(df.Open, df.High, df.Low, df.Close),  # D
            'InvertedHammer': ta.CDLINVERTEDHAMMER(df.Open, df.High, df.Low, df.Close),  # R
            'Kicking': ta.CDLKICKING(df.Open, df.High, df.Low, df.Close),  #
            'KickingByLength': ta.CDLKICKINGBYLENGTH(df.Open, df.High, df.Low, df.Close),  #
            'LadderBottom': ta.CDLLADDERBOTTOM(df.Open, df.High, df.Low, df.Close),  # RU
            'LongLeggedDoji': ta.CDLLONGLEGGEDDOJI(df.Open, df.High, df.Low, df.Close),  #
            'LongLineCandle': ta.CDLLONGLINE(df.Open, df.High, df.Low, df.Close),  #
            'Marubozu': ta.CDLMARUBOZU(df.Open, df.High, df.Low, df.Close),  #
            'MatchingLow': ta.CDLMATCHINGLOW(df.Open, df.High, df.Low, df.Close),  #
            'MatHold': ta.CDLMATHOLD(df.Open, df.High, df.Low, df.Close, penetration=0),  # M
            'MorningDoji': ta.CDLMORNINGDOJISTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RU
            'MorningStar': ta.CDLMORNINGSTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RU
            'OnNeckPattern': ta.CDLONNECK(df.Open, df.High, df.Low, df.Close),  # MD
            'PiercingPattern': ta.CDLPIERCING(df.Open, df.High, df.Low, df.Close),  # RU
            'RickshawMan': ta.CDLRICKSHAWMAN(df.Open, df.High, df.Low, df.Close),  #
            'RFThreeMethods': ta.CDLRISEFALL3METHODS(df.Open, df.High, df.Low, df.Close),  # U
            'SeparatingLines': ta.CDLSEPARATINGLINES(df.Open, df.High, df.Low, df.Close),  # M
            'ShootingStar': ta.CDLSHOOTINGSTAR(df.Open, df.High, df.Low, df.Close),  # D
            'ShortLineCandle': ta.CDLSHORTLINE(df.Open, df.High, df.Low, df.Close),  #
            'SpinningTop': ta.CDLSPINNINGTOP(df.Open, df.High, df.Low, df.Close),  #
            'StalledPattern': ta.CDLSTALLEDPATTERN(df.Open, df.High, df.Low, df.Close),  # EU
            'StickSandwich': ta.CDLSTICKSANDWICH(df.Open, df.High, df.Low, df.Close),  #
            'Takuri': ta.CDLTAKURI(df.Open, df.High, df.Low, df.Close),  #
            'TasukiGap': ta.CDLTASUKIGAP(df.Open, df.High, df.Low, df.Close),  # MU
            'ThrustingPattern': ta.CDLTHRUSTING(df.Open, df.High, df.Low, df.Close),  # M
            'TristarPattern': ta.CDLTRISTAR(df.Open, df.High, df.Low, df.Close),  # R
            'UniqueRiver': ta.CDLUNIQUE3RIVER(df.Open, df.High, df.Low, df.Close),  # R
            'UGapTwoCrows': ta.CDLUPSIDEGAP2CROWS(df.Open, df.High, df.Low, df.Close),  # U
            'UDGapThreeMethods': ta.CDLXSIDEGAP3METHODS(df.Open, df.High, df.Low, df.Close)  # U
        })
        return dfpattern

# data=GetData()
# d=data.Future_hist_tick('IC2006','2020.06.10','2020.06.12')
# print(d)
