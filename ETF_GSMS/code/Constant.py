#-*-coding:utf-8-*-
# @title: 回测的固定值
# @author: Brian Shan
# @date: 2020.07.10

import os 

money = 1_000_000

hs300 = ['000001','000002','000063','000066','000069','000100','000157','000166',
    '000333','000338','000425','000538','000568','000596','000625','000627','000651',
    '000656','000661','000671','000703','000708','000709','000723','000725','000728',
    '000768','000776','000783','000786','000858','000860','000876','000895','000938',
    '000961','000963','000977','001979','002001','002007','002008','002024','002027',
    '002032','002044','002050','002120','002129','002142','002146','002153','002157',
    '002179','002202','002230','002236','002241','002252','002271','002304','002311',
    '002352','002371','002410','002415','002422','002456','002460','002463','002466',
    '002468','002475','002493','002508','002555','002558','002594','002601','002602',
    '002607','002624','002673','002714','002736','002739','002773','002841','002916',
    '002938','002939','002945','002958','003816','300003','300014','300015','300033',
    '300059','300122','300124','300136','300142','300144','300347','300408','300413',
    '300433','300498','300601','300628','600000','600004','600009','600010','600011',
    '600015','600016','600018','600019','600025','600027','600028','600029','600030',
    '600031','600036','600038','600048','600050','600061','600066','600068','600085',
    '600089','600104','600109','600111','600115','600118','600170','600176','600177',
    '600183','600188','600196','600208','600219','600221','600233','600271','600276',
    '600297','600299','600309','600332','600340','600346','600352','600362','600369',
    '600372','600383','600390','600398','600406','600436','600438','600482','600487',
    '600489','600498','600516','600519','600522','600547','600570','600583','600585',
    '600588','600606','600637','600655','600660','600674','600690','600703','600705',
    '600741','600745','600760','600795','600809','600837','600848','600867','600886',
    '600887','600893','600900','600919','600926','600928','600958','600968','600977',
    '600989','600998','600999','601006','601009','601012','601018','601021','601066',
    '601077','601088','601100','601108','601111','601117','601138','601155','601162',
    '601166','601169','601186','601198','601211','601212','601216','601225','601229',
    '601231','601236','601238','601288','601298','601318','601319','601328','601336',
    '601360','601377','601390','601398','601555','601577','601600','601601','601607',
    '601618','601628','601633','601658','601668','601669','601688','601698','601727',
    '601766','601788','601800','601808','601816','601818','601828','601838','601857',
    '601877','601878','601881','601888','601898','601899','601901','601916','601919',
    '601933','601939','601985','601988','601989','601992','601997','601998','603019',
    '603156','603160','603259','603260','603288','603369','603501','603658','603799',
    '603833','603899','603986','603993']


whole_ls = ['510300','510330','510050','159919','510310','159949','510500',
        '159915','512500', '159968','515800','512990','512380','512160','512090',
        '159995','512760','515050','159801','512480','512290','159992','512170',
        '512010','159938','515000','515750','159807','515860','159987','515030',
        '515700','159806','512880','512000','512800','512900','159993','159928',
        '512690','515650','159996','510150','512660','512710','515210','512400',
        '515220','159966','159905','159967','510880','515180','515680','515900',
        '159976','515600','159978','511010','511260','159972','510900','159920',
        '513050','513090','513500','518880','159934','159937','518800','159980'
        ]
select_ls = ['510050', '159995', '512090', '515050', '512290', '515000', 
    '515700', '512800', '159928', '512660', '512400', '510880', '159976', 
    '511010', '510900', '518880']

commission_sz = 0.487/10000
commission_sh = 0.45/10000
commission_index = 0.23/10000
commission_bond = 3
commission_multiplier = 3
commission_stock = 2/1000


ETF_dict = {
    '510300': ('沪深300', commission_sh),
    '510330': ('沪深300', commission_sh),
    '510050': ('上证50',  commission_sh),
    '159919': ('沪深300', commission_sz),
    '510310': ('沪深300', commission_sh),
    '159949': ('创业板50',commission_sz),
    '510500': ('中证500', commission_sh),
    '159915': ('创业板500', commission_sz),
    '512500': ('中证500', commission_sh),
    '159968': ('中证500', commission_sz),
    '515800': ('中证800', commission_sh),
    '512990': ('MSCI中国A股国际通',commission_sh),
    '512380': ('MSCI中国A股',commission_sh),
    '512160': ('MSCI国际通',commission_sh),
    '512090': ('MSCI中国A股国际通',commission_sh),
    '159995': ('国证半导体芯片',commission_sz),
    '512760': ('CES半导体',commission_sh),
    '515050': ('中证5G通信主题',commission_sh),
    '159801': ('国证半导体芯片',commission_sz),
    '512480': ('中证全指半导体',commission_sh),
    '512290': ('生物医药',commission_sh),
    '159992': ('创新药产业',commission_sz),
    '512170': ('医疗',commission_sh),
    '512010': ('沪深300医药卫生',commission_sh),
    '159938': ('中证全指医药卫生',commission_sz),
    '515000': ('科技龙头',commission_sh),
    '515750': ('科技50策略',commission_sh),
    '159807': ('科技50',commission_sz),
    '515860': ('新兴科技100策略',commission_sh),
    '159987': ('研发创新100',commission_sz),
    '515030': ('新能源汽车',commission_sh),
    '515700': ('新能源汽车产业',commission_sh),
    '159806': ('新能源汽车',commission_sz),
    '512880': ('中证全指证券公司',commission_sh),
    '512000': ('中证全指证券',commission_sh),
    '512800': ('中证银行',commission_sh),
    '512900': ('中证全指证券公司',commission_sh),
    '159993': ('国证证券龙头',commission_sz),
    '159928': ('中证主要消费',commission_sz),
    '512690': ('中证酒',commission_sh),
    '515650': ('中证消费',commission_sh),
    '159996': ('中证全指家电',commission_sz),
    '510150': ('上证消费80',commission_sh),
    '512660': ('中证军工',commission_sh),
    '512710': ('中证军工龙头',commission_sh),
    '515210': ('中证钢铁',commission_sh),
    '512400': ('中证申万有色金属',commission_sh),
    '515220': ('中证煤炭',commission_sh),
    '159966': ('创业板地波蓝筹',commission_sz),
    '159905': ('深证红利',commission_sz),
    '159967': ('创业板动量成长',commission_sz),
    '510880': ('柏瑞红利',commission_sh),
    '515180': ('中证红利',commission_sh),
    '515680': ('中证央企创新驱动',commission_sh),
    '515900': ('央企创新驱动',commission_sh),
    '159976': ('粤港湾大湾区创新',commission_sz),
    '515600': ('央企创新驱动',commission_sh),
    '159978': ('港深澳大湾区',commission_sz),
    '511010': ('上证5年期国债',0),
    '511260': ('上证10年期国债',0),
    '159972': ('中证5年期地方政府债',0),
    '510900': ('恒生H股',commission_sh),
    '159920': ('恒生',commission_sz),
    '513050': ('中证海外互联',commission_sh),
    '513090': ('中证香港证券投资主题',commission_sh),
    '513500': ('标普500',commission_sh),
    '518880': ('黄金',commission_sh),
    '159934': ('黄金',commission_sz),
    '159937': ('黄金',commission_sz),
    '518800': ('黄金',commission_sh),
    '159980': ('有色金属期货',commission_sz)
}

future_ls = ['IH', 'IF', 'IC', 'TF', "T"]

future_multiplier = {
    "IH": 300, 
    "IF": 300, 
    "IC": 200, 
    "TF": 10000, 
    "T": 10000, 
}

margin_percent = {
    "IH": 0.1, 
    "IF": 0.1, 
    "IC": 0.12, 
    "TF": 0.012, 
    "T": 0.02, 
}

margin_multiplier = 3

MS_file_path = os.path.dirname(os.path.realpath(__file__)) + "\\..\\result\\MoneyStrength.csv"
log_path = os.path.dirname(os.path.realpath(__file__)) + "\\..\\log"
output_path = os.path.dirname(os.path.realpath(__file__)) + "\\..\\result"