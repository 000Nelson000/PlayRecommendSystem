# -*- coding: utf-8 -*-
""" 按照全行投資老手劃出之用戶級距，決定空中理專(舊戶+投資新戶)之個人級距    
------------------------------------------------------------------------------
目的:
    產出空專之舊戶 + 新戶之個人級距profile 

    to --- project2018.dbo.基金推薦_空專舊戶_新手profile
------------------------------------------------------------------------------
Created on Thu Aug 16 13:59:26 2018

@author: 116952
"""

import pandas as pd
import numpy as np 
import sqlalchemy 

#%% 
if __name__ == '__main__':
    engine = sqlalchemy.create_engine("mssql+pyodbc://sa:01060728@dbm_public/project2018?driver=SQL Server")
    engine.connect()

    TOPN = 10
    yyyymmdd = pd.datetime.today().strftime('%Y%m%d')
    
    sql_crm_yyyymm = '''select max(right(name,6)) from bank2018.dbo.sysobjects where xtype='V' and left(name,12)='v_CIFALL999_' '''
    yyyymm = pd.read_sql(sql_crm_yyyymm,con=engine)
    yyyymm = yyyymm.values[0][0]
    #### 
    
    '''
    取得c.舊戶(沒買) , b.投資新手(只買一檔基金) 
    join CRM 級距相關資料 , 僅用 [投資屬性,當年度ap合計_級距,年齡級距] 
    ''' 
    
    sql_read_baseid = '''select * from 基金推薦_空中理專_經管id where left(分類,1) in ('b','c') 
                        and yyyymmdd ={}'''.format(yyyymmdd)
    sql_read_crm = '''
    select a.*, b.[業績統計用資產規模(新)] as AUM,b.當年度AP合計,b.年齡, b.投資屬性
                    from ({}) a left join bank2018.dbo.v_cifall999_{} b  on a.身分證字號=b.身分證字號'''.format(sql_read_baseid,yyyymm)
    new_uid_profile = pd.read_sql(sql_read_crm,con=engine)
        
    
    sql_bins = '''select * from 基金推薦_分切級距 
    where 名稱 in ('年齡級距','當年度AP合計_級距','投資屬性') and yyyymmdd = {}'''.format(yyyymmdd)    
    bins = pd.read_sql(sql_bins,con=engine)
    ap_y_bins = bins[bins.名稱=='當年度AP合計_級距']
    ap_y_bins = ap_y_bins.級距.values.astype('float32')    

    new_ap_bins = pd.cut(new_uid_profile.當年度AP合計,bins=ap_y_bins, labels=range(len(ap_y_bins)-1))
    new_uid_profile['當年度AP合計_級距'] = new_ap_bins.astype('int')
    
    age_labels = bins[bins.名稱=='年齡級距'].級距.sort_values()
    new_age_bins = pd.cut(new_uid_profile.年齡,bins=[0,30,40,50,60,70,90],labels=age_labels)
    new_uid_profile['年齡級距'] = new_age_bins.astype('str')
    
    new_uid_profile['yyyymmdd']=yyyymmdd
    #### 存入db 
    
    new_uid_profile.to_sql('基金推薦_空專舊戶新手_profile',con=engine,index=False,if_exists='append',
                           dtype={
                                  '身分證字號':sqlalchemy.types.VARCHAR(13),
                                  '分類':sqlalchemy.types.VARCHAR(10),
                                  '投資屬性':sqlalchemy.types.SMALLINT,
                                  '年齡':sqlalchemy.types.SMALLINT,
                                  'yyyymmdd':sqlalchemy.types.VARCHAR(8),
                                  '當年度AP合計_級距':sqlalchemy.types.SMALLINT,
                                  '年齡級距': sqlalchemy.types.VARCHAR(99)
                                   })

#    
#    
