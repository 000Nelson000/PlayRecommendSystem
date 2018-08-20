# -*- coding: utf-8 -*-
"""
利用ibcf + 群組級距熱銷 推薦給僅買一檔之空專戶
---------------------------------------------------

* 適用:
    - 僅買一檔基金之新手用戶(空中理專經管限定)
    
* 方法簡述:
    1. 利用全行老手資料計算基金相似度(item-based cf)
    2. 進行最相似基金尋找(topN)
        - 針對**僅買一檔**基金之用戶提供清單
    3. 計算基於相似老手的結果，來推薦給該用戶(topN),並加入推薦理由
    4. 投票重排序(3&4)
---------------------------------------------------
Created on Fri Aug 17 10:16:43 2018

@author: 116952
"""


#%%
from rec_funds_offline_cal import *
import sqlalchemy

def topN_simItem(sim, topN=10):
    '''由基金相似矩陣(sp)建立topN基金
    input:N
        - sim_i: sparse
        - topN: int 
    output:
        -(idx, score)
            -idx ,score: ndarray
    '''
    assert sp.issparse(sim), 'oops, input should be sparse'
    num_items = sim.shape[0]
    topN_simItemIdx = np.zeros((num_items, topN), dtype='int32')
    topN_simItemVal = np.zeros((num_items, topN), dtype='float32')
    for rowidx in range(num_items):
        row_sim = sim[rowidx, :].A
        topN_idx = np.argsort(row_sim)[0][::-1][1:topN + 1]

        topN_simItemIdx[rowidx, :] = topN_idx
        topN_simItemVal[rowidx, :] = sim[rowidx, topN_idx].A
    return topN_simItemIdx, topN_simItemVal



    
#%%
if __name__ == '__main__':
    
    con = pypyodbc.connect("DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=01060728;DATABASE=project2017")
    engine = sqlalchemy.create_engine("mssql+pyodbc://sa:01060728@dbm_public/project2018?driver=SQL Server")
    engine.connect()

    TOPN = 10  # 推薦幾檔基金?

    ### 1.讀取全行老手交易狀況,建立i-i sim ####
    data = load_data(con)
    yyyymmdd = pd.datetime.today().strftime('%Y%m%d')
    old_inter_sp = data['purchased_ui']
    idx_to_itemid = data['idx_to_itemid']
    itemid_to_idx = data['itemid_to_idx']

    _, model_i = build_model(old_inter_sp, kind='ibcf')
    sim_i = model_i.sim

    ### 2.建立空專新手 基金topN相似清單 ####
    topn_sim_itemidx, topn_sim_itemvalue = topN_simItem(sim_i, topN=TOPN)
    
    
    ## 建立topn相似基金df , 存回db ##
    topn_sim_df = pd.DataFrame()
    for iidx,(topn_sim_iidx, topn_sim_v)  in enumerate(zip(topn_sim_itemidx,topn_sim_itemvalue)):
        fund_id = idx_to_itemid.get(iidx,None)
        tmp_df = pd.DataFrame()
        
        tmp_df['sim_fundid'] = [idx_to_itemid.get(e,None) for e in topn_sim_iidx]
        tmp_df['sim'] = [v for v in topn_sim_v]
        tmp_df['fundid'] = fund_id
        topn_sim_df = topn_sim_df.append(tmp_df)
        
    topn_sim_df = topn_sim_df.rename(columns={'fundid':'基金代碼','sim_fundid':'相似基金','sim':'相似度'})
    topn_sim_df['yyyymmdd'] = pd.datetime.today().strftime('%Y%m%d')
    
    topn_sim_df.to_sql('基金推薦_基金相似度',con=engine, index=False, if_exists='append',
                       dtype={'基金代碼':sqlalchemy.types.VARCHAR(10),
                              '相似基金':sqlalchemy.types.VARCHAR(10),
                              'yyyymmdd':sqlalchemy.types.VARCHAR(8),
                              '相似度':sqlalchemy.types.DECIMAL(5,4)
                               })
    
    
    ''' 
    - 空專客戶投資新手買過...(from db)
    - 利用相似基金top10給推薦
    '''
    sql_read_newc ='''select distinct b.身分證字號 uid, 基金代碼 fundid
                        from dbo.基金推薦_空中理專_經管id a
                        	left join project2017.dbo.基金推薦_近二年申購_憑證歸戶 b 
                        		on a.身分證字號 = b.身分證字號
                        where yyyymmdd = {} and left(分類,1) ='b' '''.format(yyyymmdd)
    buy1_uid_df = pd.read_sql(sql_read_newc,con=engine)
    buy1_uid_df = buy1_uid_df.dropna().reset_index(drop=True)
    buy1_uid_df['fundidx'] = buy1_uid_df.fundid.map(
        lambda x: itemid_to_idx.get(x, -999))


    top10_fundid = pd.read_sql('select 基金代碼 fundid from 基金推薦_熱銷基金 where yyyymmdd = {}'.format(yyyymmdd),con=engine)
    buy1_sim_rec_df = pd.DataFrame()

    
    for uuidx, fundidx in enumerate(buy1_uid_df.fundidx):
        tmp_df = pd.DataFrame()

        tmp_df['rank'] = range(1, TOPN + 1)
        tmp_df['uid'] = buy1_uid_df.uid[uuidx]
        
        
        if fundidx == -999:
            ###有些用戶買的基金沒有任何交易紀錄相關性,推top10基金
            tmp_df['rec_fund'] = top10_fundid.fundid
            tmp_df['fundid']  = buy1_uid_df.iloc[uuidx,1]
            
        else:
            tmp_df['rec_fund'] = [idx_to_itemid.get(
                    iidx, None) for iidx in topn_sim_itemidx[fundidx]]
            tmp_df['fundid'] = idx_to_itemid[fundidx]

        buy1_sim_rec_df = buy1_sim_rec_df.append(tmp_df)
    buy1_sim_rec_df['rank'] = buy1_sim_rec_df['rank'].astype('uint8')
    buy1_sim_rec_df.reset_index(inplace=True, drop=True)
    
                
    ### 3. 按群組照級距劃分的 推薦清單 ###
    sql_rec_colgb = '''select 身分證字號 uid,
                        	基金代碼 rec_fund,
                        	群組排名 rank_g,
	                        '投資屬性: ' + convert(varchar(2), a.投資屬性) +
	                        ', 當年度AP合計級距: ' + convert(varchar(2),a.當年度AP合計_級距) +' ,'+a.年齡級距 + '(該組熱銷基金)'  tag_reason
                        from 基金推薦_空專舊戶新手_profile a
                        	left join 基金推薦_推薦清單_分組熱銷 b on 
                        		a.當年度AP合計_級距 = b.當年度AP合計_級距 and
                        		a.年齡級距 = b. 年齡級距 and
                        		a.投資屬性 = b.投資屬性 and
                        		a.yyyymmdd = b.yyyymmdd
                        where left(分類,1) = 'b' and a.yyyymmdd={} '''.format(yyyymmdd)
    buy1_g_rec_df = pd.read_sql(sql_rec_colgb,con=engine)
    ### 4. Voting ###
    buy1_sim_rec_df['tag_reason']= '與' + buy1_sim_rec_df.fundid+ '基金相似'
    
    
    voting_rec_df = pd.merge(buy1_sim_rec_df[['uid','rec_fund','rank','tag_reason']],
                   buy1_g_rec_df[['uid','rec_fund','rank_g','tag_reason']],
                   how='outer',
                   on=['uid','rec_fund'])
    voting_rec_df['reason'] = voting_rec_df.tag_reason_x.fillna(' ') +';' + voting_rec_df.tag_reason_y.fillna(' ')
    voting_rec_df = voting_rec_df.fillna(11)
    
    voting_rec_df['score'] = 22 - voting_rec_df.sum(axis=1)
    voting_rec_df2 = voting_rec_df[['uid','rec_fund','reason','score']].\
                        sort_values(by=['uid','score'],ascending=False).\
                        groupby('uid').head(TOPN)
    voting_rec_df2['rank'] = voting_rec_df2.groupby('uid')['score'].rank(ascending=False,method='min').astype(int)

    voting_rec_df2 = voting_rec_df2.rename(columns = {'uid':'身分證字號',
                                     'score':'分數',
                                     'reason':'推薦特色',
                                     'rec_fund':'基金代碼',
                                     'rank':'排名'})
    voting_rec_df2['yyyymmdd'] = pd.datetime.today().strftime('%Y%m%d')
    
    voting_rec_df2.to_sql('基金推薦_推薦清單_空專投資新手',con=engine,
                          if_exists='append',index=False,
                          dtype={'身分證字號': sqlalchemy.types.VARCHAR(12),
                                 '基金代碼': sqlalchemy.types.VARCHAR(10),
                                 '推薦特色': sqlalchemy.types.NVARCHAR(99),
                                 '分數': sqlalchemy.types.SMALLINT,
                                 '排名': sqlalchemy.types.SMALLINT,
                                 'yyyymmdd': sqlalchemy.types.VARCHAR(8)
                                  })
