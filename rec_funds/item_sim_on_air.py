#!encoding : utf8

'''此腳本用於空中理專之基金推薦
---------------------------------------------------
* 適用:
    - 僅買一檔基金之新手用戶
* 方法簡述:
    1. 利用全行老手資料計算基金相似度(item-based cf)
    2. 進行最相似基金尋找(topN)
        - 針對**僅買一檔**基金之用戶提供清單
    3. 計算基於相似老手的結果，來推薦給該用戶(topN)
    4. 投票重排序(3&4)
---------------------------------------------------
** 待改
    1. db/table --> 有些用的是暫時table
    2. ...
    未完成
'''
from rec_funds_offline_cal import *


def topN_simItem(sim, topN=10):
    '''由基金相似矩陣(sp)建立topN基金
    input:
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


if __name__ == '__main__':
    con = pypyodbc.connect(
        "DRIVER={SQL Server};SERVER=dbm_public;UID=sa;PWD=01060728;DATABASE=project2017"
    )

    TOPN = 10  # 推薦幾檔基金?

    ### 1.讀取全行老手交易狀況,建立i-i sim ####
    data = load_data(con)

    old_inter_sp = data['purchased_ui']
    idx_to_itemid = data['idx_to_itemid']
    itemid_to_idx = data['itemid_to_idx']

    _, model_i = build_model(old_inter_sp, kind='ibcf')
    sim_i = model_i.sim

    ### 2.建立全基金topN相似 ####
    topn_sim_itemidx, topn_sim_itemvalue = topN_simItem(sim_i, topN=TOPN)

    ''' 
    - 空專客戶買過...(from db)
    - 
    '''
    sql_buy1_fund_id = \
        '''        
    select distinct 基金代碼 [fundid],身分證字號 [uid]
    from project2017.dbo.基金推薦_近二年申購_憑證歸戶
    where 身分證字號 in 
    (
        select 身分證字號
        from project2017.dbo.基金推薦_近二年申購_憑證歸戶
        where 身分證字號 in (
            select 身分證字號 from project2016.dbo.空中理專經營成效_基金推薦客戶分類 where 客戶分類 = 'b.信託舊戶'
        ) group by 身分證字號
        having count(distinct 基金代碼) =1
    ) 
    '''
    buy1_uid_df = pd.read_sql(sql_buy1_fund_id, con)
    buy1_uid_df['fundidx'] = buy1_uid_df.fundid.map(
        lambda x: itemid_to_idx.get(x, -999))

    buy1_sim_rec_df = pd.DataFrame()

    for uuidx, fundidx in enumerate(buy1_uid_df.fundidx):
        tmp_df = pd.DataFrame()
        if fundidx == -999:
            continue

        tmp_df['rank'] = range(1, TOPN + 1)
        tmp_df['uid'] = buy1_uid_df.uid[uuidx]
        tmp_df['fundid'] = idx_to_itemid[fundidx]
        tmp_df['rec_fund'] = [idx_to_itemid.get(
            iidx, None) for iidx in topn_sim_itemidx[fundidx]]

        buy1_sim_rec_df = buy1_sim_rec_df.append(tmp_df)
    buy1_sim_rec_df['rank'] = buy1_sim_rec_df['rank'].astype('uint8')
    buy1_sim_rec_df.reset_index(inplace=True, drop=True)

    ### 3.相似老手的推薦基金 ###

    ### 4. Voting ###
