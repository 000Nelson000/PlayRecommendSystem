select  a.申購登錄年,
	a.基金中文名稱,
	a.投資型態,
	a.購買次數,
	b.國內外基金註記,
	b.基金目前規模區間,
	b.商品投資屬性,
	b.淨值,
	b.sharpe,
	b.beta,
	b.[自今年以來報酬率(%)],
	b.[自成立日起報酬率(%)],
	b.基金評等,
	b.投資型態別,
	b.cluster
from test.dbo.ihong_基金推薦demo_申購紀錄 a
left join test.dbo.ihong_基金推薦demo_基金特徵 b
	on a.基金代碼=b.基金代碼
where 身分證字號='A1221880930'

select 
	b.[基金代碼],
	b.[嘉實資訊基金評等] 基金評等,
	b.[境內外],
	c.基金中文名稱,
-- 	b.[基金名稱],
	b.[基金類型],
	b.[投資區域],
	b.淨值,
	b.sharpe,
	b.beta,
	b.[自今年以來報酬率(%)],
	b.[自成立日起報酬率(%)]
from dbo.ihong_基金推薦demo_推薦清單 a
left join  external.dbo.MMA基金基本資料_每週更新v2 b
	on a.fundid=b.基金代碼
left join DB_WM.dbo.v_FUND c
	on c.基金代碼 = b.基金代碼
where userid = 'A1221880930' and 	model ='popular'
and  更新時間 > getdate()-7 


select top 10 * from dbo.ihong_基金推薦demo_用戶特徵
-- where uid = 'A1221880930'
select top 10 * from dbo.ihong_基金推薦demo_基金特徵


dbo.ihong_基金推薦demo_基金特徵
select top 10 * from external.dbo.MMA基金基本資料_每週更新v2
left join 
where  更新時間 > getdate()-7


select 基金中文名稱,* from DB_WM.dbo.v_FUND where 基金代碼='J0A'

