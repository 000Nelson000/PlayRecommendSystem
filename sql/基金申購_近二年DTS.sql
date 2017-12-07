

--- DTS  START==================================================================
---------------------------------------------------------------------------------------------------------------------------------
-- 基金申購(近N年)
---------------------------------------------------------------------------------------------------------------------------------
use project2017
exec source.dbo.up_droptable 'project2017.dbo.基金推薦_近二年申購_憑證歸戶'
exec source.dbo.up_droptable 'project2017.dbo.基金推薦_近二年申購_ID歸戶'
declare @startdate smalldatetime,
-- 	@enddate smalldatetime,
	@today smalldatetime

set @today=getdate()
set @startdate = dateadd(yy,-2,getdate())  -- 二年前

-------------------------------------------------------------
-- BASE 
-------------------------------------------------------------
select * 
into #近二年申購
from db_wm.dbo.基金申購扣款追蹤
where 申購登錄日>=@startdate
	and 申購登錄日<=@today
	and 申購扣款日 >= @startdate
	and 申購扣款日 <= @today
	and len(身分證字號) = 11
-------------------------------------------------------------
--憑證歸戶
-------------------------------------------------------------

select 身分證字號,
	憑證分行別+憑證基金別+憑證流水號 as 憑證,
	憑證基金別 as 基金代碼,
	商品投資屬性,
	convert(varchar(8),申購登錄日,112) as 申購登錄日,
	count(*) as 扣款次數,
	sum(申購扣款金額_台幣) as 申購扣款金額_台幣,
	國內外基金註記,
	AUM基金型態別
into #近二年申購_憑證歸戶 
from #近二年申購
group by 身分證字號,
	憑證分行別+憑證基金別+憑證流水號,
	憑證基金別,
	商品投資屬性,
	申購登錄日,
	國內外基金註記,
	AUM基金型態別

select *,
	case when substring(a.憑證,7,1) in ('1','3','5','B','D') then 'a.定時定額'
		when substring(a.憑證,7,1) in ('0','2','4','A','C') then 'b.單筆申購' else 'ND' end as 投資型態,
	case when a.國內外基金註記=1 then 'a.境外基金'
		when a.國內外基金註記=0 then 'b.國內基金' else 'ND' end as 投資地區,
	case when a.AUM基金型態別='E' then 'a.股票型'
		when a.AUM基金型態別='B' then 'b.債券型'
		when a.AUM基金型態別='M' then 'c.貨幣型'
		when a.AUM基金型態別 in ('O','W','F','FT','I') then 'd.其他型' else 'ND' end as 型態別,
	case when a.AUM基金型態別='E' then 'a.股票型'
		when a.AUM基金型態別='B' then 'b.債券型'
		when a.AUM基金型態別='M' then 'c.貨幣型'
		when a.AUM基金型態別='W' then 'd.平衡型'
		when a.AUM基金型態別='F' then 'e.組合型'
		when a.AUM基金型態別='FT' then 'f.期貨型'
		when a.AUM基金型態別='I' then 'g.指數型'
		when a.AUM基金型態別='O' then 'h.其他型' else 'ND' end as AUM型態別,
	case when a.國內外基金註記=1 and a.AUM基金型態別='E' then '國外股票型'
		when a.國內外基金註記=0 and a.AUM基金型態別='E' then '國內股票型'
		when a.國內外基金註記=1 and a.AUM基金型態別='B' then '國外債券型' 
		when a.國內外基金註記=0 and a.AUM基金型態別='B' then '國內債券型' 
		when a.國內外基金註記=1 and a.AUM基金型態別='M' then '國外貨幣型'
		when a.國內外基金註記=0 and a.AUM基金型態別='M' then '國內貨幣型'
		when a.國內外基金註記=1 and a.AUM基金型態別 in ('O','W','F','FT','I') then '國外其他型'
		when a.國內外基金註記=0 and a.AUM基金型態別 in ('O','W','F','FT','I') then '國內其他型' else 'ND' end as AUM計算類別
into 基金推薦_近二年申購_憑證歸戶
from #近二年申購_憑證歸戶 a

go


-------------------------------------------------------------
--ID歸戶
-------------------------------------------------------------

select 身分證字號,
	convert(int,round(sum(申購扣款金額_台幣*convert(numeric(10),right(商品投資屬性,1)))/sum(申購扣款金額_台幣),0)) as 風險偏好,
	count(distinct 憑證) as 憑證數,
	count(*) as 扣款次數,
	sum(申購扣款金額_台幣) as 申購扣款金額_台幣
into 基金推薦_近二年申購_ID歸戶
from 基金推薦_近二年申購_憑證歸戶 
group by 身分證字號




--- DTS  END  ==================================================================


---------------------------------------------------------------------------------------------------------------------------------
-- 近兩年申購基金特徵 
---------------------------------------------------------------------------------------------------------------------------------

--- mma基金屬性 ----
select *
into #基金資料temp
from external.dbo.MMA基金基本資料_每週更新v2 a
where 更新時間 > getdate()-7
go 
--- mma 現在有 + 歷史申購基金也有 的  ---- 基金id
use project2017
select distinct a.基金代碼
into #基金id 
from 基金推薦_近二年申購_憑證歸戶 a
	inner join #基金資料temp b 
		on a.基金代碼 = b.基金代碼 

go

delete from #基金資料temp
where 基金代碼 not in (select * from #基金id)
--- 熱門申購基金(top20)註記 ----------

select top 20 基金代碼,'1' 熱門基金註記
into #熱門基金代碼
from project2017.dbo.基金推薦_近二年申購_憑證歸戶
group by 基金代碼
order by 2 desc

--- top10% 夏普值基金 

select 基金代碼,'1' [sharpe_前百分之十註記]  
into #基金_sharpe前百分之十
from #基金資料temp where sharpe in (
select top 10 percent sharpe  from #基金資料temp order by sharpe desc )

delete from #基金_sharpe前百分之十 
where 基金代碼 not in (select 基金代碼 from #基金id)

--- top10% 貝他值註記 
select 基金代碼
into #基金_beta前百分之十
from #基金資料temp where beta in (
select top 10 percent beta  from #基金資料temp order by beta desc )

delete #基金_beta前百分之十 
where 基金代碼 not in (select 基金代碼 from #基金id)

--- top10% 標準差變化(小)註記 

select 基金代碼
into #基金_年化標準差變化前百分之十
from #基金資料temp where  [年化標準差(%)]  in (
	select top 10 percent [年化標準差(%)] from #基金資料temp 
	where [年化標準差(%)] is not null
	order by [年化標準差(%)] 
)

delete #基金_年化標準差變化前百分之十 
where 基金代碼 not in (select 基金代碼 from #基金id)

-------------------------------------------------------------
-- 近兩年申購基金特徵 
-------------------------------------------------------------

use project2017 



select 基金代碼,
	case when 投資區域 = '' then NULL else 投資區域 end as 投資區域,
	基金投資產業分類1,
	基金投資產業分類2,
	基金投資產業分類3,
	基金投資區域分類1,
	基金投資區域分類2,
	基金投資區域分類3,
	基金類型,
	配息頻率,
	基金公司,
	基金經理人,
	基金目前規模區間,
	case when 基金代碼 in (select 基金代碼 from #熱門基金代碼) then '熱門基金' 
		else NULL end as '熱門基金_top20',
	case when 基金代碼 in (select 基金代碼 from #基金_sharpe前百分之十)  then '高Sharpe值'
		else NULL end as 'sharpe_top10%',
	case when 基金代碼 in (select 基金代碼 from #基金_beta前百分之十) then '高Beta值' 
		else NULL end as 'beta_top10%',
	case when 基金代碼 in (select 基金代碼 from #基金_年化標準差變化前百分之十) then '低波動' 
		else NULL end as '年化標準差top10%'
into #基金推薦_基金特徵
from #基金資料temp

select  a.*,
	case when b.高收益債註記 ='1' then '高收益債'
	else NULL end as 高收益債,
	case when b.商品投資屬性 = 'RR5' then '高風險'
	else NULL end as 風險屬性
into 基金推薦_申購基金特徵
 from #基金推薦_基金特徵 a
	left join db_wm.dbo.v_fund b
		on a.基金代碼 = b.基金代碼


--- TEST  ==================================================================

select * from 基金推薦_申購基金特徵