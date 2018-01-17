use project2017
---- 觀察幾個用戶 ---
select top 10 p.* from (select distinct userid from 基金推薦_推薦清單) p

F2221893600
E1779223910
G2727633800
L2278802960
D2721254300
F2255842850
A2709125630
F2794123860
T2710199940
Q2721076920
------------------------------------------------------------------------------------------------------------
--- 觀察用戶: Q2721076920 ----
------------------------------------------------------------------------------------------------------------
--- 1. 買啥 , Y38,J1R
select * from 基金推薦_推薦清單
where userid = 'Q2721076920'
	and model = 'ubcf' and rank =0 
select top 10 * from 基金推薦_近二年申購_憑證歸戶

--- 2. 推啥 (ubcf)
select * from 基金推薦_推薦清單
where userid = 'Q2721076920' 
	and model = 'ibcf' and rank <>0
T34
Y57
MU5
FL7
MU7
321
T35
280
J84
L0C

--- 觀察Y38.J1R用戶也買....
select distinct 身分證字號 
into #用戶A群
from 基金推薦_近二年申購_憑證歸戶 
where 基金代碼 in ('Y38','J1R')
select * 
into #A群買
from 基金推薦_近二年申購_憑證歸戶
where 身分證字號 in (select * from #用戶A群)

select 基金代碼, count(*) 賣出憑證數, sum(申購扣款金額_台幣) 申購總金額
into #A群量
from #A群買
group by 基金代碼
order by 2 desc

select top 100 * from #A群量
where 基金代碼 = 'Y57' -- 504 --- No8



--- 最相似的人 : E2724327700 , L2750917820, H1773283250, F2294556580, N1258150620
select * from 基金推薦_近二年申購_憑證歸戶
where 身分證字號 = 'E2724327700'
J87
Y38
J1R
Y38
213
303

select * from 基金推薦_近二年申購_憑證歸戶
where 身分證字號 = 'L2750917820'
Y38
J21



select * from 基金推薦_近二年申購_憑證歸戶
where 身分證字號 = 'N1258150620'


select 基金代碼, count(*) 憑證數, sum(申購扣款金額_台幣) 申購總金額 
from 基金推薦_近二年申購_憑證歸戶
group by 基金代碼
order by 2 desc
