create view v_ihong_基金推薦demo_基金特徵 as 

select 基金代碼,
	case when 國內外基金註記=1 then '國外'
	when 國內外基金註記=0 then '國內' end as [特徵1_國內外],
	基金投資產業分類3 [特徵2_產業別],
	case when 高收益債註記 = 1 then '高收益債'
	else null end as [特徵3_高收益債],
	case when 商品投資屬性 ='RR5' then '高風險'
	else null end as [特徵4_高風險],
	case when 熱賣基金註記 = 1 then '熱銷基金'
	else null end as [特徵5_熱門基金],
	投資型態別 [特徵6_基金型態別],
	case when [自今年以來報酬率(%)] >10 then '今年來報酬率大於10%'
	else null end as [特徵7_高報酬]
from dbo.ihong_基金推薦demo_基金特徵

-----------
select top 100 * from v_ihong_基金推薦demo_基金特徵 
select count(*) from 