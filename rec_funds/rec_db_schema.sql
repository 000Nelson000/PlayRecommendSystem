CREATE table 基金推薦_用戶屬性  (
	userid varchar(12),
	國內債券型 smallint,
	國內其他型 smallint,
	國內股票型 smallint,
	國內貨幣型 smallint,
	國外債券型 smallint,
	國外其他型 smallint,
	國外股票型 smallint,
	國外貨幣型 smallint,
	[aum<10萬] smallint,
	[aum10~50萬] smallint,
	[aum50~100萬] smallint,
	[aum100~300萬] smallint,
	[aum>300萬] smallint		
)

CREATE TABLE 基金推薦_推薦清單(
	rank varchar(2),
	fundid varchar(10),
	userid varchar(15),
	model varchar(10),
	score decimal(7,6),
	tag_features varchar(200)
)

CREATE TABLE 基金推薦_模型評估(
	model varchar(10),
	recall decimal(5,4),
	eval_time varchar(10),
	num_users int,
	num_funds int,
	topN smallint,
	[date] varchar(8)	
)  