CREATE table �������_�Τ��ݩ�  (
	userid varchar(12),
	�ꤺ�Ũ髬 smallint,
	�ꤺ��L�� smallint,
	�ꤺ�Ѳ��� smallint,
	�ꤺ�f���� smallint,
	��~�Ũ髬 smallint,
	��~��L�� smallint,
	��~�Ѳ��� smallint,
	��~�f���� smallint,
	[aum<10�U] smallint,
	[aum10~50�U] smallint,
	[aum50~100�U] smallint,
	[aum100~300�U] smallint,
	[aum>300�U] smallint		
)

CREATE TABLE �������_���˲M��(
	rank varchar(2),
	fundid varchar(10),
	userid varchar(15),
	model varchar(10),
	score decimal(7,6),
	tag_features varchar(200)
)

CREATE TABLE �������_�ҫ�����(
	model varchar(10),
	recall decimal(5,4),
	eval_time varchar(10),
	num_users int,
	num_funds int,
	topN smallint,
	[date] varchar(8)	
)  