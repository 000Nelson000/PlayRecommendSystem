select  a.���ʵn���~,
	a.�������W��,
	a.��ꫬ�A,
	a.�ʶR����,
	b.�ꤺ�~������O,
	b.����ثe�W�Ұ϶�,
	b.�ӫ~����ݩ�,
	b.�b��,
	b.sharpe,
	b.beta,
	b.[�ۤ��~�H�ӳ��S�v(%)],
	b.[�ۦ��ߤ�_���S�v(%)],
	b.�������,
	b.��ꫬ�A�O,
	b.cluster
from test.dbo.ihong_�������demo_���ʬ��� a
left join test.dbo.ihong_�������demo_����S�x b
	on a.����N�X=b.����N�X
where �����Ҧr��='A1221880930'

select 
	b.[����N�X],
	b.[�Ź��T�������] �������,
	b.[�Ҥ��~],
	c.�������W��,
-- 	b.[����W��],
	b.[�������],
	b.[���ϰ�],
	b.�b��,
	b.sharpe,
	b.beta,
	b.[�ۤ��~�H�ӳ��S�v(%)],
	b.[�ۦ��ߤ�_���S�v(%)]
from dbo.ihong_�������demo_���˲M�� a
left join  external.dbo.MMA����򥻸��_�C�g��sv2 b
	on a.fundid=b.����N�X
left join DB_WM.dbo.v_FUND c
	on c.����N�X = b.����N�X
where userid = 'A1221880930' and 	model ='popular'
and  ��s�ɶ� > getdate()-7 


select top 10 * from dbo.ihong_�������demo_�Τ�S�x
-- where uid = 'A1221880930'
select top 10 * from dbo.ihong_�������demo_����S�x


dbo.ihong_�������demo_����S�x
select top 10 * from external.dbo.MMA����򥻸��_�C�g��sv2
left join 
where  ��s�ɶ� > getdate()-7


select �������W��,* from DB_WM.dbo.v_FUND where ����N�X='J0A'

