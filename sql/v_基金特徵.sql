create view v_ihong_�������demo_����S�x as 

select ����N�X,
	case when �ꤺ�~������O=1 then '��~'
	when �ꤺ�~������O=0 then '�ꤺ' end as [�S�x1_�ꤺ�~],
	�����겣�~����3 [�S�x2_���~�O],
	case when �����q�ŵ��O = 1 then '�����q��'
	else null end as [�S�x3_�����q��],
	case when �ӫ~����ݩ� ='RR5' then '�����I'
	else null end as [�S�x4_�����I],
	case when ���������O = 1 then '���P���'
	else null end as [�S�x5_�������],
	��ꫬ�A�O [�S�x6_������A�O],
	case when [�ۤ��~�H�ӳ��S�v(%)] >10 then '���~�ӳ��S�v�j��10%'
	else null end as [�S�x7_�����S]
from dbo.ihong_�������demo_����S�x

-----------
select top 100 * from v_ihong_�������demo_����S�x 
select count(*) from 