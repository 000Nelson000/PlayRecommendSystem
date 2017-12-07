

--- DTS  START==================================================================
---------------------------------------------------------------------------------------------------------------------------------
-- �������(��N�~)
---------------------------------------------------------------------------------------------------------------------------------
use project2017
exec source.dbo.up_droptable 'project2017.dbo.�������_��G�~����_�����k��'
exec source.dbo.up_droptable 'project2017.dbo.�������_��G�~����_ID�k��'
declare @startdate smalldatetime,
-- 	@enddate smalldatetime,
	@today smalldatetime

set @today=getdate()
set @startdate = dateadd(yy,-2,getdate())  -- �G�~�e

-------------------------------------------------------------
-- BASE 
-------------------------------------------------------------
select * 
into #��G�~����
from db_wm.dbo.������ʦ��ڰl��
where ���ʵn����>=@startdate
	and ���ʵn����<=@today
	and ���ʦ��ڤ� >= @startdate
	and ���ʦ��ڤ� <= @today
	and len(�����Ҧr��) = 11
-------------------------------------------------------------
--�����k��
-------------------------------------------------------------

select �����Ҧr��,
	���Ҥ���O+���Ұ���O+���Ҭy���� as ����,
	���Ұ���O as ����N�X,
	�ӫ~����ݩ�,
	convert(varchar(8),���ʵn����,112) as ���ʵn����,
	count(*) as ���ڦ���,
	sum(���ʦ��ڪ��B_�x��) as ���ʦ��ڪ��B_�x��,
	�ꤺ�~������O,
	AUM������A�O
into #��G�~����_�����k�� 
from #��G�~����
group by �����Ҧr��,
	���Ҥ���O+���Ұ���O+���Ҭy����,
	���Ұ���O,
	�ӫ~����ݩ�,
	���ʵn����,
	�ꤺ�~������O,
	AUM������A�O

select *,
	case when substring(a.����,7,1) in ('1','3','5','B','D') then 'a.�w�ɩw�B'
		when substring(a.����,7,1) in ('0','2','4','A','C') then 'b.�浧����' else 'ND' end as ��ꫬ�A,
	case when a.�ꤺ�~������O=1 then 'a.�ҥ~���'
		when a.�ꤺ�~������O=0 then 'b.�ꤺ���' else 'ND' end as ���a��,
	case when a.AUM������A�O='E' then 'a.�Ѳ���'
		when a.AUM������A�O='B' then 'b.�Ũ髬'
		when a.AUM������A�O='M' then 'c.�f����'
		when a.AUM������A�O in ('O','W','F','FT','I') then 'd.��L��' else 'ND' end as ���A�O,
	case when a.AUM������A�O='E' then 'a.�Ѳ���'
		when a.AUM������A�O='B' then 'b.�Ũ髬'
		when a.AUM������A�O='M' then 'c.�f����'
		when a.AUM������A�O='W' then 'd.���ū�'
		when a.AUM������A�O='F' then 'e.�զX��'
		when a.AUM������A�O='FT' then 'f.���f��'
		when a.AUM������A�O='I' then 'g.���ƫ�'
		when a.AUM������A�O='O' then 'h.��L��' else 'ND' end as AUM���A�O,
	case when a.�ꤺ�~������O=1 and a.AUM������A�O='E' then '��~�Ѳ���'
		when a.�ꤺ�~������O=0 and a.AUM������A�O='E' then '�ꤺ�Ѳ���'
		when a.�ꤺ�~������O=1 and a.AUM������A�O='B' then '��~�Ũ髬' 
		when a.�ꤺ�~������O=0 and a.AUM������A�O='B' then '�ꤺ�Ũ髬' 
		when a.�ꤺ�~������O=1 and a.AUM������A�O='M' then '��~�f����'
		when a.�ꤺ�~������O=0 and a.AUM������A�O='M' then '�ꤺ�f����'
		when a.�ꤺ�~������O=1 and a.AUM������A�O in ('O','W','F','FT','I') then '��~��L��'
		when a.�ꤺ�~������O=0 and a.AUM������A�O in ('O','W','F','FT','I') then '�ꤺ��L��' else 'ND' end as AUM�p�����O
into �������_��G�~����_�����k��
from #��G�~����_�����k�� a

go


-------------------------------------------------------------
--ID�k��
-------------------------------------------------------------

select �����Ҧr��,
	convert(int,round(sum(���ʦ��ڪ��B_�x��*convert(numeric(10),right(�ӫ~����ݩ�,1)))/sum(���ʦ��ڪ��B_�x��),0)) as ���I���n,
	count(distinct ����) as ���Ҽ�,
	count(*) as ���ڦ���,
	sum(���ʦ��ڪ��B_�x��) as ���ʦ��ڪ��B_�x��
into �������_��G�~����_ID�k��
from �������_��G�~����_�����k�� 
group by �����Ҧr��




--- DTS  END  ==================================================================


---------------------------------------------------------------------------------------------------------------------------------
-- ���~���ʰ���S�x 
---------------------------------------------------------------------------------------------------------------------------------

--- mma����ݩ� ----
select *
into #������temp
from external.dbo.MMA����򥻸��_�C�g��sv2 a
where ��s�ɶ� > getdate()-7
go 
--- mma �{�b�� + ���v���ʰ���]�� ��  ---- ���id
use project2017
select distinct a.����N�X
into #���id 
from �������_��G�~����_�����k�� a
	inner join #������temp b 
		on a.����N�X = b.����N�X 

go

delete from #������temp
where ����N�X not in (select * from #���id)
--- �������ʰ��(top20)���O ----------

select top 20 ����N�X,'1' ����������O
into #��������N�X
from project2017.dbo.�������_��G�~����_�����k��
group by ����N�X
order by 2 desc

--- top10% �L���Ȱ�� 

select ����N�X,'1' [sharpe_�e�ʤ����Q���O]  
into #���_sharpe�e�ʤ����Q
from #������temp where sharpe in (
select top 10 percent sharpe  from #������temp order by sharpe desc )

delete from #���_sharpe�e�ʤ����Q 
where ����N�X not in (select ����N�X from #���id)

--- top10% ���L�ȵ��O 
select ����N�X
into #���_beta�e�ʤ����Q
from #������temp where beta in (
select top 10 percent beta  from #������temp order by beta desc )

delete #���_beta�e�ʤ����Q 
where ����N�X not in (select ����N�X from #���id)

--- top10% �зǮt�ܤ�(�p)���O 

select ����N�X
into #���_�~�ƼзǮt�ܤƫe�ʤ����Q
from #������temp where  [�~�ƼзǮt(%)]  in (
	select top 10 percent [�~�ƼзǮt(%)] from #������temp 
	where [�~�ƼзǮt(%)] is not null
	order by [�~�ƼзǮt(%)] 
)

delete #���_�~�ƼзǮt�ܤƫe�ʤ����Q 
where ����N�X not in (select ����N�X from #���id)

-------------------------------------------------------------
-- ���~���ʰ���S�x 
-------------------------------------------------------------

use project2017 



select ����N�X,
	case when ���ϰ� = '' then NULL else ���ϰ� end as ���ϰ�,
	�����겣�~����1,
	�����겣�~����2,
	�����겣�~����3,
	������ϰ����1,
	������ϰ����2,
	������ϰ����3,
	�������,
	�t���W�v,
	������q,
	����g�z�H,
	����ثe�W�Ұ϶�,
	case when ����N�X in (select ����N�X from #��������N�X) then '�������' 
		else NULL end as '�������_top20',
	case when ����N�X in (select ����N�X from #���_sharpe�e�ʤ����Q)  then '��Sharpe��'
		else NULL end as 'sharpe_top10%',
	case when ����N�X in (select ����N�X from #���_beta�e�ʤ����Q) then '��Beta��' 
		else NULL end as 'beta_top10%',
	case when ����N�X in (select ����N�X from #���_�~�ƼзǮt�ܤƫe�ʤ����Q) then '�C�i��' 
		else NULL end as '�~�ƼзǮttop10%'
into #�������_����S�x
from #������temp

select  a.*,
	case when b.�����q�ŵ��O ='1' then '�����q��'
	else NULL end as �����q��,
	case when b.�ӫ~����ݩ� = 'RR5' then '�����I'
	else NULL end as ���I�ݩ�
into �������_���ʰ���S�x
 from #�������_����S�x a
	left join db_wm.dbo.v_fund b
		on a.����N�X = b.����N�X


--- TEST  ==================================================================

select * from �������_���ʰ���S�x