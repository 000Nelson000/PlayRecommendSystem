use project2017
---- �[��X�ӥΤ� ---
select top 10 p.* from (select distinct userid from �������_���˲M��) p

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
--- �[��Τ�: Q2721076920 ----
------------------------------------------------------------------------------------------------------------
--- 1. �Rԣ , Y38,J1R
select * from �������_���˲M��
where userid = 'Q2721076920'
	and model = 'ubcf' and rank =0 
select top 10 * from �������_��G�~����_�����k��

--- 2. ��ԣ (ubcf)
select * from �������_���˲M��
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

--- �[��Y38.J1R�Τ�]�R....
select distinct �����Ҧr�� 
into #�Τ�A�s
from �������_��G�~����_�����k�� 
where ����N�X in ('Y38','J1R')
select * 
into #A�s�R
from �������_��G�~����_�����k��
where �����Ҧr�� in (select * from #�Τ�A�s)

select ����N�X, count(*) ��X���Ҽ�, sum(���ʦ��ڪ��B_�x��) �����`���B
into #A�s�q
from #A�s�R
group by ����N�X
order by 2 desc

select top 100 * from #A�s�q
where ����N�X = 'Y57' -- 504 --- No8



--- �̬ۦ����H : E2724327700 , L2750917820, H1773283250, F2294556580, N1258150620
select * from �������_��G�~����_�����k��
where �����Ҧr�� = 'E2724327700'
J87
Y38
J1R
Y38
213
303

select * from �������_��G�~����_�����k��
where �����Ҧr�� = 'L2750917820'
Y38
J21



select * from �������_��G�~����_�����k��
where �����Ҧr�� = 'N1258150620'


select ����N�X, count(*) ���Ҽ�, sum(���ʦ��ڪ��B_�x��) �����`���B 
from �������_��G�~����_�����k��
group by ����N�X
order by 2 desc
