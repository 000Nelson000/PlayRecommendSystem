/**********************
	api  
		基金購買紀錄

@ 2017/10/30		
***********************/

var sql = require('mssql')

module.exports = {
	getBuyHistory : getBuyHistoryAction
}

function getBuyHistoryAction(req,res){
	var request = new sql.Request(cp);
	var id = req.params.id;

	request.query(sqlBuyHistory(id),function(err,data){
		if(err){
			res.send('Error,query failed!')
			// throw err
		}else{
			res.send(data);
		}
	})
}

/*sql helper*/
function sqlBuyHistory(seedid){
	var id = seedid;
	var sql_history =
	`select  a.申購登錄年,
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
	where 身分證字號='${id}'
	`;
	return sql_history;
}