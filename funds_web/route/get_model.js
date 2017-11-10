/**********************
	api  
		基金推薦清單

@ 2017/10/30		
***********************/

var sql = require('mssql')

module.exports = {
	getModel: getModelAction
}



function getModelAction(req, res) {
	var request = new sql.Request(cp);
	var id = req.params.id;
	var model = req.params.model;
	console.log("uid:" + id + "model: " + model );
	
	// res.send();

	request.query(sqlGetRecommended(id, model),function(err,data){
		if(err){
			res.send('Error,query failed!!')
			// throw err;
		}else{
			res.send(data);
		}
	});
}

/*sql helper*/


function sqlGetRecommended(seedid, model) {
	var id = seedid;
	var sql_recommend =
		`select b.[基金代碼],
		b.[嘉實資訊基金評等] 基金評等,
		b.[境內外],
		c.基金中文名稱 基金名稱,
		b.[基金類型],
		b.[投資區域],
		b.淨值,
		b.sharpe,
		b.beta,
		b.[自今年以來報酬率(%)],
		b.[自成立日起報酬率(%)],
		a.[tag_features] [基金特徵]
	from dbo.ihong_基金推薦demo_推薦清單 a
	left join external.dbo.MMA基金基本資料_每週更新v2 b
		on a.fundid=b.基金代碼
	left join dbo.ihong_基金中文id c
		on c.基金代碼 = b.基金代碼
	where userid = '${id}' and 	model ='${model}'
	and  更新時間 > getdate()-7 
	order by score desc
	`;
	return sql_recommend;
}
