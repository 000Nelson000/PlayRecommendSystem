/**********************
	api  
		基金推薦清單

@ 2017/10/30		
***********************/

var sql = require('mssql')

module.exports = {
	getModel: getModelAction
}


// function getModelPopAction(req,res){
// 	var request = new sql.Request(cp);
// 	var id = req.params.id;
// 	request.query(sqlGet)
// }

function getModelAction(req, res) {
	// var request = new sql.Request(cp);
	var id = req.params.id;
	var model = req.params.model;
	console.log("model: " + model);
	res.send({
		"model": model,
		"id": id
	});

	// request.query(sqlGetRecommended(id, model),function(err,data){
	// 	if(err){
	// 		throw err;
	// 	}else{
	// 		res.send(data);
	// 	}
	// });
}

/*sql helper*/


function sqlGetRecommended(seedid, model) {
	var id = seedid;
	var sql_recommend =
		`select b.[基金代碼],
		b.[嘉實資訊基金評等] 基金評等,
		b.[境內外],
		b.[基金名稱],
		b.[基金類型],
		b.[投資區域],
		b.淨值,
		b.sharpe,
		b.beta,
		b.[自今年以來報酬率(%)],
		b.[自成立日起報酬率(%)]
	from dbo.ihong_基金推薦demo_推薦清單 a
	left join external.dbo.MMA基金基本資料_每週更新v2 b
		on a.fundid=b.基金代碼
	where userid = '${id}' and 	model ='${model}'
	and  更新時間 > getdate()-7 
	`;
	return sql_recommend;
}
