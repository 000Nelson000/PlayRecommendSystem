var express = require('express');
var app = express();

//set db 
var sql = require('mssql');

// database config
var config = {
  user: 'sa',
  password: '01060728',
  server: 'dbm_public',
  database: 'Test',
  options: {
    tdsVersion: '7_1',
    encrypt: false
  }
};

global.cp = new sql.Connection(config); //connection pool

app.use(express.static(__dirname + "/public"));

// api router 

var history = require('./route/buy_history');
var model = require('./route/get_model')
// history
app.get('/history/:id', history.getBuyHistory);
//model
app.get('/model/:model/:id', model.getModel);


// connect the pool and start web server
cp.connect().then(function () {
  console.log('connection pool open...');
  var server = app.listen(8000, function () {
    var host = server.address().address;
    var port = server.address().port;
    console.log('App listen at http://%s:%s', host, port);

  });
}).catch(function (err) {
  console.error('Error creating connection pool', err);
});

