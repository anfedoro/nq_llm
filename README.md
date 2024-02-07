 ##### Quick note ####

 This is quite unusual and strange approach to stock price prediction.
 In few works there is an attempt to convert series of candles to kind of a language, then tokenize them and process in Transformed autoregression decoder similar ad GPT model does. The goal it ot let model tread candlestick chart as a natural language, lean its semantics and try to generate new sentences (series of new tokens representing new candles)  to  predict possible price moves.
 Frankly I am struggling to train the model due limited HW resources and I am keeping sequence len too short to  make it possible to train, so this is probably one of the limiting issues. While in general the approach seems to be promising.

 Will update here if any progress.
