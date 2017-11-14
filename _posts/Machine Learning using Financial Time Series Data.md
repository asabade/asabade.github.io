---
title: Machine Learnig using Financial Time Series Data with TesorFlow
subtitle: By following a Sun we can predict the performance of Stock markets that close later on in the day based on what happpend in markets earlier in the day! 
tags: [TensorFlow, Time Series, Financial Data, Vizualization]
---

**Outline...**

Time series are the lifeblood that circulates the body of finance and time series analysis is the heart that moves that fluid. That's the way finance has always functioned and always will. However, the nature of that blood, body and heart has evolved over time. There is more data, both more sources of data (e.g. more exchanges, plus social media, plus news, etc.) and more frequent delivery of data (i.e. IOS of messages per second 10 
years ago has become IOOs of 1,000s of messages per second today). More and different analysis techniques are being brought to bear. Most of 
the analysis techniques are not different in the sense of being new, and have their basis in statistics, but their applicability has closely followed 
the amount of computing power available. The growth in available computing power is faster than the growth in time series volumes and so it is 
possible to analyze time series today at scale in ways that weren't tractable previously. 
This is particularly true of machine learning, especially deep learning, and these techniques hold great promise for time series. As time series 
become more dense and many time series overlap, machine learning offers a way to the signal from the noise even when the noise can 
see overwhelming, and deep learning holds great potential because it is often the best fit for the almost random nature of financial time series. 

The premise for my investigation is straightforward, that financial markets are increasingly global and if we follow the sun from Asia to Europe to the US and so on we can use information from an earlier timezone to our advantage in a later timezone. 

Financial markets are increasingly global and if we follow the sun from Asia to Europe to the US and so on we can use information from an earlier timezone to our advantage in a later timezone.

The table below shows a number of stock market indices from around the globe, their closing times in EST, and the delay in hours between the close that index and the close of the S&P 500 in New York (hence taking EST as a base timezone). For example, Australian markets close for the day 15 hours before US markets close. So, if the close of the All Ords in Australia is a useful predictor of the close of the S&P 500 for a given day we can use that information to guide our trading activity. Continuing our example of the Australian All Ords, if this index closes up and we think that means the S&P 500 will close up as well then we should buy either stocks that compose the S&P 500 or more likely an ETF that tracks the S&P 500. 

| Index | Country |	Closing Time (EST) |	Hours Before S&P Close |
| :------ |:--- | :--- |
| All Ords |	Australia	| 0100 |	15 |
| Nikkei 225	| Japan | 0200	| 14 |
| Hang Seng |	Hong Kong	| 0400 |	12 |
| DAX	| Germany |	1130 |	4.5 |
| FTSE 100 |	UK	| 1130 |	4.5 |
| NYSE Composite	| US |	1600 |	0 |
| Dow Jones Industrial Average |	US |	1600 |	0 |
| S&P 500 |	US |	1600 |	0 |

