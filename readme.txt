** 1. Scrape Data **

To scrape EPL data:
python src/scrape_data.py --league=Premier_League

To scrape Bundesliga data:
python src/scrape_data.py --league=Bundesliga

To also scrape FIFA rankings (pass --fifa flag):
python src/scrape_data.py --league=Bundesliga --fifa


** 2. Run Experiment **

To run:
python src/<filename>.py --model=<insert model>

Experiment filenames:
1. bundesliga_players.py : Player one hot embedding on Bundesliga league
2. bundesliga_fifa.py : Player FIFA embedding on Bundesliga league
3. epl_players.py : Player one hot embedding on EPL league
4. epl_fifa.py : Player FIFA embedding on EPL league

Allowable models:
1. lr
2. rf
3. xgb
4. svm
5. lstm
6. stacking