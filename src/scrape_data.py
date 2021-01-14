import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import csv
import time
import re
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument('--league', default=None, type=str, help='Premier_League, Bundesliga')
parser.add_argument('--fifa', action='store_true', help='If passed, then scrape FIFA ratings')
args = parser.parse_args()

path_to_save = os.path.join("./data")

def scrape_league(league_name="Premier_League"):
	if league_name not in ["Premier_League", "Bundesliga"]:
		raise NotImplementedError("Input valid league name")

	years = []
	for i in range(2,10):
		years.append(str(2010+i)+'-'+str(10+i+1))

	if league_name=="Premier_League":
		cols = ['year','date','venue','attendance','home_team', 'home_goals']
	else:
		cols = ['year','home_team', 'home_goals']

	attrs = ['Possession %',
	'Total Shots',
	'On Target',
	'Off Target',
	'Blocked',
	'Passing %',
	'Clear-Cut Chances',
	'Corners',
	'Offsides',
	'Tackles %',
	'Aerial Duels %',
	'Saves',
	'Fouls Committed',
	'Fouls Won',
	'Yellow Cards',
	'Red Cards']

	for a in attrs:
		cols.append('home_'+a)

	cols.append('away_team')
	cols.append('away_goals')
	for a in attrs:
		cols.append('away_'+a)

	for i in range(1,7):
		cols.append('home_form_'+str(i))
	for i in range(1,7):
		cols.append('away_form_'+str(i))
	for i in range(1,12):
		cols.append('home_player_'+str(i))
	for i in range(1,12):
		cols.append('away_player_'+str(i))

	csv_file_name = league_name + '.csv'
	csv_file = open(os.path.join(path_to_save,csv_file_name),'w')
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(cols)


	print("==> Scraping " + league_name + " data.....")


	data = []
	count = 0
	not_scrape = []
	for year in years:

		if league_name=="Premier_League":
			url = 'https://www.skysports.com/premier-league-results/' + year
		else:
			url = 'https://www.skysports.com/bundesliga-results/' + year
		print(year)
		try:
		  r = requests.get(url)
		except:
		  continue
		content = r.content
		soup = BeautifulSoup(content)
		match_tags = soup.find_all('a',class_='matches__item matches__link',href=True)
		matches = [m['href'] for m in match_tags]
		for match in tqdm(matches):
			try:
				count +=1
				url = match[:-6] + 'stats/' + match[-6:]
				time.sleep(1)
				r = requests.get(url)
				content = r.content
				soup = BeautifulSoup(content)
				attrs = soup.select('.sdc-site-match-stats__name')
				teams = soup.select('.sdc-site-match-stats__team-name')

				if league_name=="Premier_League":
					date = soup.find('time',class_='sdc-site-match-header__detail-time').text
					venue = soup.find('span',class_='sdc-site-match-header__detail-venue sdc-site-match-header__detail-venue--with-seperator').text
					attendance = soup.find('span',class_='sdc-site-match-header__detail-attendance').text.split('Attendance')[-1]

				vals = [i.text for i in soup.select('.sdc-site-match-stats__val')]

				score = soup.find_all('span',class_='sdc-site-match-header__team-score-block')

				goals = soup.select('.sdc-site-last-games__score')
				goals = [i.text for i in goals]
				goals = [' '.join(i.split('\n')[1].split(' ')[-3:]) for i in goals]

				teams = soup.select('.sdc-site-match-stats__team-name')

				if league_name=="Premier_League":
					newVals = [year,date,venue,attendance,teams[0].text]
				else:
					newVals = [year,teams[0].text]

				newVals.append(score[0].text)
				for i in range(0, len(vals), 2):
					newVals.append(vals[i])
				newVals.append(teams[1].text)
				newVals.append(score[1].text)
				for i in range(1, len(vals), 2):
					newVals.append(vals[i])


				for goal in goals:
					newVals.append(goal)

				url = match[:-6] + 'teams/' + match[-6:]
				time.sleep(1)
				r = requests.get(url)
				content = r.content
				soup = BeautifulSoup(content)
				players_temp = soup.find_all('dl',class_='sdc-site-team-lineup__players')

				players = players_temp[0].select('.sdc-site-team-lineup__player-name')

				for i in players[0:11]:
					newVals.append(' '.join((i.text.split("\n")[1:3])))

				players = players_temp[2].select('.sdc-site-team-lineup__player-name')

				for i in players[0:11]:
					newVals.append(' '.join((i.text.split("\n")[1:3])))

				if league_name=="Premier_League":
					data.append(newVals)
					if([year,match] in not_scrape):
					  not_scrape.remove([year,match])

				if(count%10==0):
					print('Writing to CSV')
					df = pd.DataFrame(data,columns=cols)
					df.to_csv(os.path.join(path_to_save,csv_file_name))
			except:

				if league_name=="Premier_League":
					time.sleep(10)
					count -=1
					print(match)
					
					if([year,match] not in not_scrape):
					  matches.append(match)
					  not_scrape.append([year,match])
					else:
					  continue
				else:
					not_scrape.append(match)
				continue
					   

	if league_name=="Premier_League":
		temp = 'Attendance: Attendance22,859.'
		temp.split('Attendance')[-1]   


	matches = not_scrape
	for m in matches:
		try:

			if league_name=="Premier_League":
				year = m[0]
				match = m[1]
			else:
				match = m

			url = match[:-6] + 'stats/' + match[-6:]
			time.sleep(1)
			r = requests.get(url)
			content = r.content
			soup = BeautifulSoup(content)
			attrs = soup.select('.sdc-site-match-stats__name')
			teams = soup.select('.sdc-site-match-stats__team-name')

			vals = [i.text for i in soup.select('.sdc-site-match-stats__val')]

			score = soup.find_all('span',class_='sdc-site-match-header__team-score-block')

			goals = soup.select('.sdc-site-last-games__score')
			goals = [i.text for i in goals]
			goals = [' '.join(i.split('\n')[1].split(' ')[-3:]) for i in goals]

			teams = soup.select('.sdc-site-match-stats__team-name')

			if league_name=="Premier_League":	
				newVals = [year,date,venue,attendance,teams[0].text]
			else:
				newVals = [year,teams[0].text]

			newVals.append(score[0].text)
			for i in range(0, len(vals), 2):
				newVals.append(vals[i])
			newVals.append(teams[1].text)

			newVals.append(score[1].text)
			for i in range(1, len(vals), 2):
				newVals.append(vals[i])

			for goal in goals:
				newVals.append(goal)

			url = match[:-6] + 'teams/' + match[-6:]
			time.sleep(1)
			r = requests.get(url)
			content = r.content
			soup = BeautifulSoup(content)

			players_temp = soup.find_all('dl',class_='sdc-site-team-lineup__players')

			players = players_temp[0].select('.sdc-site-team-lineup__player-name')

			for i in players[0:11]:
				newVals.append(' '.join((i.text.split("\n")[1:3])))

			players = players_temp[2].select('.sdc-site-team-lineup__player-name')

			for i in players[0:11]:
				newVals.append(' '.join((i.text.split("\n")[1:3])))

			data.append(newVals)

		except:
			continue



	df = pd.DataFrame(data,columns=cols)

	df.to_csv(os.path.join(path_to_save,csv_file_name))



def scrape_fifa():


	#Get the player information from the players page for Each year


	#Base URLs for All the years starting from 14 to 21


	#base_url = "https://sofifa.com/players?offset="

	base_urls = {"14":"https://sofifa.com/players?r=140052&set=true&offset=",
				"15":"https://sofifa.com/players?r=150059&set=true&offset=",
				"16":"https://sofifa.com/players?r=170099&set=true&offset=",
				"17":"https://sofifa.com/players?r=180084&set=true&offset=",
				"18":"https://sofifa.com/players?r=180084&set=true&offset=",
				"19":"https://sofifa.com/players?r=190075&set=true&offset=",
				"20":"https://sofifa.com/players?r=200061&set=true&offset=",
				"21":"https://sofifa.com/players?r=210011&set=true&offset="}

	yrs = ["14","15","16","17","18","19","20","21"]


	for yr in yrs:
		#Main columns that we want
		base_url = base_urls[yr]

		columns = ['ID', 'Name', 'Age', 'Photo', 'Nationality', 'Flag', 'Overall', 'Potential', 'Club', 'Club Logo', 'Value', 'Wage', 'Special']
		data = pd.DataFrame(columns = columns)

		for offset in range(0, 300):
			url = base_url + str(offset * 61)
			source_code = requests.get(url)
			plain_text = source_code.text
			soup = BeautifulSoup(plain_text, 'html.parser')
			table_body = soup.find('tbody')
			for row in table_body.findAll('tr'):
				td = row.findAll('td')
				picture = td[0].find('img').get('data-src')
				pid = td[0].find('img').get('id')
				nationality = td[1].find('a').get('title')
				flag_img = td[1].find('img').get('data-src')
				#site changed lol. this doesn't work now. 
				#name = td[1].findAll('a')[1].text
				name = td[1].findAll('a')[0]['data-tooltip']
				#age = td[2].find('div').text.strip()
				age = td[2].text.strip()
				overall = td[3].text.strip()
				potential = td[4].text.strip()
				club = td[5].find('a').text
				club_logo = td[5].find('img').get('data-src')
				value = td[6].text.strip()
				wage = td[7].text.strip()
				special = td[8].text.strip()
				player_data = pd.DataFrame([[pid, name, age, picture, nationality, flag_img, overall, potential, club, club_logo, value, wage, special]])
				player_data.columns = columns
				data = data.append(player_data, ignore_index=True)
			print(offset, " pages done and table size ", data.shape)
			print(player_data)
			

		data = data.drop_duplicates()
		data.to_csv(os.path.join(path_to_save,'players_'+yr+'.csv'), encoding='utf-8-sig')


if __name__ == '__main__':

	scrape_league(league_name=args.league)
	if args.fifa:
		scrape_fifa()
