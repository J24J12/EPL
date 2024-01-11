import pandas as pd
from bs4 import BeautifulSoup, NavigableString
import requests
import pickle
from scipy.stats import poisson
import re
from datetime import datetime

tables = pd.read_html('https://www.premierleague.com/tables')
df_table = tables[0].iloc[0::2]
df_table.pop('Form'), df_table.pop('Unnamed: 12'), df_table.pop('Next')
df_table['Position  Pos'] = df_table['Position  Pos'].apply(lambda x: int(x.split()[0]))
df_table.rename(columns={'Position  Pos': 'Position', 'Played  Pl': 'Played', 'Won  W': 'Won', 'Drawn  D': 'Draw', 'Lost  L': 'Lost', 'GF': 'GoalsFor', 'GA':'GoalsAgainst', 'GD':'GoalDiff','Points  Pts':'Points'}, inplace=True)
df_table['Club'] = df_table['Club'].apply(lambda x: re.sub(r'\s+\b\w{3}\b$', '', x))

web = 'https://www.skysports.com/premier-league-results'
response = requests.get(web)
content = response.text
soup = BeautifulSoup(content, 'lxml')
matches = soup.find_all('div', class_='fixres__item')
score = []
home = []
away = []

for match in matches:
    score.append(match.find('span', class_='matches__teamscores').get_text())
    score_lst = [[int(num) for num in item.split() if num.isdigit()] for item in score]
    home.append(match.find('span', class_='matches__item-col matches__participant matches__participant--side1').get_text())
    home_lst = [item.strip() for item in home]
    away.append(match.find('span', class_='matches__item-col matches__participant matches__participant--side2').get_text())
    away_lst = [item.strip() for item in away]

dict_epl = {'Home': home_lst, 'Score': score_lst, 'Away': away_lst}
df_results = pd.DataFrame(dict_epl)

web = 'https://www.skysports.com/premier-league-fixtures'
response = requests.get(web)
content = response.text
soup = BeautifulSoup(content, 'lxml')

fixture_date = soup.find('div', class_='fixres__body')
date_lst = []
home = []
away = []

current_date = None
for element in fixture_date:
    if isinstance(element, NavigableString):
        continue
    if element.name == 'h4' and 'fixres__header2' in element.get('class', []):
        current_date = element.text
    elif element.name == 'div' and 'fixres__item' in element.get('class', []):
        if current_date is not None:
            date_lst.append(current_date)
            home.append(element.find('span', class_='matches__item-col matches__participant matches__participant--side1').get_text().strip())
            away.append(element.find('span', class_='matches__item-col matches__participant matches__participant--side2').get_text().strip())

dict_fix = {'date': date_lst, 'Home': home, 'Away': away}
df_fixture = pd.DataFrame(dict_fix)

current_year = datetime.now().year
current_month = datetime.now().strftime('%B')
first_half = ["January", "February", "March", "April", "May"]
second_half = ["June", "July", "August", "September", "October", "November", "December"]

def add_year(date_str):
    day, date, month = date_str.split()
    date = date.replace('st', '').replace('nd', '').replace('rd', '').replace('th', '')
    if month in second_half and current_month in second_half:
        return f"{day}, {date} {month} {current_year}"
    elif month in first_half and current_month in first_half:
        return f"{day}, {date} {month} {current_year}"
    elif month in second_half and current_month in first_half:
        return f"{day}, {date} {month} {current_year - 1}"
    elif month in first_half and current_month in second_half:
        return f"{day}, {date} {month} {current_year + 1}"
    else:
        return date_str
    
df_fixture['date'] = df_fixture['date'].apply(add_year)

df_results['HomeGoals'] = df_results['Score'].apply(lambda x: x[0])
df_results['AwayGoals'] = df_results['Score'].apply(lambda x: x[1])
df_results.drop('Score', axis=1, inplace=True)
df_results = df_results.astype({'HomeGoals': int, 'AwayGoals': int})
df_results['TotalGoals'] = df_results['HomeGoals'] + df_results['AwayGoals']
df_home = df_results[['Home', 'HomeGoals','AwayGoals']]
df_away = df_results[['Away', 'HomeGoals','AwayGoals']]

df_home = df_home.rename(columns={'Home':'Team','HomeGoals':'GoalsScored','AwayGoals':'GoalsConceded'})
df_away = df_away.rename(columns={'Home':'Team','HomeGoals':'GoalsConceded','AwayGoals':'GoalsScored'})
df_stats = pd.concat([df_home,df_away], ignore_index=True).groupby('Team').mean()

def point_prediction(home, away):
    if home in df_stats.index and away in df_stats.index:
        # 2 different lambda for home team and away team
        lamb_home = df_stats.at[home, 'GoalsScored'] * df_stats.at[away, 'GoalsConceded']
        lamb_away = df_stats.at[away, 'GoalsScored'] * df_stats.at[home, 'GoalsConceded']
        prob_home, prob_away, prob_draw = 0,0,0
        for x in range(0,11):
            for y in range(0,11):
                p = poisson.pmf(x, lamb_home) * poisson.pmf(y, lamb_away)
                # p = 0 or 1
                if x == y:
                    prob_draw += p
                elif x > y:
                    prob_home += p
                else:
                    prob_away += p
        # win = 3 points, draw = 1 point, lose = 0 point
        points_home = 3 * prob_home + prob_draw
        points_away = 3 * prob_away + prob_draw
        return (points_home, points_away)
    else:
        return(0,0)


final_table = df_table.copy()
final_table['Points'] = final_table['Points'].astype(float)
teams = final_table['Club'].values
fixtures = df_fixture[df_fixture['Home'].isin(teams)]
for index, row in fixtures.iterrows():
    home, away = row['Home'], row['Away']
    points_home, points_away = point_prediction(home,away)
    final_table.loc[final_table['Club'] == home, 'Points'] += points_home
    final_table.loc[final_table['Club'] == away, 'Points'] += points_away

final_table = final_table.sort_values('Points', ascending=False).reset_index()
final_table = final_table[['Club', 'Points']]
final_table = final_table.round(0)

final_fixture = df_fixture.copy()
final_fixture['Winner'] = '?'

def get_winner(fixture):
    for index, row in fixture.iterrows():
        home, away = row['Home'], row['Away']
        points_home, points_away = point_prediction(home, away)
        if points_home > points_away:
            winner = home
        else:
            winner = away
        fixture.loc[index, 'Winner'] = winner
    return fixture

print(get_winner(final_fixture))
print(final_table)