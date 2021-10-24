from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as p
import os.path
import getpass
import time

browser = webdriver.Firefox()

login_url = 'https://www.strava.com/login'
browser.get(login_url)
time.sleep(0.3)

while browser.current_url == login_url:
    username = input('Your email address: ')
    pwd = getpass.getpass('Your password: ')
    
    user_box = browser.find_element_by_id('email')
    user_box.send_keys(username)
    pwd_box = browser.find_element_by_id('password')
    pwd_box.send_keys(pwd)
    browser.find_element_by_id('login-button').click()

month = '202103'
url = 'https://www.strava.com/athletes/57257433#interval_type?chart_type=miles&interval_type=month&interval={}&year_offset=0'.format(month)
browser.get(url)
time.sleep(0.3)

stats = browser.find_elements_by_xpath('//div[@class="Stat--stat-value--3bMEZ "]')
test_run = browser.find_elements_by_xpath('//span[.="Pace"]//following')

for stat in stats:
    print(stat.text)

print('test')
#test-comment

for test in test_run:
    print(test.text)

#print(browser.find_elements_by_class_name('ActivityEntryBody--media--aeJhN'))
#browser.quit()
