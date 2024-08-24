import urllib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import os

driver = webdriver.Firefox()
driver.get('https://donjon.bin.sh/d20/dungeon/')
start = 0
target_folder = './dungeons'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

for i in range(1000):
    btn = driver.find_element(By.XPATH,'/html/body/div[2]/div[2]/div/div/div[1]/div[2]/div/input')
    btn.click()

    select = Select(driver.find_element(By.ID,'input-dungeon_layout'))
    select.select_by_value('Square')
    
    img = driver.find_element(By.XPATH,'/html/body/div[2]/div[2]/div/div/div[1]/div[3]/div[2]/div/p/img')
    src = img.get_attribute('src')
    urllib.request.urlretrieve(src, "./dungeons/dungeon_{}.png".format(i+start))
    print("Map no. " + str(i+start) + " saved")
    



