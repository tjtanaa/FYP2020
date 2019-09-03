from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import urllib.request

# Specifying incognito mode as you launch your browser[OPTIONAL]
option = webdriver.ChromeOptions()
option.add_argument("--incognito")
option.add_argument("download.default_directory=.")
# Create new Instance of Chrome in incognito mode
browser = webdriver.Chrome(executable_path='C:\\Users\\TJian\\Downloads\\chromedriver.exe', chrome_options=option)

# # Go to desired website
browser.get("https://pixabay.com/videos/search/?resolution_4k=1")

# Go to desired website
# browser.get("https://github.com/TheDancerCodes")

# Wait 20 seconds for page to load
timeout = 20
try:
    # Wait until the final element [Avatar link] is loaded.
    # Assumption: If Avatar link is loaded, the whole page would be relatively loaded because it is among
    # the last things to be loaded.
    WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='media']")))
except TimeoutException:
    print("Timed out waiting for page to load")
    browser.quit()

skip_pages = 11
for i in range(skip_pages):
    # browser.find_element_by_class_name("pure-button").click()   
    browser.find_element_by_xpath('//div[@class="media_list"]//a[@class="pure-button" and contains(text(),"Next")]').click()
    # exit()
    # Wait 20 seconds for page to load
    timeout = 20
    try:
        # Wait until the final element [Avatar link] is loaded.
        # Assumption: If Avatar link is loaded, the whole page would be relatively loaded because it is among
        # the last things to be loaded.
        WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='media_list']")))
    except TimeoutException:
        print("Timed out waiting for page to load")
        browser.quit()

titles_element = browser.find_elements_by_xpath('//div[@class="item"]//a')
url_list = []
for title in titles_element:
    url_list.append(title.get_attribute('href'))
print(url_list)

page = skip_pages + 1
# try:
while True:
    print("Scrapping page:{}".format(page))

    for i, url in enumerate(url_list):
        print(url)
        if url.find('search') != -1:
            continue
        time.sleep(1)
        # Create new Instance of Chrome in incognito mode
    #     browser = webdriver.Chrome(executable_path='C:\\Users\\TJian\\Downloads\\chromedriver.exe', chrome_options=option)
    #     browser.get("https://pixabay.com/videos/search/?resolution_4k=1")
    #     time.sleep(1)
        browser.get(url)
        
        timeout = 20
        try:
            # Wait until the final element [Avatar link] is loaded.
            # Assumption: If Avatar link is loaded, the whole page would be relatively loaded because it is among
            # the last things to be loaded.
            WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='download_menu']")))
        except TimeoutException:
            print("Timed out waiting for page to load")
            browser.quit()
        
    #     time.sleep(1)
        download = browser.find_element_by_class_name("download_menu")
        download.click()

        time.sleep(1)
        try:
            try:
                change_selection = browser.find_element_by_xpath('//div[@class="bubble se"]//table//tr[5]')
            except:
                change_selection = browser.find_element_by_xpath('//div[@class="bubble ne"]//table//tr[5]')
        except:
            try:
                change_selection = browser.find_element_by_xpath('//div[@class="bubble se"]//table//tr[4]')
            except:
                change_selection = browser.find_element_by_xpath('//div[@class="bubble ne"]//table//tr[4]')
        change_selection.click()
        
        time.sleep(1)
        browser.find_element_by_xpath('//a[@class="dl_btn pure-button button-green"]').click()
        time.sleep(1)
        browser.back()
    #     browser.quit()
    #     break

    # Wait 20 seconds for page to load
    timeout = 20
    try:
        # Wait until the final element [Avatar link] is loaded.
        # Assumption: If Avatar link is loaded, the whole page would be relatively loaded because it is among
        # the last things to be loaded.
        WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='media_list']")))
    except TimeoutException:
        print("Timed out waiting for page to load")
        browser.quit()

    browser.find_element_by_xpath('//div[@class="media_list"]//a[@class="pure-button" and contains(text(),"Next")]').click()
    url_list = []
    page+=1
    # Wait 20 seconds for page to load
    timeout = 20
    try:
        # Wait until the final element [Avatar link] is loaded.
        # Assumption: If Avatar link is loaded, the whole page would be relatively loaded because it is among
        # the last things to be loaded.
        WebDriverWait(browser, timeout).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='media']")))
    except TimeoutException:
        print("Timed out waiting for page to load")
        browser.quit()

    titles_element = browser.find_elements_by_xpath('//div[@class="item"]//a')
    url_list = []
    for title in titles_element:
        url_list.append(title.get_attribute('href'))
    print(url_list)
