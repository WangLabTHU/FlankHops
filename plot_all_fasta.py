import subprocess
import os
from selenium import webdriver
import time


def main():
    data_path = "C:/Users/hello/Desktop/cache/"
    filenames = os.listdir(data_path)
    chrome_driver = r'C:/chrome_driver/chromedriver.exe'
    driver = webdriver.Chrome(executable_path=chrome_driver)
    driver.get('http://weblogo.berkeley.edu/logo.cgi')
    for file in filenames:
        if '.fasta' in file:
            png_name = file.split('.fasta')[0]
            driver.find_element_by_name("submit").send_keys(data_path + file)
            time.sleep(5)
            debug = 0


if __name__ == '__main__':
    main()