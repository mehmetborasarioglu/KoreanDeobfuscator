import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize Selenium
options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
options.add_argument('--headless')  # Run in headless mode for better performance

driver = webdriver.Chrome(options=options)

# URL of the encoder
encoder_url = 'https://airbnbfy.hanmesoft.com/'

# Load Korean text data from the Kowiki corpus
input_file_path = 'data/kowiki.txt'
with open(input_file_path, 'r', encoding='utf-8') as f:
    texts_to_encode = [line.strip() for line in f if line.strip()]

# Limit to 100,000 lines for encoding
texts_to_encode = texts_to_encode[:100000]

# Prepare the CSV file for output
csv_filename = 'encoded_korean_texts.csv'
with open(csv_filename, 'w', encoding='utf-8') as f:
    f.write('original_text,encoded_text\n')

# Start the encoding process
for index, text in enumerate(texts_to_encode, start=1):
    try:
        driver.get(encoder_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "textarea.form-control.my-2")))

        input_box = driver.find_element(By.CSS_SELECTOR, "textarea.form-control.my-2")
        input_box.clear()
        input_box.send_keys(text)

        driver.switch_to.window(driver.current_window_handle)

        # Find and click the encode button
        encode_button = driver.find_element(By.CSS_SELECTOR, "button.m-2.btn.btn-primary")
        if encode_button.is_displayed() and encode_button.is_enabled():
            encode_button.click()

        # Wait until the output box is populated
        output_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea.form-control[placeholder='변환된 문장들이 여기에 나타납니다.']"))
        )
        encoded_text = output_box.get_attribute('value').strip()

        # Write the data to the CSV file line-by-line
        with open(csv_filename, 'a', encoding='utf-8') as f:
            f.write(f'"{text}","{encoded_text if encoded_text else "<empty>"}"\n')

        if index % 1000 == 0:
            print(f'Encoded {index} lines...')

    except Exception as e:
        with open(f'error_page_source_{index}.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        print(f"Error encoding line {index}: {str(e)}")
        continue

driver.quit()

print("Encoding completed and data saved to 'encoded_korean_texts.csv'.")
