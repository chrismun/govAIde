import requests
from bs4 import BeautifulSoup

def fetch_apartments(page_number=0):
    base_url = "https://delaware.craigslist.org/search/apa"
    params = {
        's': page_number * 120  # Adjust based on listings per page
    }
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code != 200:
        print("Failed to fetch apartment listings")
        return ""
    return response.text

def parse_apartment_listings(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    apartment_listings = soup.find_all('li', class_='cl-static-search-result')
    apartments = []
    for listing in apartment_listings:
        title_element = listing.find('div', class_='title')
        price_element = listing.find('div', class_='price')
        location_element = listing.find('div', class_='location')
        link_element = listing.find('a')

        title = title_element.text.strip() if title_element else 'Title not specified'
        price = price_element.text.strip() if price_element else 'Price not specified'
        location = location_element.text.strip() if location_element else 'Location not specified'
        link = link_element['href'] if link_element else 'Link not specified'

        apartments.append({'title': title, 'price': price, 'location': location, 'link': link})
    return apartments

def search_apartments():
    apartments = []
    for page_number in range(5):  # Adjust the range as needed
        html_content = fetch_apartments(page_number)
        if not html_content:
            print("Failed to fetch HTML content or no more apartments to fetch.")
            break
        new_apartments = parse_apartment_listings(html_content)
        if not new_apartments:
            print("No more apartments found.")
            break
        apartments.extend(new_apartments)
        print(f"Fetched {len(new_apartments)} apartments from page {page_number + 1}.")

    with open('apartments.txt', 'w') as file:
        for apartment in apartments:
            file.write(f"Title: {apartment['title']}, Price: {apartment['price']}, Location: {apartment['location']}, Link: {apartment['link']}\n")

# Example usage
search_apartments()
