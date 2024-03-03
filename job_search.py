import requests
from bs4 import BeautifulSoup

def fetch_jobs(page_number=0):
    base_url = "https://delaware.craigslist.org/search/jjj"
    params = {
        's': page_number * 120  # Craigslist lists about 120 jobs per page; adjust if needed
    }
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    response = requests.get(base_url, params=params, headers=headers)
    # print(response.text)
    if response.status_code != 200:
        print("Failed to fetch job listings")
        return ""
    return response.text

def parse_job_listings(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    job_listings = soup.find_all('li', class_='cl-static-search-result')
    jobs = []
    for listing in job_listings:
        job_info = listing.find('a')
        if job_info:
            title = job_info.find('div', class_='title').text.strip()
            link = job_info['href']
            location = job_info.find('div', class_='location').text.strip() if job_info.find('div', class_='location') else 'Location not specified'
            # Assume the date might not be present or parse a specific element for the date
            date = 'Date not specified'  # Placeholder; adjust based on actual date parsing logic
            jobs.append({'title': title, 'link': link, 'location': location, 'date': date})
    return jobs


def search_jobs():
    jobs = []
    for page_number in range(5):  # Iterate through the first 5 pages
        html_content = fetch_jobs(page_number)
        if not html_content:
            print("Failed to fetch HTML content or no more jobs to fetch.")
            break
        new_jobs = parse_job_listings(html_content)
        if not new_jobs:
            print("No more jobs found.")
            break
        jobs.extend(new_jobs)
        print(f"Fetched {len(new_jobs)} jobs from page {page_number + 1}.")

    with open('jobs.txt', 'w') as file:  # Open file in write mode
        for job in jobs:  # Writing each job to the file, excluding date
            file.write(f"Title: {job['title']}, Link: {job['link']}, Location: {job['location']}\n")

# Example usage
search_jobs()
