import requests
import pandas as pd
import time

# GitHub API setup
BASE_URL = 'https://api.github.com/search/repositories'
HEADERS = {
    'Accept': 'application/vnd.github+json',
    'User-Agent': '<github-username>',
    'Authorization': 'Bearer <github-PAT>'
}

# List of all request result repo names
repo_names = []

# Fetch 1000 repositories: 10 pages * 100 results per page
for page in range(1, 11):
    print(f"Fetching page {page}...")
    params = {
        'q': 'stars:>50', # query: fetch repositories by number of stars, only allow repositories with 50+ stars
        'sort': 'stars', # sort result by number of stars
        'order': 'desc', # sort in descending order
        'per_page': 100, # show 100 results per page
        'page': page # page number to fetch
    }
    
    response = requests.get(BASE_URL, headers=HEADERS, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch page {page}: {response.status_code}")
        break

    items = response.json().get('items', [])
    for repo in items:
        repo_names.append(repo['full_name'])
    
    time.sleep(1)  # Sleep to slow down and respect request rate limits

# List of all the request results
all_repos = []
# Fetch more detailed data about each of the top 1000 repositories
for i, full_name in enumerate(repo_names):
    if i%10==0:
        print(f"Fetching details for {full_name} ({i+1}/{len(repo_names)})")
    
    repo_url = f"https://api.github.com/repos/{full_name}"
    retries = 10
    success = False

    for attempt in range(retries):
        try:
            detail_resp = requests.get(repo_url, headers=HEADERS, timeout=10)
            if detail_resp.status_code == 200:
                all_repos.append(detail_resp.json())
                success = True
                break
            else:
                print(f"Status {detail_resp.status_code} for {full_name}")
        except requests.exceptions.ConnectTimeout:
            print(f"Timeout on attempt {attempt+1} for {full_name}")
            time.sleep(3)  # Wait before retrying

    if not success:
        print(f"Skipping {full_name} after {retries} attempts")

    time.sleep(0.1) # Sleep to slow down and respect request rate limits

# Convert to Pandas DataFrame
df = pd.DataFrame(all_repos)
# Save to CSV
filename = 'raw_github_repo_data.csv'
df.to_csv(filename, index=False)
print(f"Saved {len(df)} repositories to '{filename}'")