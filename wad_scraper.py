from bs4 import BeautifulSoup
from requests import get
import random
import envirment_utils

# Grab some extra doom WADs and save them to the wads folder

def get_header():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
        'Mozilla/5.0 (Linux; Android 11; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.72 Mobile Safari/537.36'
    ]
    user_agent = random.choice(user_agents)
    headers = {'User-Agent': user_agent}

    return headers


def get_mirror_from_page(url):
    res = get(url, headers=get_header(), verify=False)
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'html.parser')
        mirrors = soup.find("table", attrs={'class': 'download'}).find_next('ul').find_all_next('li')

        for i in mirrors:

            if i.a.text == 'Germany (SSL)':
                return i.a['href']
            else:
                continue
    else:
        return None


def get_levels():
    URLS = []


    host = 'https://www.doomworld.com/idgames/'
    link = f'{host}/levels/doom2/d-f/'
    html_doc = get(link, headers=get_header(), verify=False)

    if html_doc.status_code == 200:
        soup = BeautifulSoup(html_doc.text, 'html.parser')
        names = soup.find_all('td', attrs={'class': 'wadlisting_name'})

        for i in names:
            url_path = host + i.a['href']
            if 'levels/' in url_path:
                URLS.append(get_mirror_from_page(url_path))
    print(URLS)
    return URLS


def download_levels(levels_list, path):
    print(levels_list)
    for url in levels_list:
        res = None
        try:
            res = get(url, headers=get_header(), verify=False)

        except Exception as err:
            print(err)
            continue

        if res.status_code == 200:
            file_name = url.split('/')[-1]

            try:
                with open(f'{path}{file_name}', 'wb') as f:
                    f.write(res.content)

            except Exception as err:
                print(err)
                continue


if __name__ == '__main__':
    download_levels(get_levels(), envirment_utils.wads)