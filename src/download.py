from bs4 import BeautifulSoup
import requests
from clint.textui import progress

links = []
chunk_size = 16

print("Enter 'Index Of' URL")
# 'http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/images/'

url = input()
print("Enter The Path To Save Files")
path = input()

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

for link in (soup.find_all('a')):
    links.append(link['href'])
    
print(len(links))

for link in links:
    req = requests.get(url + '/' + link, stream = True)
    try:
        with open(path + link, "wb") as file:
            length = int(req.headers.get('content-length'))
            for chunk in progress.bar(req.iter_content(chunk_size), expected_size = (length/chunk_size), label = link + "  "):
                if chunk:
                    file.write(chunk)
            print("Download Successful\n")
            
    except OSError as e:
        continue