import os

import requests
import xmltodict
from bs4 import BeautifulSoup

# lists
urls = set()


# function created
def scrape(site, depth):
  print(site)
  print(depth)
  if depth > 1:
    return

  # getting the request from url
  r = requests.get(site)

  # converting the text
  s = BeautifulSoup(r.text, "html.parser")

  sidebar = s.select('aside')
  if len(sidebar) == 0:
    return

  # for i in s.find_all("a"):
  for i in sidebar[0].find_all('a'):

    if 'href' not in i.attrs:
      continue
    href = i.attrs['href']

    if href.startswith("/") and href != '/':
      print("XXXXXXXXXXXXXx")
      print(site)
      print(href)

      href = href[1:]
      site = os.path.join(site, href)
      print(site)

      if site not in urls:
        urls.add(site)
        print(site)
        # calling it self
        scrape(site, depth + 1)


# main function
if __name__ == "__main__":

  # # website to be scrape
  # site = "https://docs.clarifai.com/"

  # # calling function
  # scrape(site, depth=0)

  # # getting the request from url
  # r = requests.get(site)

  # # converting the text
  # s = BeautifulSoup(r.text, "html.parser")

  # sidebar = s.select('aside')

  url = "https://docs.clarifai.com/sitemap.xml"
  res = requests.get(url)
  raw = xmltodict.parse(res.text)

  urls = raw['urlset']['url']

  for url in urls:
    loc = url['loc']
    if loc.find('changelog') > 0:
      continue
    print(loc)
    r = requests.get(loc)
    s = BeautifulSoup(r.text, "html.parser")

    mds = s.find_all("div", {"class": "markdown"})
    for md in mds:
      # if loc.find("pagination") > 0:
      #   import pdb
      #   pdb.set_trace()
      t = md.text
      if str(md).find("tabs") > 0:
        import pdb
        pdb.set_trace()

  # from bs4 import BeautifulSoup
  # import requests

  # xmlDict = {}

  # r = requests.get("https://docs.clarifai.com/sitemap.xml")
  # xml = r.text

  # soup = BeautifulSoup(xml)
  # sitemapTags = soup.find_all("sitemap")

  # print("The number of sitemaps are {0}".format(len(sitemapTags)))

  # for sitemap in sitemapTags:
  #   xmlDict[sitemap.findNext("loc").text] = sitemap.findNext("lastmod").text

  # print(xmlDict)
