import csv, requests, urlparse, urllib, sys, threading
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()

URL_FILE = "typepad_blogs.txt"
STARTING_URL = 'http://betterthanieverexpected.blogspot.com/'
CSV_NAME = 'TGB_Propagation_typepad.csv'
INCOMPLETE_FILENAME = 'incomplete.txt'
BLOGSPOT = 1
WORDPRESS = 2
TYPEPAD = 3
KEYWORDS = ['TGB Substitution Test', 'TGB Bias Test', 'Bias Test', 'Substitution Test', 'Time Goes By', 'Time Goes By Substitution Test', 'Ronni', 'Time Goes By Substitute Test', 'Time Goes By Test']
propagations_fname = "propagations.csv"

#Grabs title of each blog post
def find_titles(soup):
	if platform == TYPEPAD:
		return soup.select('.entry-header') #typepad
	if platform == BLOGSPOT:
		titles = soup.select('h3.post-title.entry-title') #blogspot
		if len(titles) == 0:
			titles = soup.select('h3.posttitle')
		if len(titles) == 0:
			titles = soup.select('h3.posttitle a')
		if len(titles) == 0:
			titles = soup.select('.post-title')
		if len(titles) == 0:
			titles = soup.select('h1.title.entry-title a')
		return titles
	if platform == WORDPRESS:
		titles = soup.select('.entry-title') #wordpress
		if len(titles) == 0:
			titles = soup.select('.post-title')
		if len(titles) == 0:
			titles = soup.select('h2.posttitle a')
		if len(titles) == 0:
			titles = soup.select('h2 a') #####
		return titles
	return None

#Grabs the text content of each blog post
def find_contents(soup):
	if platform == TYPEPAD:
		contents = soup.select('.entry-body') #typepad
	if platform == BLOGSPOT:
		contents = soup.select('div.post-body.entry-content') #blogspot
		if len(contents) == 0:
			contents = soup.select('div.post')
			if len(contents) == 0:
				contents = soup.select('div.post-body')
			if len(contents) == 0:
				contents = soup.select('div.article-content.entry-content')
	if platform == WORDPRESS:
		contents = soup.select('.entry-content') #wordpress
		if len(contents) == 0:
			contents = soup.select('div.entry')
		if len (contents) == 0:
			contents = soup.select('.posttitle p')
	if contains_keywords(contents):
		return contents
	return None

#Checks for the presence of a keyword
def contains_keywords(contents):
	for word in KEYWORDS:
		if (word in str(contents)) or (word.lower() in str(contents)):
			return True
	return False

#Grabs date timestamp of the blog post
def find_dates(soup):
	if platform == TYPEPAD:
		dates = soup.select('.date-header') #blogspot, typepad #'date-header'
		if len(dates) < 1:
			dates = soup.select('.post-footers')
		return dates
	if platform == BLOGSPOT:
		dates = soup.select('.date-header')
		if len(dates) == 0:
			dates = soup.select('div.post h3.post-title')
		if len(dates) == 0:
			dates = soup.select('h2.date-header')
		if len(dates) == 0:
			dates = soup.select('.abbr.time.published')
		return dates
	if platform == WORDPRESS:
		dates = soup.select('.entry-date') #wordpress div.date a
		if len(dates) == 0:
			dates = soup.select('div.date')
		if len(dates) == 0:
			dates = soup.select('p.commentmeta')
		if len(dates) == 0:
			dates = soup.select('p.postmetadata')
		if len(dates) == 0:
			dates = soup.select('.small')
		return dates
	return None

#Grabs links on the webpage to navigate to the previous/next post
def find_refs(soup):
	if platform == WORDPRESS:
		refs = soup.select('a.next.page-numbers') #wordpress #'.nav-previous a'
		if len(refs) == 0:
			refs = soup.select('p a')
		if len(refs) == 0:
			refs = soup.select('.nav-previous a')
		return refs
	if platform == BLOGSPOT:
		return soup.select('.blog-pager-older-link') #blogspot
	if platform == TYPEPAD:
		#refs = soup.select('.pager-right a') #typepad
		#if len(refs) == 0:
		refs = soup.select('div.pager-inner a')
		return refs
	return None

#Checks to see if all data fields have been found
def dataComplete(titles, contents, dates):
	if titles is None and contents is None and dates is None:
		return False
	elif contents is None:
		return False
	else:
		return True

#Begins scraping each blog site
def crawl(url, wr, output_file):
	if (url == None):
		sem.release()
		return
	r = None
	try:
		r = requests.get(url)
	except requests.exceptions.Timeout:
		print "timeout"
		return
	except requests.exceptions.TooManyRedirects:
		print "redirects"
		return
	except requests.exceptions.RequestException as e:
		print "except"
		return

	post_soup = BeautifulSoup(r.text, "html.parser")

	titles = find_titles(post_soup)
	contents = find_contents(post_soup)
	dates = find_dates(post_soup)
	next_page_refs = find_refs(post_soup)
	
	#if not dataComplete(titles, contents, dates):
		#incomplete.write(url)
		#print "incomplete"
		#return

	if contents == None: return

	for i in range(0, len(titles)):
		content = None
		date = None
		title = titles[i].get_text().encode("utf-8")
		if i < len(contents):
			content = contents[i].get_text().encode("utf-8")
		if i < len(dates):
			date = dates[i].get_text().encode("utf-8")
		wr.writerow([url, title, date, content])

	
	if len(next_page_refs) is 0:
		crawl(None, None, None)
	else:
		next_page_link = next_page_refs[-1].attrs['href']
		crawl(next_page_link, wr, output_file)

#Spins up a thread to crawl an individual blog
def startup_crawl(url):
	sem.acquire()
	print url
	start_sub = 7 #after "http://"
	end_sub = url.index('.')
	#fname = "_" + url[start_sub:end_sub] + ".csv"
	#count_start = count_start + 1
	output_file = open(propagations_fname, 'wb')
	wr = csv.writer(output_file, dialect='excel')
	crawl(url, wr, output_file)
	output_file.close()

#output_file = open(CSV_NAME, 'wb')
#incomplete = open(INCOMPLETE_FILENAME, 'wb')
#wr = csv.writer(output_file, dialect='excel')
#crawl(STARTING_URL)
#URL = sys.argv[1]
platform = TYPEPAD
count_start = 0
sem = threading.BoundedSemaphore(1)


with open(URL_FILE) as f:
 	content = f.readlines()

 	for i in range(0, len(content)):
 		content[i] = content[i].strip()

 	for i in range(0, len(content)):
 		url = content[i]
 		if not url == "":
 			t = threading.Thread(target=startup_crawl, args=(url,))
 			t.daemon = False
 			t.start()
