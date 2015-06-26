from bs4 import BeautifulSoup 
from common import createVideoMeatadata, getNewVideoName, downloadVideo
import socket
import urllib2
import urllib
import pafy
import sys
import json
import re

def parseWebpage(query_url):

    watch_urls = []
    try:
        html_doc = urllib2.urlopen(query_url).readlines()

    except urllib2.URLError, e:
        # For Python 2.7
        print 'URLError %r' % e
        exit(-1)

    except socket.timeout, e:
        # For Python 2.7
        print 'Timeout %r' % e
        exit(-1)

    for line in html_doc:
        if line.find('</div><div class="yt-lockup-content">') >= 0:
            watch_url = line.split('a href')[1].split('class=')[0].split('"')[1].split('v=')[1]
            watch_urls += [watch_url]
        
     
    return watch_urls
 


def getWatchUrls(query, max_number=50):

    # https://www.youtube.com/results?filters=creativecommons&search_query=dog+playing&page=1
    query_prefix = "https://www.youtube.com/results?search_query=" + urllib.quote_plus(query) + "&filters=creativecommons&page="
    watch_urls = []

    page_number = 1
    while len(watch_urls) < max_number:
        
        query_url = query_prefix + str(page_number)
        page_urls = parseWebpage(query_url)
        watch_urls.extend(page_urls)
        print page_number, page_urls, len(watch_urls)
        page_number += 1
    
    return watch_urls[:max_number]
    
if __name__ == "__main__":

    input_str = (raw_input('$ '))

    watch_urls = getWatchUrls(input_str, 1)
    
    for watch_url in watch_urls:
        video = pafy.new(watch_url) 
        downloadVideo(video) 
