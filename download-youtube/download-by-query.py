from bs4 import BeautifulSoup 
from common import createVideoMeatadata, getNewVideoName, downloadVideo
import urllib2
import pafy
import sys
import json
import re


def getWatchUrls(query):

    query_url = "https://www.youtube.com/results?search_query=" + query
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
            watch_url = line.split('a href')[1].split('class=')[0].split('"')[1]
            watch_urls += [watch_url]
            
    return watch_urls
 

if __name__ == "__main__":

    input_str = (raw_input('$ '))
    query = '+'.join(input_str.split(' '))

    #https://www.youtube.com/results?search_query=basketball        
    watch_urls = getWatchUrls(query)
    
    # get top-10 urls
    watch_urls = [0: min(10, len(watch_urls))]

    for watch_url in watch_urls:
        video = pafy.new(watch_url) 
        downloadVideo(video) 
        break
            
