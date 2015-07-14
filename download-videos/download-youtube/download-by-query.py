from bs4 import BeautifulSoup 
from common import createVideoMeatadata, getNewVideoName, downloadVideo
import socket
import urllib2
import urllib
import pafy
import sys
reload(sys)
sys.setdefaultencoding("utf8")
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
        if line.find('><div class="yt-lockup-content"><h3 class="yt-lockup-title"><a href="/watch?') >= 0:
            print line
            watch_url = line.split('a href')[1].split('class=')[0].split('"')[1].split('v=')[1]
            watch_urls += [watch_url]
        
     
    return watch_urls
 


def getWatchUrls(query, max_number=10, max_length = 300):

    # https://www.youtube.com/results?filters=creativecommons&search_query=dog+playing&page=1
    #query_prefix = 'https://www.youtube.com/results?search_query=' + urllib.quote_plus(query) + '&filters=creativecommons&page='
    query_prefix = 'https://www.youtube.com/results?search_query=' + urllib.quote_plus(query) + '&page='
    watch_urls = []
    print query_prefix
    page_number = 1
    while len(watch_urls) < max_number:
        
        query_url = query_prefix + str(page_number)
        page_urls = parseWebpage(query_url)
        for watch_url in page_urls:
            video = pafy.new(watch_url) 
            if video.length < max_length:# secs            
                watch_urls += [watch_url]
        print page_number, page_urls, len(watch_urls)
        page_number += 1
    
    return watch_urls[:max_number]
    

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'Usage:', sys.argv[0], ' test_mode'
        exit(-1)

    test_mode = int(sys.argv[1])
     
    input_str = (raw_input('$ '))
   
    if test_mode == 0:
        with open('queries', 'a') as fh:
            fh.write(input_str + '\n')
    
    watch_urls = getWatchUrls(input_str, 30)
    
    for watch_url in watch_urls:
        video = pafy.new(watch_url) 
        video_info = downloadVideo(video)
        
        if test_mode == 0:
            # write download log      
            with open('log', 'a') as fh:
                output_str = input_str
                for key in video_info:
                    if key == 'description' or key == 'author' or key == 'keywords':
                        continue
                    if type(video_info[key]) == list:
                        output_str += '\t' + ','.join(video_info[key])
                    else:
                        output_str += '\t' + str(video_info[key])
                fh.write(output_str + '\n')
