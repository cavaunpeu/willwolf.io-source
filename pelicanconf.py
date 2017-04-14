#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

import os
HOME = os.getenv('HOME')

# Site settings
AUTHOR = 'Will Wolf'
SITENAME = 'will wolf'
SITESUBTITLE = 'data science things and thoughts on the world'
SITEURL = ''
PATH = 'content'
TIMEZONE = 'America/New_York'
DEFAULT_LANG = 'en'

# Theme settings
THEME = 'theme'
STATIC_PATHS = ['figures', 'images', 'favicon.ico']

# URL settings
ARTICLE_URL = '{date:%Y}/{date:%m}/{date:%d}/{slug}/'
ARTICLE_SAVE_AS = '{date:%Y}/{date:%m}/{date:%d}/{slug}/index.html'
PAGE_URL = '{slug}/'
PAGE_SAVE_AS = '{slug}/index.html'

# Category settings
USE_FOLDER_AS_CATEGORY = False
DEFAULT_CATEGORY = 'Uncategorized'

# Page settings
ABOUT_PAGE = '/about/'
ARCHIVES_PAGE = '/archives/'

# Plugin settings
PLUGIN_PATHS = ['./plugins', os.path.join(HOME, 'repos/pelican-plugins')]
PLUGINS = ['render_math', 'disqus_static', 'ipynb.liquid']
MARKUP = ['md']

# Date formatting
DATE_FORMATS = {
    'en': '%B %-d, %Y',
}

# Comments
DISQUS_SITENAME = 'willwolf'
DISQUS_SECRET_KEY = os.getenv('DISQUS_SECRET_KEY')
DISQUS_PUBLIC_KEY = os.getenv('DISQUS_PUBLIC_KEY')

# Feed generation is usually not desired when developing
SHOW_FEED = True # this is useful for showing link during development
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Travel Blog', 'http://www.willtravellife.com'),)

# Social
GITHUB_USERNAME = 'cavaunpeu'
TWITTER_USERNAME = 'willwolf_'
LINKEDIN_USERNAME = 'williamabrwolf'
EMAIL_ADDRESS = 'williamabrwolf@gmail.com'

# Analytics
GOOGLE_ANALYTICS = 'UA-97412095-1'

# Pagination
DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
