#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

import os

# Site settings
AUTHOR = 'Will Wolf'
SITENAME = 'will wolf'
SITESUBTITLE = 'writings on life, geopolitics, machine learning'
SITEURL = ''
PATH = 'content'
TIMEZONE = 'America/New_York'
DEFAULT_LANG = 'en'

# Theme settings
THEME = './theme/'

# Static path settings
STATIC_PATHS = ['figures', 'images', 'favicon.ico', 'CNAME']

# URL settings
ARTICLE_URL = '{date:%Y}/{date:%m}/{date:%d}/{slug}/'
ARTICLE_SAVE_AS = '{date:%Y}/{date:%m}/{date:%d}/{slug}/index.html'
PAGE_URL = '{slug}/'
PAGE_SAVE_AS = '{slug}/index.html'

# Category settings
USE_FOLDER_AS_CATEGORY = True
CATEGORY_URL = '{slug}/'
CATEGORY_SAVE_AS = '{slug}/index.html'

# Page settings
ABOUT_PAGE = 'about/'
ABOUT_PAGE_LINK_TEXT = 'About'

ARCHIVE_PAGE = 'archive/'
ARCHIVE_PAGE_LINK_TEXT = 'Archive'

CV_PAGE = 'cv/'
CV_PAGE_LINK_TEXT = 'Résumé'

BOOKS_PAGE = 'books/'
BOOKS_PAGE_LINK_TEXT = 'Books'

# Plugin settings
PLUGIN_PATHS = ['./plugins', './plugins/pelican-plugins']
PLUGINS = [
    'render_math',
    'disqus_static',
    'ipynb.liquid',
    'i18n_subsites',
    'bootstrapify'
]
MARKUP = ['md']

# Date formatting
DATE_FORMATS = {
    'en': '%B %-d, %Y',
}

# Multilanguage
DEFAULT_LANG = 'en'
I18N_UNTRANSLATED_ARTICLES = 'remove'
I18N_SUBSITES = {
    'es': {
        'SITESUBTITLE': 'escritura sobre la vida, la geopolítica, y machine learning',
        'AVATAR': '../images/will.jpg'
    }
}
language_name_lookup = {
    'en': 'English',
    'es': 'Español',
}

def lookup_lang_name(lang_code):
    return language_name_lookup[lang_code]

JINJA_FILTERS = {
    'lookup_lang_name': lookup_lang_name,
}

# Comments
DISQUS_SITENAME = 'willwolf'
DISQUS_SECRET_KEY = os.getenv('DISQUS_SECRET_KEY')
DISQUS_PUBLIC_KEY = os.getenv('DISQUS_PUBLIC_KEY')

# Feed generation is usually not desired when developing
SHOW_FEED = False
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Social
GITHUB_USERNAME = 'cavaunpeu'
TWITTER_USERNAME = 'willwolf_'
LINKEDIN_USERNAME = 'williamabrwolf'
EMAIL_ADDRESS = 'williamabrwolf@gmail.com'

# Twitter cards
TWITTER_CARDS = True
AVATAR = 'images/will.jpg'

# Analytics
GOOGLE_ANALYTICS = os.getenv('GOOGLE_ANALYTICS_TRACKING_ID')

# Pagination
DEFAULT_PAGINATION = 10

# License
LICENSE = 'MIT'

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
