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
THEME = './theme/'
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
ABOUT_PAGE_LINK_TEXT = 'About'

ARCHIVE_PAGE = '/archives/'
ARCHIVE_PAGE_LINK_TEXT = 'Archive'

# Plugin settings
PLUGIN_PATHS = ['./plugins', './plugins/pelican-plugins']
PLUGINS = ['render_math', 'disqus_static', 'ipynb.liquid', 'i18n_subsites']
MARKUP = ['md']

# Date formatting
DATE_FORMATS = {
    'en': '%B %-d, %Y',
}

# Multilanguage
DEFAULT_LANG = 'en'
I18N_SUBSITES = {
    'es': {
        'SITESUBTITLE': 'cosas de data science y pensamientos sobre el mundo',
        'ABOUT_PAGE_LINK_TEXT': 'Acerca de',
        'ARCHIVE_PAGE_LINK_TEXT': 'Archivo',
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
SHOW_FEED = True # this is useful for showing link during development
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

# Analytics
GOOGLE_ANALYTICS = 'UA-97412095-1'

# Pagination
DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
