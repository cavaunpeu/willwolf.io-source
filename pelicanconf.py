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
THEME = os.path.join(HOME, 'repos/pelican-themes/aboutwilson')
STATIC_PATHS = ['figures', 'images']

# Plugin settings
PLUGIN_PATHS = [os.path.join(HOME, 'repos/pelican-plugins')]
PLUGINS = ['render_math', 'disqus_static']

# Comments
DISQUS_SITENAME = 'willwolf'
DISQUS_SECRET_KEY = os.getenv('DISQUS_SECRET_KEY')
DISQUS_PUBLIC_KEY = os.getenv('DISQUS_PUBLIC_KEY')

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Travel Blog', 'http://www.willtravellife.com'),)

# Social
TWITTER_USERNAME = '@willwolf_'
SOCIAL = (
    ('GitHub', 'www.github.com/cavaunpeu'),
    ('Twitter', 'www.twitter.com/willwolf_'),
    ('Email', 'mailto:williamabrwolf@gmail.com'),
    ('LinkedIn', 'http://linkedin.com/in/williamabrwolf'),
)

# Categories
DISPLAY_CATEGORIES_ON_MENU = False

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
