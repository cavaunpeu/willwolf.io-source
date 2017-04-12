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
PLUGINS = ['render_math']

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Pelican', 'http://getpelican.com/'),
         ('Python.org', 'http://python.org/'),
         ('Jinja2', 'http://jinja.pocoo.org/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
