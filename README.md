This repository contains the source code for [my data blog](http://willwolf.io).

# Build
To build this blog, clone the repository plus submodules.

```
$ git clone git@github.com:cavaunpeu/willwolf.io-source.git
$ git submodule update --init --recursive
```

Then, build the html and serve on your local machine.

```
$ make clean
$ make html
$ make serve
$ open http://localhost:8000
```

# Deploy to GitHub Pages

```
$ make publish-to-github
```
