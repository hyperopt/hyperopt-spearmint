#!/usr/bin/env python

descr = """A wrapper turns Spearmint optimizers into hyperopt algorithms"""

import sys
import os
import shutil
import glob

DISTNAME = 'hpspearmint'
DESCRIPTION = descr
LONG_DESCRIPTION = ""

MAINTAINER = 'James Bergstra'
MAINTAINER_EMAIL = 'anon@anon.com'
URL = 'http://jaberg.github.com/hyperopt-spearmint/'
LICENSE = 'BSD-3'
DOWNLOAD_URL = 'https://github.com/jaberg/hyperopt-spearmint/tarball/master'
VERSION = '0.0.1-dev'

import setuptools  # we are using a setuptools namespace

if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    # python 3 compatibility stuff.
    # Simplified version of scipy strategy: copy files into
    # build/py3k, and patch them using lib2to3.
    if sys.version_info[0] == 3:
        try:
            import lib2to3cache
        except ImportError:
            pass
        local_path = os.path.join(local_path, 'build', 'py3k')
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        print("Copying source tree into build/py3k for 2to3 transformation"
              "...")

        import lib2to3.main
        from io import StringIO
        print("Converting to Python3 via 2to3...")
        _old_stdout = sys.stdout
        try:
            sys.stdout = StringIO()  # supress noisy output
            res = lib2to3.main.main("lib2to3.fixes",
                                    ['-x', 'import', '-w', local_path])
        finally:
            sys.stdout = _old_stdout

        if res != 0:
            raise Exception('2to3 failed, exiting ...')

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setuptools.setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        packages=setuptools.find_packages(),
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        zip_safe=True,  # the package can run out of an .egg file
        install_requires=[
            ],
        #scripts=glob.glob(os.path.join("bin","*")),
        classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: BSD License',
            #'Programming Language :: C',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS'
            ],
    )
