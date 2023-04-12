import os

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def get_version():
    """ Find the version of the package"""
    version = None
    version_file = os.path.join(BASEDIR, 'precise-lite-trainer', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if alpha and int(alpha) > 0:
        version += f"a{alpha}"
    return version


setup(
    name='precise-lite-trainer',
    version=get_version(),
    packages=['precise-lite-trainer'],
    url='https://github.com/OpenVoiceOS/precise-lite-trainer',
    license='Apache-2.0',
    include_package_data=True,
    install_requires=["sonopy==0.1.2"],
    extras_require={
        'tflite': ["tflite-runtime"],
        'full': ["tensorflow"]
    },
    author='jarbas',
    author_email='jarbasai@mailfence.com',
    description=''
)
