import versioneer
from setuptools import setup, find_packages


setup(name='pyalgae_ai',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='A python library to predict alage using machine learning',
      url='https://github.com/Kwater-AILab/algae_prediction.git',
      author='JiYoung Jung, YoungDon Choi',
      author_email='choiyd1115@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'tensorflow',
          'xgboost',
          'pandas',
          'matplotlib',
          'dask',
          'distributed',
          'toolz',
          'sklearn',
          'matplotlib',
          'joblib',
          'hydroeval'
          ],
      include_package_data=True)