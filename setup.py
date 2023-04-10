import versioneer
from setuptools import setup, find_packages


setup(name='pyalgae_ai',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='A python library to analyze alage on reserviors using Sentinel-2 and machine learning',
      url='https://github.com/Kwater-AILab/algae_prediction.git',
      author='JiYoung Jung, HyunJun Jang, YoungDon Choi',
      author_email='choiyd1115@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'xgboost',
          'pandas',
          'matplotlib',
          'dask',
          'distributed',
          'toolz',
          'scikit-learn',
          'matplotlib',
          'joblib',
          'hydroeval',
          'geopandas',
          'folium',
          'sentinelsat',
          'seaborn'
          ],
      include_package_data=True)
