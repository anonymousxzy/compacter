import setuptools


def setup_package():
  long_description = "seq2seq"
  setuptools.setup(
      name='seq2seq',
      version='0.0.1',
      description='Anonymous submission',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Anonymous',
      license='Anonymous',
      packages=setuptools.find_packages(
          exclude=['docs', 'tests', 'scripts', 'examples']),
      install_requires=[
        'importlib-metadata<4',
        'click<8.0,>=7.0',
        'tqdm<4.50.0,>=4.27',
        'tensorboard',
        'scikit-learn',
        'seqeval',
        'psutil',
        'sacrebleu',
        'rouge-score',
        'matplotlib',
        'git-python', 
        'streamlit',
        'elasticsearch',
        'nltk',
        'pandas',
        'datasets',
        'fire',
        'pytest',
        'conllu',
        'sentencepiece',
        'transformers==4.6.0',
        'tabulate',
        'fairscale',
      ],
      classifiers=[],
      keywords='text nlp machinelearning',
  )


if __name__ == '__main__':
  setup_package()
