from setuptools import setup, find_packages

setup(
  name = 'multihead-latent-attention',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'Multi-head Latent Attention (MLA) - PyTorch',
  long_description_content_type = 'text/markdown',
  author = 'Farhan Mohammed',
  author_email = 'mfa200312@gmail.com',
  url = 'https://github.com/Nemesis-12/multihead-latent-attention',
  keywords = [
    'artificial intelligence',
    'attention',
    'attention mechanism',
    'deep learning',
    'natural language processing',
    'pytorch',
    'transformer'   
  ],
  install_requires=[
    'rotary-embedding-torch',
    'torch'
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13'
  ],
)