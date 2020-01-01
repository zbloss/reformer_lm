import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='reformer_lm',  
     version='1.0.1',
     scripts=[] ,
     author="Zachary Bloss",
     author_email="zacharybloss@gmail.com",
     description="a Pytorch implementation of the Reformer network (https://openreview.net/forum?id=rkgNKkHtvB)",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/zbloss/reformer_lm",
     install_requires=[
         "numpy>=1.18.0",
         "torch>=1.3"
     ],
     packages=['reformer_lm'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     keywords=['Reformer', 'ReverseNetwork', 'Efficient Transformer']
 )
