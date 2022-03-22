import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    

setuptools.setup(
    name='examl',
    version='0.1',
    author="Valéry N'ZI",
    author_email='valenother@gmail.com',
    description='ml lib',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=['examl'],
    install_requires=['pandas', 'openpyxl'],
)