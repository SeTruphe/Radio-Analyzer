from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='radio-analyzer',
    version='0.1',
    author='Albert Lattke',
    author_email='albert.lattke@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SeTruphe/Radio-Analyzer',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=[
        'whisper',
        'openai-whisper',
        'pydub',
        'transformers >= 4.33',
        'torch',
        'ffmpeg',
        'evaluate',
        'datasets',
        'numpy',
        'tokenizers',
        'scipy',
        'gradio',
        'noisereduce',
        'geopy',
        'plotly'
    ],
    entry_points={
        'console_scripts': [
            'radio-analyzer=radio_analyzer.visualiser:run_radio_analyzer',
        ],
    },
)