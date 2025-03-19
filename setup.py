from setuptools import setup, find_packages

setup(
    name='plot_boundary_extractor',  # Replace with your package name
    version='0.3.2',
    author='Hansae Kim',
    author_email='kim4012@purdue.edu',
    description='Plot boundary extractor using segment anything model',
    long_description=open('README.md').read(),  # Ensure README.md exists
    long_description_content_type='text/markdown',
    packages=find_packages(),  # Automatically finds package directories
    install_requires=[
                    'd2spy',
                    'fiona',
                    'geojson',
                    'geopandas',
                    'jupyter',
                    'leafmap',
                    'numpy',
                    'opencv-python',
                    'pandas',
                    'pyproj',
                    'rasterio',
                    'scikit-image',
                    'scikit-learn',
                    'shapely',
                    'openrs-python'
                      ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10, <3.12',
)
