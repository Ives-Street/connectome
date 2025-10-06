# Connectome

### License

Connectome is NOT FREE FOR COMMERCIAL USE. If you wish to use Connectome for a commercial purpose, you MUST contact dtr@ives.st for a licesne or you will be legally responsible. Commercial purposes include any activity by a for-profit entity, such as consulting services, real estate investment analysis, or the planning of private transportation services. 

Connectome is freely available for NON-COMMERCIAL USE ONLY. It is free for use by governments, nonprofits, academic groups, and individual researchers. As such, the codebase is available here as a "source-available" rather than a true "open-source" license.

### About

The connectome model is a tool to measure general access-to-destinations in an urban area. By "general" access-to-destinations, we mean that the connectome tool includes all modes, all places, and all people. Most conventional approaches to access-to-destiations used in transportation planning practice are "specific", meaning that they only measure access to a specific set of destinations (often jobs) by a specific mode (often public transport). Because the connectome approach is general, we hope it will be more useful for evaluating transport interventions, such as road pricing or road space reallocations, that affect multiple different modes of transport. This approach may have applications for high-level evaluations of proposed transportation investments, or as a supplement to conventional travel demand models.

The code in this repository is an in-development pre-alpha refactor by [Ives Street](www.ives.st). Previous commits include the older version, which was used for a [study](https://ggwash.org/files/202501_getting-there-with-congestion-pricing.pdf) of potential congestion pricing in Washington, DC. Unfortunately, that version has several problems. It is slow and difficult to use, it has depreciated / obsolete dependencies, and the code is poorly structured and difficult to extend. For those reasons, we have decided to refactor the code.

![dcimpacts](https://ggwash.org/images/posts/_resized/taylor-roadpricing3_2.png)

We take the term connectome from neuroscience, where it refers to the entire network of all the connections between the hundreds of billions of neurons in a brain, much as the genome refers to the set of all genetic material in a cell. Our goal is to measure the full network of connections between people and places in an urban area.

This connectome implementation of general access-to-destinations is licensed under Apache and will remain open source. 

### Installation

We have not yet configured a package for Connetcome, we will do so when we complete a first alpha release. For now, you must clone this repository and run the python files directly. I use these parameters to establish a Conda environment with all dependencies:
~~~
conda create -n cxome -c conda-forge --strict-channel-priority gdal osmnx geopandas rasterstats spyder tqdm openpyxl census folium matplotlib geoplot pysal overturemaps geojson r5py lonboard
pip install pygris osmium mobility-db-api
apt install osmium-tool osmctools openjdk-21-jdk-headless
~~~
### Usage

We include a test_run.py script that allows easy execution of all complete sections of the refactor.

### Contributions

We welcome contributions whether through code, review, crticism, or theory. Because this is a relatively small project, we are currently handling issue management and roadmap internally to Ives Street, but with external interest we can used shared/open process management. Please contact [Taylor Reich](mailto:dtr@ives.st) to get involved.
