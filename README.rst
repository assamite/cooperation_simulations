Cooperation Simulations
=======================

The code is developed with Python 3.6+.

1. Clone the repo
2. Create a virtual environment in your cloned repo's root using: ``python3.6 -m venv env``
3. Activate virtualenv using: ``source env/bin/activate``
4. Install general requirements: ``pip install -r requirements.txt``
5. Run the first simulation test: ``python first_sim.py``

The first simulation just uses multiple cores on the same node, each core having its own slave environment. A master
environment is used to tick the slave environments and for visualisation as a form of pyplot animation.