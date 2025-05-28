Installation
============

Requirements
-----------

- Python 3.8 or higher
- pip (Python package installer)
- virtualenv (recommended)

Installation Steps
----------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/yourusername/trading-bot.git
   cd trading-bot

2. Create and activate a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows

3. Install the package:

.. code-block:: bash

   pip install -e .

4. Install development dependencies:

.. code-block:: bash

   pip install -r requirements-dev.txt

5. Configure environment variables:

.. code-block:: bash

   cp .env.example .env
   # Edit .env with your settings

6. Validate the configuration:

.. code-block:: bash

   python utils/config_validator.py

Verification
-----------

To verify the installation, run:

.. code-block:: bash

   python -c "import trading_bot; print(trading_bot.__version__)"

You should see the version number printed.

Common Issues
------------

1. Missing dependencies
~~~~~~~~~~~~~~~~~~~~~~

If you encounter missing dependency errors, try:

.. code-block:: bash

   pip install --upgrade pip
   pip install -r requirements.txt

2. Permission errors
~~~~~~~~~~~~~~~~~~

If you get permission errors, try:

.. code-block:: bash

   pip install --user -e .

3. Virtual environment issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have issues with the virtual environment:

.. code-block:: bash

   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -e . 