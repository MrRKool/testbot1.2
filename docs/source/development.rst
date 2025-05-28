Development
==========

Development Setup
--------------

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

3. Install development dependencies:

.. code-block:: bash

   pip install -r requirements-dev.txt

4. Install the package in development mode:

.. code-block:: bash

   pip install -e .

Code Style
---------

We use the following tools for code quality:

1. Black
~~~~~~~

For code formatting:

.. code-block:: bash

   black .

2. Flake8
~~~~~~~~

For linting:

.. code-block:: bash

   flake8 .

3. MyPy
~~~~~~

For type checking:

.. code-block:: bash

   mypy .

4. isort
~~~~~~~

For import sorting:

.. code-block:: bash

   isort .

Or use the Makefile:

.. code-block:: bash

   make format
   make lint

Testing
------

1. Running Tests
~~~~~~~~~~~~~~

Run all tests:

.. code-block:: bash

   pytest

With coverage:

.. code-block:: bash

   pytest --cov=.

Or using the Makefile:

.. code-block:: bash

   make test

2. Writing Tests
~~~~~~~~~~~~~~

Test files should:
- Be named ``test_*.py``
- Be placed in the ``tests`` directory
- Use pytest fixtures
- Include docstrings
- Test both success and failure cases

Example test:

.. code-block:: python

   def test_price_fetcher_get_price():
       """Test getting price from exchange."""
       fetcher = PriceFetcher()
       price = fetcher.get_price("BTCUSDT")
       assert isinstance(price, float)
       assert price > 0

3. Test Coverage
~~~~~~~~~~~~~~

Maintain test coverage above 80%:

.. code-block:: bash

   pytest --cov=. --cov-report=term-missing

Documentation
-----------

1. Docstrings
~~~~~~~~~~~

Use Google style docstrings:

.. code-block:: python

   def function(param1: str, param2: int) -> bool:
       """Short description.

       Longer description if needed.

       Args:
           param1: Description of param1.
           param2: Description of param2.

       Returns:
           Description of return value.

       Raises:
           ExceptionType: Description of when this exception is raised.
       """
       pass

2. Building Documentation
~~~~~~~~~~~~~~~~~~~~~~~

Build the documentation:

.. code-block:: bash

   make docs

3. Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~~~

- ``docs/source/installation.rst``: Installation guide
- ``docs/source/configuration.rst``: Configuration guide
- ``docs/source/usage.rst``: Usage guide
- ``docs/source/api.rst``: API reference
- ``docs/source/development.rst``: Development guide
- ``docs/source/contributing.rst``: Contributing guide

Version Control
-------------

1. Git Workflow
~~~~~~~~~~~~~

- Use feature branches
- Write descriptive commit messages
- Keep commits atomic
- Use pull requests for code review

2. Commit Messages
~~~~~~~~~~~~~~~

Format:
- First line: Summary (50 chars or less)
- Blank line
- Detailed description (72 chars or less)

Example:
```
feat: add price fetcher with caching

- Implement PriceFetcher class
- Add caching mechanism
- Add rate limiting
- Add error handling
```

3. Branch Naming
~~~~~~~~~~~~~~

Format: ``type/description``

Types:
- ``feat/``: New feature
- ``fix/``: Bug fix
- ``docs/``: Documentation
- ``style/``: Formatting
- ``refactor/``: Code restructuring
- ``test/``: Adding tests
- ``chore/``: Maintenance

Example: ``feat/price-fetcher``

Continuous Integration
--------------------

The project uses GitHub Actions for CI:

1. Workflows
~~~~~~~~~~~

- ``test.yml``: Run tests
- ``lint.yml``: Check code style
- ``docs.yml``: Build documentation
- ``deploy.yml``: Deploy to production

2. Running Locally
~~~~~~~~~~~~~~~~

Install act:

.. code-block:: bash

   brew install act  # Mac
   # or
   choco install act-cli  # Windows

Run workflows:

.. code-block:: bash

   act -j test
   act -j lint
   act -j docs

Debugging
--------

1. Logging
~~~~~~~~~

Use the logging module:

.. code-block:: python

   import logging

   logger = logging.getLogger(__name__)
   logger.debug("Debug message")
   logger.info("Info message")
   logger.warning("Warning message")
   logger.error("Error message")

2. Debug Mode
~~~~~~~~~~~

Run in debug mode:

.. code-block:: bash

   python main.py --debug

3. Profiling
~~~~~~~~~~

Use cProfile:

.. code-block:: bash

   python -m cProfile -o output.prof main.py

Analyze results:

.. code-block:: bash

   python -m pstats output.prof 