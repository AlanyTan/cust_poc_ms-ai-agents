import unittest
from unittest.mock import patch, MagicMock
import doctest
import importlib
import os
import sys
import pkgutil
from pathlib import Path
import importlib


def run_doctests(module_file: str = '') -> doctest.TestResults:
    """
    Run all the doctests in the module.
    """
    current_path = os.path.dirname(__file__)
    relative_path_to_modeule = os.path.relpath(module_file, current_path)
    package_name = __package__
    module_path = ("." if package_name else '') + relative_path_to_modeule.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=__package__)
    if not module:
        raise ImportError(f"Module {module_path} could not be imported.")
    # Check if the module has a __doc__ attribute
    if hasattr(module, '__doc__'):
        # Run the doctests in the module
        return doctest.testmod(module, verbose=True, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    else:
        raise AttributeError(f"Module {module_path} does not have a __doc__ attribute.")


class TestMainClass(unittest.IsolatedAsyncioTestCase):

    def test_main_executes_fine(self):
        """
        Test that when main.py is executed, it calls the package's main function.
        This tests the default behavior when no arguments are provided.
        """
        # Mock the package's main function to avoid running the actual application
        with patch('runpy.run_module') as mock_run_module, \
                patch('importlib.import_module') as mock_import_module:

            mock_run_module.return_value = None

            # Create a mock module that has the main function
            mock_import_module.return_value = MagicMock()

            # Mock sys.argv to simulate running `python main.py` with no arguments
            with patch('sys.argv', ['main.py']):
                # Import the main module and run it
                import runpy
                sys.argv = ['main.py']
                runpy.run_path('main.py', run_name='__main__')

            # Verify that mock_run_module was called and the main function was called

            mock_run_module.assert_called_once()

    def test_main_with_function_arguments(self):
        """
        Test that main.py correctly handles command line arguments.
        """
        with patch('importlib.import_module') as mock_import:

            # Mock the imported module and its test function
            mock_module = MagicMock()
            mock_module.test_function = MagicMock()
            mock_module.test_function.return_value = "test_result"
            mock_import.return_value = mock_module

            # Import the main module and run it
            import runpy
            sys.argv = ['main.py', 'test_module', '--function', 'test_function', 'test_result']
            runpy.run_module('main', run_name='__main__')

            # Verify the module was imported and function was called
            mock_import.assert_called()
            mock_module.test_function.assert_called_once()
            mock_module.test_function.assert_called_with("test_result")

    def test_main_with_arguments_but_function(self):
        """
        Test that main.py correctly handles command line arguments.
        """
        with patch('importlib.import_module') as mock_import:

            # Mock the imported module and its test function
            mock_module = MagicMock()
            mock_module.test_function = MagicMock()
            mock_module.test_function.return_value = "test_result"
            mock_import.return_value = mock_module

            # Import the main module and run it
            import runpy
            sys.argv = ['main.py', 'unittest.mock']
            runpy.run_module('main', run_name='__main__')

            # Verify the module was imported and function was called
            mock_module.test_function.assert_not_called()
            mock_import.assert_not_called()


class Code:
    """
    A class to hold the first line number of a doctest.
    This is used to ensure that doctests are run in the correct context.
    """

    def __init__(self, filename: str, first_line_no: int = 1):
        self.co_firstlineno = first_line_no
        self.co_filename = filename

    def __repr__(self):
        return f"Code({self.co_filename}, {self.co_firstlineno})"


class Func:
    """
    A class to hold the first line number of a doctest.
    This is used to ensure that doctests are run in the correct context.
    """

    def __init__(self, filename: str, first_line_no: int = 1):
        self.__code__ = Code(filename, first_line_no)

    def __repr__(self):
        return f"Func(Code({self.__code__.co_filename}, {self.__code__.co_firstlineno}))"


# Patch DocTestCase to hack the module path issue for VSCode testing grouping
original_init = doctest.DocTestCase.__init__


def patched_init(self, test, optionflags=0, setUp=None, tearDown=None, checker=None):
    # testing hacking the names
    if hasattr(test, 'name'):
        module_hierarchy = test.name.split('.')
        # Check if the test name is a module path
        hierarchy_level = 1
        module_name = ''
        while (hierarchy_level := hierarchy_level + 1) < len(module_hierarchy):
            try_module_name = ".".join(module_hierarchy[:hierarchy_level])
            try:
                if sys.modules.get(try_module_name):
                    module_name = try_module_name
            except Exception as e:
                break

        test_name = test.name.removeprefix(module_name + ".")
        if test.filename.endswith("__init__.py"):
            # If the test is in an __init__.py file, we need to adjust the module name
            test.name = module_name + ".__init__._module_." + test_name
        elif test.filename.endswith(f"/{test_name}.py"):
            test.name = module_name + "." + test_name + "._module_./"
        elif not "." in test_name:
            test.name = module_name + "._module_." + test_name
    test.firstlineno = test.lineno
    test.__func__ = Func(test.filename, test.lineno)
    # Call original init
    original_init(self, test, optionflags, setUp, tearDown, checker)


# Apply the patch
doctest.DocTestCase.__init__ = patched_init


def load_tests(loader, tests, ignore) -> unittest.TestSuite:
    """
    standard hook for unittest to load tests
    Load all doctests from the current package and its subpackages.
    """

    current_dir = os.path.dirname(__file__)
    for _, module_name, _ in pkgutil.walk_packages([current_dir]):
        # package_name = module_name.split(".")[0]
        # Fix duplicated levels in module names
        # if module_name.startswith(package_name + "."):
        #     module_name = module_name[len(package_name) + 1:]
        try:
            module = importlib.import_module(module_name)
            module.__name__ = module_name  # Ensure the module name is correctly set to avoid duplication
        except Exception as e:
            print(f"Error importing {module_name}: {e}")
            continue

        tests.addTests(doctest.DocTestSuite(module, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS))
    return tests


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
