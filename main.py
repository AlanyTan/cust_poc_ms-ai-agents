"""Entry point for debugging and executing functions or methods within the package.

This script allows dynamic execution of a specified module and function or class method
based on command-line arguments. It supports both synchronous and asynchronous functions.

Usage:
    python -m rename_me_to_your_project_name <module> --function <function_or_class.method> [args...]

Args:
    1st: The module to load (e.g., 'src.config').
    2nd & 3rd: --function function_name or method_name to call
    The rest: Additional arguments to pass to the function or method.

Behavior:
    - If the specified function is a coroutine, it will be executed in an event loop.
    - If no arguments are provided, the default behavior is to execute the module similar to `python -m <module>`.
    - Note, if you want to try an async function, you have to use the --function option, and provide the async function name.

Returns:
    The result of the executed function or method, printed to stdout.

Raises:
    AttributeError: If the specified attribute or callable is not found.
"""

import sys
import importlib
import os
from pathlib import Path
import runpy

if __name__ == "__main__":
    parent_ = (__package__ + ".") if __package__ else ""
    packages = [f"{parent_}{p.name}" for p in
                Path(__file__).parent.iterdir()
                if p.is_dir() and (p / '__init__.py').exists()]

    args = sys.argv[1:]
    if args:
        module_name = args[0]
        if module_name.endswith('.py'):
            # provided modulename is a .py file instead of a module name, we need to derive the module name from the file path
            module_name = module_name.removeprefix(os.path.abspath(os.curdir))[1:-3].replace('/', '.')

        f_idx = args.index('--function') if '--function' in args else -1
        if 0 < f_idx < len(args) - 1:
            # if arguments are provided, the first argument is the function or method to call
            module = importlib.import_module(module_name, package=parent_)
            func_path = args[f_idx+1].split('.')

            current = module
            for index, part in enumerate(func_path):
                if not hasattr(current, part):
                    raise AttributeError(f"Attribute '{part}' not found in {current}")
                attr = getattr(current, part)
                if isinstance(attr, type) and index < len(func_path) - 1:
                    # If attr is a class and we're not at the final part, instantiate it
                    current = attr()
                else:
                    current = attr
            if callable(current):
                import inspect
                if inspect.iscoroutinefunction(current):
                    # If the function is a coroutine, run it in an event loop
                    import asyncio
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(current(*args[3:]))
                    print(result)
                else:
                    print(current(*args[3:]))
            else:
                raise AttributeError(f"{args} is not callable")
        else:
            # no --function flag is provided, just run this module
            runpy.run_module(module_name, run_name="__main__")
    else:
        # Default behavior if no arguments are provided is to
        # run the first package that contains __main__.py
        # or if none package are found to have __main__.py, the first that contains main.py
        packages = [f"{parent_}{p.name}" for p in
                    Path(__file__).parent.iterdir()
                    if p.is_dir() and ((p / '__init__.py').exists() and (p / '__main__.py').exists() or (p / 'main.py').exists())]
        if packages:
            module_name = packages[0]
            runpy.run_module(module_name, run_name="__main__")
        else:
            print("No suitable package found.")
