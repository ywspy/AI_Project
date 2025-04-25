# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for evaluating programs proposed by the Sampler."""
from __future__ import annotations

from abc import abstractmethod, ABC
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any, Type
import profile

from implementation import code_manipulation
from implementation import programs_database


class _FunctionLineVisitor(ast.NodeVisitor):
    """Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
        """Collects the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None  # Check internal correctness.
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """Extracts the body of the generated function, trimming anything after it.

    RZ: the arg generated_code must only include the body of the generated function (an example is shown below):
    --------------
        a = item
        return a
    --------------
    Please note that the indentation is REQUIRED !!!
    """
    if not generated_code:
        return ''

    # If the code includes function header, we don't add the fake header
    if generated_code.strip().startswith('def '):
        # The code already contains the full function definition, so no need for a fake header
        code = generated_code
    else:
        # If the function body is provided alone, add a fake function header to parse it correctly
        code = f'def fake_function_header():\n{generated_code}'

    # Get the function name dynamically (from 'def ' part)
    function_name = code.split('(')[0].split()[1]  # Extract the function name from the definition line

    tree = None
    # We keep trying and deleting code from the end until the parser succeeds.
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            # Debugging log to see which part is causing the issue
            print(f"SyntaxError at line {e.lineno}: {e.text}")
            # Remove code after the error
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

            # If truncation removes too much of the body, stop
            if len(code.splitlines()) < 3:
                print("Warning: Code was truncated too much, returning empty string.")
                return ''

    if not code:
        return ''

    # Use _FunctionLineVisitor to locate the end of the function body
    visitor = _FunctionLineVisitor(function_name)  # Pass the dynamic function name
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    if not body_lines:
        return ''
    return '\n'.join(body_lines) + '\n\n'




def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """Returns the compiled generated function and the full runnable program.
    RZ: This function removes the code after the generated function body.
    """
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    return evolved_function, str(program)


class Sandbox(ABC):
    """Sandbox for executing generated code.
    RZ: Sandbox 1) avoids the generated code to be harmful (accessing the internet, take up too much RAM).
    2) stops the execution of the code in time (avoid endless loop).
    """

    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,  # refers to the full dataset, added by RZ
            test_input: str,  # refers to the current instance
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded.
        RZ: If the generated code (generated by LLM) is executed successfully, the output of this function
        """
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """Returns whether the generated function is calling an earlier version."""
    for name in code_manipulation.get_functions_called(program):
        # In `program` passed into this function the most recently generated
        # function has already been renamed to `function_to_evolve` (wihout the
        # suffix). Therefore, any function call starting with `function_to_evolve_v`
        # is a call to an ancestor function.
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False


class Evaluator:
    """Class that analyses functions generated by LLMs."""

    def __init__(
            self,
            database: programs_database.ProgramsDatabase,
            template: code_manipulation.Program,
            function_to_evolve: str,  # RZ: refers to the name of the function to evolve (e.g., 'priority')
            function_to_run: str,  # RZ: refers to the name of the function to run (e.g., 'evaluate')
            inputs: Sequence[Any],  # RZ: I guess this refers to the evaluate instance
            timeout_seconds: int = 30,
            sandbox_class: Type[Sandbox] = Sandbox
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()

    def analyse(
        self,
        sample: str,
        island_id: int | None,
        version_generated: int | None,
        **kwargs,
    ) -> None:
        """
        Compiles the sample into a full program, runs multi-objective scoring
        on all instances, and registers only the composite score.
        """
        # 1) Build the new program source
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve
        )

        # 2) Start timing
        start_time = time.time()

        # 3) Run sandbox once to get full multi-objective metrics
        metrics, ok = self._sandbox.run(
            program,
            self._function_to_run,
            self._function_to_evolve,
            self._inputs,
            None,  # no single test_input
            self._timeout_seconds,
            **kwargs,
        )

        # 4) Prepare scores_per_test with only composite
        if ok and "composite" in metrics:
            scores_per_test = {"composite": metrics["composite"]}
        else:
            scores_per_test = {"composite": -float("inf")}

        elapsed = time.time() - start_time

        # 5) Register in database
        if scores_per_test:
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=elapsed,
            )
        else:
            # If no scores, push to profiler if provided
            profiler: profile.Profiler = kwargs.get("profiler", None)
            if profiler:
                new_function.global_sample_nums = kwargs.get("global_sample_nums", None)
                new_function.score = None
                new_function.sample_time = kwargs.get("sample_time", None)
                new_function.evaluate_time = elapsed
                profiler.register_function(new_function)
