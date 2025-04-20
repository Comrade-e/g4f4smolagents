from types import FunctionType
from typing import Union

import g4f
from smolagents import CodeAgent, DuckDuckGoSearchTool, tool

from src.main import G4fModel

@tool
def solve_quadratic_equation(a: float, b: float, c: float) -> tuple:
    """
    Solves quadratic equation (ax^2+bx+c=0) by the a, b, c coefficients. It's preferred to make the result perceived by
     human well without parenthesis, but you must describe all the roots.

    Values: A tuple with roots of the equation

    Args:
        a (float): A real first coefficient.
        b (float): A real second coefficient.
        c (float): A real third coefficient

    """
    D = b ** 2 - 4 * a * c
    if D < 0:
        return ()
    x1 = (-1 * b - D ** 0.5) / (2*a)
    if D == 0:
        return (x1)
    x2 = (-1 * b + D ** 0.5) / (2*a)
    return (x1, x2)

@tool
def solve_linear_equation(a: float, b: float) -> tuple:
    '''
    Solves linear equation (ax+b=0) by the a, b coefficients. It's preferred to make the result perceived by
     human as well as possible, without tuple parenthesis.

    Values: A tuple with one root if any, or empty tuple

    Args:
        a (float): A angle coefficient of the equation.
        b (float): A free coefficient.
    '''
    if a == 0:
        return ()
    res = (-1 * b) / a
    return tuple([res])

@tool
def solve_by_bisection(fn: FunctionType, start: float, stop: float, epsilon: float) -> Union[float, None]:
    '''
        Solves any function's roots in determined interval using bisective calculations. Note that root returned from this
        function may not be the only one in the definition domain, so you may need to call it several times with
        different interval borders, including negative numbers. Even if the signs of values of function on interval's borders may be different,
        it's not guaranteed that there's no another root in this interval. If the signs of function values on borders become
        same or an error with definition domain exceed occurs, function returns nothing. Note that increasing epsilon
        parameter would make result more precise, but would increase execution time.

        Values: A root of the function in the interval, or None if something gone wrong.

        Args:
            fn (FunctionType): A float-to-float function to find its roots.
            start (float): The left border of interval (less).
            stop (float): the right border of interval (bigger). Must have different sign from start.
            epsilon (float): Precision of the answer, increasing influences execution time.
        '''
    try:
        if fn(start) * fn(stop) < 0:
            cur = stop
            while abs(fn(cur)) > epsilon:
                cur = (start + stop) / 2
                if fn(cur) * fn(start) < 0:
                    stop = cur
                else:
                    start = cur
            return cur
        else:
            return None
    except (TypeError, ZeroDivisionError):
        return None



agent = CodeAgent(tools=[DuckDuckGoSearchTool(), solve_quadratic_equation, solve_linear_equation, solve_by_bisection],
                  model=G4fModel('gpt-4o-mini'))
agent.run('solve zeta(x)=2')