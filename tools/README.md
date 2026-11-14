# Your tools

Each `.py` file here is one tool. Enable a tool by adding its file name, without
`.py`, to `[tools].enabled` in `config/arietta_voice.local.toml`, or switch it on
from the console's Tools page. Restart the voice runtime afterwards.

Only this README and `example_dice.py` are tracked by Git; your own tools stay
local. The folder can be moved with `[tools].directory`.

A tool is a class with a name, a description of when to use it, a JSON-schema
description of its arguments, and a `run` method:

```python
from arietta_voice.tools import BaseTool, ToolResult


class LightsTool(BaseTool):
    name = "lights"  # must match the file name, lights.py
    description = "the user wants to know whether the office lights are on"
    parameters = {
        "type": "object",
        "properties": {"room": {"type": "string", "enum": ["office", "kitchen"]}},
        "required": ["room"],
    }

    def run(self, arguments):
        state = read_light_state(arguments["room"])  # your code
        return ToolResult(content=f"The {arguments['room']} lights are {state}.")


def create_tool(context):
    return LightsTool(context)
```

- **`content`** is information for the model to use in its answer.
- **`speak`** is a finished answer. When a trigger runs the tool, `speak` is said
  directly without asking the model.
- **`triggers`** are optional regular expressions. When one matches a request,
  the tool runs at once; named groups such as `(?P<room>\w+)` become arguments.
  Triggers are tried in the order tools are listed in `[tools].enabled`.
- **`model_callable = False`** keeps a trigger-only tool out of the model's list.
- **`timeout_s`** (default 10) bounds how long `run` may take.
- **`consequential = True`** marks a tool that changes something in the world.
  The planned agent loop will ask before running these.

Supported argument types are string, integer, number and boolean, with `enum`,
`minimum`, `maximum` and `maxLength`. Arguments are checked against the schema
before `run` is called, and anything the model sends that does not fit is
rejected. `self.context.config` gives access to the configuration.

The console's Tools page can test a tool with a sample request or with
arguments, and validate it, without enabling it.
