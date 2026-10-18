"""工具文本协议解析测试。"""
import json

from swift_agent.tool_parser import (
    parse_tool_calls, is_submit, render_observation,
    render_error, BUILTIN_TOOL_NAMES,
)


def _call(name, arguments):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


def test_parse_single_bash():
    calls = parse_tool_calls(_call('bash', {'command': 'ls -la'}))
    assert len(calls) == 1
    assert calls[0].name == 'bash'
    assert calls[0].arguments['command'] == 'ls -la'


def test_parse_multiple_and_fenced():
    text = (
        _call('read_file', {'path': 'a.py'})
        + '\nthinking\n'
        + '```tool_call\n' + json.dumps({'name': 'submit', 'arguments': {}}) + '\n```'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ['read_file', 'submit']
    assert is_submit(calls)


def test_malformed_is_skipped():
    text = (
        '<tool_call>not valid json</tool_call>\n'
        + _call('bash', {'command': 'echo x'})
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1 and calls[0].name == 'bash'


def test_builtin_names():
    assert {'bash', 'read_file', 'write_file', 'submit'} <= BUILTIN_TOOL_NAMES


def test_render_observation():
    ok = render_observation('bash', 'done')
    assert '<tool_result' in ok and 'status="result"' in ok
    err = render_error('bash', 'boom')
    assert 'status="error"' in err and 'boom' in err
