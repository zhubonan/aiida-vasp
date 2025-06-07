"""
This module contains the parsing interfaces to ``parsevasp`` used to parse standard streams
for VASP related notification, warnings, and errors.
"""

# pylint: disable=abstract-method
import re

from parsevasp.stream import Stream

from aiida_vasp.parsers.content_parsers.base import BaseFileParser


class StreamParser(BaseFileParser):
    """
    Parser used for parsing errors and warnings from VASP.

    :ivar DEFAULT_SETTINGS: Default settings for quantities to parse.
    :ivar PARSABLE_QUANTITIES: The quantities that can be parsed.
    """

    DEFAULT_SETTINGS = {'quantities_to_parse': ['notifications']}

    PARSABLE_QUANTITIES = {
        'notifications': {
            'inputs': [],
            'name': 'notifications',
            'prerequisites': [],
        }
    }

    def _init_from_handler(self, handler):
        """
        Initialize a ``parsevasp`` object of ``Stream`` using a file like handler.

        :param handler: A file like object that provides the necessary standard stream content to be parsed.
        :type handler: file-like object
        """

        # First get any special config from the parser settings, else use the default
        stream_config = None
        history = False
        if self._settings is not None:
            stream_config = self._settings.get('stream_config', None)
            history = self._settings.get('stream_history', False)
        try:
            self._content_parser = Stream(
                file_handler=handler, logger=self._logger, history=history, config=stream_config
            )
        except SystemExit:
            self._logger.warning('Parsevasp exited abnormally.')

        # Parse additional errors
        handler.seek(0)
        lines = handler.read().split('\n')
        self.captured_error_box = parse_vasp6_error_box(lines)

    @property
    def notifications(self):
        """
        Fetch the notifications that VASP generated.

        :returns: A list of all notifications from VASP. Each entry is a dict with the keys ``name``,
            ``kind``, ``message``
            and ``regex`` containing the name of the message, what kind it is (``ERROR`` or ``WARNING``),
            a description
            of the notification, and the regular expression detected as string values.
        :rtype: list
        """

        # ``parsevasp`` returns ``VaspStream`` objects, which we cannot serialize. We could serialize this, but
        # eventually, we would like to move to a dedicated node for the notifications with its own data class.
        # This should be fixed in AiiDA core and coordinated across many plugins. For now, we convert the relevant info
        # into dict entries explicitly.
        notifications = []
        for item in self._content_parser.entries:
            if isinstance(item.regex, type(re.compile(''))):
                regex = item.regex.pattern
            else:
                regex = item.regex
            notifications.append({'name': item.shortname, 'kind': item.kind, 'message': item.message, 'regex': regex})

        # Additional handling - this should be incorporated into parsevasp!
        if hasattr(self, 'captured_error_box') and self.captured_error_box:
            duplicated = False
            # Check if we found this error already
            for item in notifications:
                if any(re.match(item['regex'], line) for line in self.captured_error_box):
                    duplicated = True
                    break
            if not duplicated:
                notifications.append(
                    {
                        'name': 'generic_box_error',
                        'kind': 'error',
                        'message': '\n'.join(self.captured_error_box),
                        'regex': self.captured_error_box[0],
                    }
                )

        return notifications

    @property
    def errors(self):
        """
        Fetch the errors that VASP generated.

        :returns: A list of all errors from VASP. Each entry is a dict with the keys ``name``, ``kind``, ``message``
            and ``regex`` containing the name of the message, what kind it is (``ERROR``), a description
            of the error, and the regular expression detected as string values.
        :rtype: list
        """

        return [item for item in self._content_parser.entries if item.kind == 'ERROR']

    @property
    def warnings(self):
        """
        Fetch the warnings that VASP generated.

        :returns: A list of all warnings from VASP. Each entry is a dict with the keys ``name``, ``kind``, ``message``
            and ``regex`` containing the name of the message, what kind it is (``WARNING``), a description
            of the warning, and the regular expression detected as string values.
        :rtype: list
        """

        return [item for item in self._content_parser.entries if item.kind == 'WARNING']

    @property
    def has_entries(self):
        """
        Check if there are notifications from VASP present according to the config after parsing.

        :returns: ``True`` if notifications were detected, ``False`` otherwise.
        :rtype: bool
        """

        entries = self._content_parser.has_entries
        return entries

    @property
    def number_of_entries(self):
        """
        Find the number of unique notifications from VASP.

        :returns: The number of unique notification entries that VASP generated.
        :rtype: int
        """

        number_of_entries = len(self._content_parser)
        return number_of_entries


def parse_vasp6_error_box(lines: list) -> list:
    """
    Parse the error box from VASP6.
    :param lines: The lines to parse.
    :returns: Lines inside the captured error box
    """

    capture = False
    captured = []
    for line in lines:
        if line.startswith(r'|     EEEEEEE  R     R  R     R  OOOOOOO  R     R     ###     ###     ###     |'):
            capture = True
            continue
        elif capture and line.startswith(r'|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----'):
            capture = False
            break
        if capture:
            content = line[1:-1].strip()
            if content:
                captured.append(content)
    return captured
