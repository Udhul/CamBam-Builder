"""Workspace-relative I/O under a trusted, locally controlled root."""
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile

MAX_XML_BYTES = 10 * 1024 * 1024


class DomainError(Exception):
    def __init__(self, code, message, field=None):
        super().__init__(message)
        self.code, self.field = code, field


class Workspace:
    def __init__(self, root):
        supplied = Path(root)
        if not supplied.is_absolute():
            raise ValueError("Workspace must be an existing absolute directory")
        self.root = supplied.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be an existing absolute directory")
        self.id = hashlib.sha256(os.path.normcase(str(self.root)).encode("utf-8")).hexdigest()

    def path(self, relative, *, destination=False):
        parts = relative.split("/")
        if (not relative.lower().endswith(".cb") or any(c in relative for c in "\\:\x00")
                or any(not p or p in (".", "..") or p[-1] in ". " or
                       any(ord(c) < 32 for c in p) or any(c in p for c in '<>"|?*') or
                       re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])", p.split(".")[0])
                       for p in parts)):
            raise DomainError("PATH_INVALID", "Use a safe workspace-relative .cb path", "path")
        current = self.root
        for index, part in enumerate(parts):
            current = current / part
            last = index == len(parts) - 1
            try:
                info = current.lstat()
            except FileNotFoundError:
                if last and destination:
                    break
                raise DomainError("PATH_INVALID", "Path or parent directory is unavailable", "path") from None
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise DomainError("PATH_INVALID", "Linked paths are unsupported", "path")
            if not last and not stat.S_ISDIR(info.st_mode):
                raise DomainError("PATH_INVALID", "Parent must be an ordinary directory", "path")
            if last:
                if destination:
                    raise DomainError("PATH_EXISTS", "Destination already exists", "path")
                self._regular(info)
        try:
            current.resolve().relative_to(self.root)
        except ValueError:
            raise DomainError("PATH_INVALID", "Path must stay in the workspace", "path") from None
        return current

    @staticmethod
    def _regular(info):
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise DomainError("PATH_INVALID", "Input must be an unlinked regular file", "path")

    def read(self, relative):
        path = self.path(relative)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            self._regular(os.fstat(stream.fileno()))
            self.path(relative)
            data = stream.read(MAX_XML_BYTES + 1)
        if len(data) > MAX_XML_BYTES:
            raise DomainError("LIMIT_EXCEEDED", "XML exceeds 10 MiB", "path")
        return data

    def stage_save(self, project, relative):
        destination = self.path(relative, destination=True)
        fd, temporary = tempfile.mkstemp(prefix=".cambam-mcp-", suffix=".cb", dir=destination.parent)
        os.close(fd)
        temporary = Path(temporary)
        try:
            project.save(str(temporary))
            with temporary.open("r+b") as stream:
                data = stream.read(MAX_XML_BYTES + 1)
                if not data or len(data) > MAX_XML_BYTES:
                    raise DomainError("LIMIT_EXCEEDED", "Saved XML must be 1 byte through 10 MiB")
                os.fsync(stream.fileno())
            return temporary, {"path": relative, "absolute_path": str(destination),
                               "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        except BaseException:
            self.cleanup(temporary)
            raise

    def serialize(self, project):
        """Serialize one bounded XML snapshot without publishing a workspace file."""
        from cambam_builder.cambam_writer import serialize_cambam_bytes

        data = serialize_cambam_bytes(project)
        if not data or len(data) > MAX_XML_BYTES:
            raise DomainError(
                "LIMIT_EXCEEDED", "Exported XML must be 1 byte through 10 MiB"
            )
        return data

    def publish(self, temporary, relative):
        destination = self.path(relative, destination=True)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise DomainError("PATH_EXISTS", "Destination already exists", "path") from None

    def cleanup(self, temporary):
        # Only a path returned by this instance's staging operation is passed here.
        path = Path(temporary)
        if not path.name.startswith(".cambam-mcp-"):
            raise ValueError("Not an adapter temporary")
        path.absolute().relative_to(self.root)
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError:
            return False
