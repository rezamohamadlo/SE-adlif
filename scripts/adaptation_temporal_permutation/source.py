"""Authenticated Git-backed imports; never overwrite the integrated checkout."""
import hashlib
import importlib.abc
import importlib.util
import linecache
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def normalized_hash(value):
    return hashlib.sha256(value.replace(b'\r\n', b'\n')).hexdigest()


class FrozenSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, revision):
        if git('rev-parse', revision + '^{commit}').decode().strip() != revision:
            raise ValueError('Scientific revision must be a full, available commit identity')
        self.revision = revision
        paths = git('ls-tree', '-r', '--name-only', revision).decode().splitlines()
        self.modules = {}
        for path in paths:
            if path.endswith('.py') and not path.startswith(('results/', 'tests/', 'archive/')):
                module = path[:-3].replace('/', '.')
                package = module.endswith('.__init__')
                self.modules[module[:-9] if package else module] = (path, package)
        self.loaded = {}

    def read(self, path):
        return git('show', self.revision + ':' + path)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            return importlib.util.spec_from_loader(fullname, self, is_package=self.modules[fullname][1])
        return None

    def create_module(self, spec):
        return None

    def get_source(self, fullname):
        return self.read(self.modules[fullname][0]).decode('utf-8-sig')

    def exec_module(self, module):
        path, package = self.modules[module.__name__]
        data = self.read(path)
        # __file__ resolves artifacts against the live repository, not a temp CWD.
        module.__file__ = str(ROOT / path)
        # TorchScript/inspect must see the same frozen text as compile(), rather
        # than reading a newer integrated file with the same portable pathname.
        linecache.cache[module.__file__] = (len(data), None,
            data.decode('utf-8-sig').splitlines(keepends=True), module.__file__)
        if package:
            module.__path__ = [str((ROOT / path).parent)]
        self.loaded[path] = normalized_hash(data)
        exec(compile(data, module.__file__, 'exec'), module.__dict__)

    def install(self):
        occupied = sorted(set(self.modules).intersection(sys.modules) - {'scripts'})
        if occupied:
            raise ValueError('Use a fresh module-mode process; scientific modules already imported: ' + ', '.join(occupied))
        sys.meta_path.insert(0, self)

    def authenticate(self, expected):
        audit = {}
        for path, checksum in expected.items():
            data = self.read(path)
            if normalized_hash(data) != checksum:
                raise ValueError('Frozen scientific source identity differs: ' + path)
            current = ROOT / path
            audit[path] = dict(expected=checksum, frozen_matches=True,
                integrated_matches=current.is_file() and normalized_hash(current.read_bytes()) == checksum)
        return audit
