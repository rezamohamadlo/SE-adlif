"""Portable artifact paths; historical records are normalized in memory only."""
import argparse
import json
import posixpath
from pathlib import Path, PureWindowsPath

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
PATH_KEYS = {'logdir', 'launcher_logdir', 'datadir', 'cachedir', 'data_path',
             'cache_root', 'ckpt_path', 'init_ckpt_path', 'checkpoint',
             'sweep_checkpoint', 'sweep_output', 'search_output', 'evaluation_file'}


def portable_path(value, root=None):
    """Map only registered checkout prefixes; leave external paths explicit."""
    if value is None or not isinstance(value, (str, Path)):
        return value
    root = Path(root or ROOT)
    text = str(value).replace('\\', '/')
    if not text:
        return text
    # Keep both spellings: Windows temp directories may use 8.3 short names.
    roots = [root.absolute().as_posix(), root.resolve().as_posix()]
    aliases = ROOT / 'config/path_aliases.yaml'
    if aliases.exists():
        roots += list(OmegaConf.load(aliases).get('repository_roots', []))
    for prefix in sorted(roots, key=len, reverse=True):
        prefix = str(prefix).replace('\\', '/').rstrip('/')
        # Windows checkout roots are case insensitive; POSIX roots are not.
        actual, expected = (text.casefold(), prefix.casefold()) if PureWindowsPath(prefix).drive else (text, prefix)
        if actual == expected or actual.startswith(expected + '/'):
            suffix = posixpath.normpath(text[len(prefix):].lstrip('/') or '.')
            if suffix == '..' or suffix.startswith('../'):
                raise ValueError(f'Path escapes registered checkout: {value}')
            return suffix
    if PureWindowsPath(text).is_absolute() or text.startswith('/'):
        return text
    if PureWindowsPath(text).drive:
        raise ValueError(f'Ambiguous drive-relative path: {value}')
    return posixpath.normpath(text)


def local_path(value, root=None, must_exist=False):
    root = Path(root or ROOT).resolve()
    portable = portable_path(value, root)
    if portable is None:
        return None
    if PureWindowsPath(portable).is_absolute() and not Path(portable).is_absolute():
        raise ValueError(f'Unmapped foreign path: {value}; add its checkout root to config/path_aliases.yaml')
    path = Path(portable)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f'Missing artifact: {path} (recorded as {value}). Copy the exact file; no checkpoint substitution is performed.')
    return path


def portable_config(value, root=None, context=()):
    """Normalize known path fields, never arbitrary text or model settings."""
    if isinstance(value, dict):
        return {key: (portable_path(item, root) if key in PATH_KEYS or
                      (key == 'dir' and 'hydra' in context) else
                      portable_config(item, root, (*context, key)))
                for key, item in value.items()}
    if isinstance(value, list):
        return [portable_config(item, root, context) for item in value]
    return value


def relocate_checkpoint_callbacks(checkpoint, root=None):
    """Relocate Lightning checkpoint callback paths without touching tensors."""
    for state in checkpoint.get('callbacks', {}).values():
        if not isinstance(state, dict):
            continue
        for key in ('dirpath', 'best_model_path', 'last_model_path', 'kth_best_model_path'):
            if state.get(key):
                state[key] = str(local_path(state[key], root))
        if isinstance(state.get('best_k_models'), dict):
            state['best_k_models'] = {str(local_path(path, root)): score
                                      for path, score in state['best_k_models'].items()}
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', type=Path, required=True, help='Print a portable view without changing the saved record')
    args = parser.parse_args()
    obj = OmegaConf.to_container(OmegaConf.load(local_path(args.inspect, must_exist=True)), resolve=True)
    print(json.dumps(portable_config(obj), indent=2))


if __name__ == '__main__':
    main()
